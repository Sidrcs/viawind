"""
Module Name: viawind
Description: Package to perform visual impact assessment (via)
@author: Siddharth Ramavajjala
"""

import os
import math

import geopandas as gpd
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import pyproj
from osgeo import gdal

import numpy as np
from numpy import int16, int32

import rioxarray as rxr
from shapely import Point, Polygon
from shapely import box
from shapely.geometry import Point, MultiPolygon

import rasterio as rio
from rasterio.plot import show,show_hist
from rasterio.merge import merge
from rasterio.enums import Resampling, MergeAlg
from rasterio import features


class Calc_Visual_Impact:
    """Perform via using wind turbine locations and Digital Elevation Model (DEM)"""
    def __init__(self, windturbine_fpath, dem_fpath):
        self.windturbine_fpath = windturbine_fpath
        self.dem_fpath = dem_fpath

    def read_windturbine_file(self):
        """Function to read US Wind Turbine dataset"""
        if self.windturbine_fpath.endswith(".shp"):
            gdf = gpd.read_file(f"{self.windturbine_fpath}")
            gdf = gdf.to_crs(3857)
            return gdf
        if self.windturbine_fpath.endswith(".csv"):
            gdf = gpd.read_file(f"{self.windturbine_fpath}")
            gdf["geometry"] = gdf.apply(lambda x: Point(x['xlong'], x['ylat']), axis=1)
            gdf = gdf.set_crs(4326)
            gdf = gdf.to_crs(3857)
            return gdf
        return None
    
    def read_dem(self):
        """Function to read, reproject and return reproject raster path"""
        dem = rxr.open_rasterio(f"{self.dem_fpath}")
        dem_reproj = dem.rio.reproject(3857)
        reproj_fpath = self.dem_fpath.replace(".tif", "_reproj.tif")
        dem_reproj.to_raster(reproj_fpath)
        return reproj_fpath
    
    def create_relative_viewshed(self, output_dir, height, height_name):
        """Function to create relative viewsheds based on wind turbine height"""
        gdf = self.read_windturbine_file()
        input_dem = self.read_dem()
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        # Create a relative viewshed for each point
        for idx, row in enumerate(gdf.itertuples(), start=1):
            x = row.geometry.x
            y = row.geometry.y
            # Create a unique output filename for each point with an index
            output_filename = os.path.join(output_dir, f'viewshed_{idx}_{height_name}.tif')
            # Run gdal_viewshed for each point
            command = f'gdal_viewshed -md {height} -ox {x} -oy {y} {input_dem} {output_filename}'
            os.system(command)
        fpath = os.path.join(os.getcwd(), output_dir)
        print(f'{height_name} viewshed created for {len(gdf)} points and saved at {fpath}')
        # Close the input and output datasets
        input_dem = None
        gdf = None

    def create_relative_turbine_viewsheds(self):
        """Function to compute viewsheds for Turbine Blade End, Hub, Rotor Sweep """
        gdf = self.read_windturbine_file()
        blade_end_height = gdf["t_ttlh"][0]
        hub_height = gdf["t_hh"][0]
        rsa = gdf["t_rsa"][0]
        rotor_sweep_height = math.sqrt(float(rsa)/math.pi)

        self.create_relative_viewshed("viewsheds_blade_end", blade_end_height, "blade")
        self.create_relative_viewshed("viewsheds_hub", hub_height, "hub")
        self.create_relative_viewshed("viewsheds_rotor_sweep", rotor_sweep_height, "sweep")
    
    def reclass_relative_viewsheds(self, viewshed_folder_path, new_value, original_value=255):
        """Function to reclassify relative viewshed as per Palmer 2022"""
        for file in os.listdir(viewshed_folder_path):
            input_raster = os.path.join(viewshed_folder_path, file)
            # Open the input raster
            if input_raster.endswith('.tif'):
                with rio.open(input_raster) as src:
                    # Read the raster data
                    data = src.read(1)
                    # Reclassify the data
                    reclassified_data = np.where(data == original_value, new_value, data)
                    # Update the data type to ensure it matches the output
                    reclassified_data = reclassified_data.astype(src.profile['dtype'])
                    # Copy the metadata from the source raster
                    profile = src.profile
                    output_raster = file.replace('.tif','_rc.tif')
                    output_path = os.path.join(viewshed_folder_path, output_raster)
                    # Write the reclassified data to the output raster
                    with rio.open(output_path, 'w', **profile) as dst:
                        dst.write(reclassified_data, 1)
            print(f"Reclassification completed and outputs saved as {viewshed_folder_path}")
        return None
    
    def reclass_relative_turbine_viewsheds(self):
        """Function to reclassify relative viewsheds - Blade End, Turbine, Rotor Sweep as per Palmer 2022"""
        self.reclass_relative_viewsheds("viewsheds_blade_end", new_value=10)
        self.reclass_relative_viewsheds("viewsheds_hub", new_value=20)
        self.reclass_relative_viewsheds("viewsheds_rotor_sweep",new_value=30)

    
    def merge_viewshed_rasters(self, input_file_paths, output_folder_path):
        """Function to merge reclassified viewshed rasters"""
        # Open each input raster into list
        src_files_to_mosaic = [rio.open(file) for file in input_file_paths]
        # Merge rasters to hold maximum value in each pixel
        merged_data, merged_transform = merge(src_files_to_mosaic, method='max', resampling=Resampling.nearest)
        # Update metadata for the output raster
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({"driver": "GTiff", 
                        "height": merged_data.shape[1], 
                        "width": merged_data.shape[2], 
                        "transform": merged_transform})
        # Write the merged raster to the output file
        with rio.open(output_folder_path, "w", **out_meta) as dest:
            dest.write(merged_data)

    def perform_viewsheds_merge(self, output_dir="viewsheds_merged"):
        """Function to merge - blade end, hub, rotor sweep reclassed rasters"""
        viewsheds_blade_end = [file for file in os.listdir('viewsheds_blade_end') if "rc" in file]
        viewsheds_hub = [file for file in os.listdir('viewsheds_hub') if "rc" in file]
        viewsheds_rotor_sweep = [file for file in os.listdir('viewsheds_rotor_sweep') if "rc" in file]

        sorted_blade_end = sorted(viewsheds_blade_end, key=lambda x: int(x.split('_')[1]))
        sorted_hub = sorted(viewsheds_hub, key=lambda x: int(x.split('_')[1]))
        sorted_rotor_sweep = sorted(viewsheds_rotor_sweep, key=lambda x: int(x.split('_')[1]))

        for i in range(0,len(self.read_windturbine_file)):
            input_paths = [os.path.join('viewsheds_blade_end',sorted_blade_end[i]), 
                        os.path.join('viewsheds_hub',sorted_hub[i]),
                        os.path.join('viewsheds_rotor_sweep',sorted_rotor_sweep[i])]
            # Create the output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = f"{output_dir}/merged_{i+1}_viewshed.tif"
            self.merge_viewshed_rasters(input_paths, output_path)
        fpath = os.path.join(os.getcwd(), output_dir)
        print(f"Performed raster merge for {len(self.read_windturbine_file)*3} rasters and save to {fpath} ")
        
    def create_multiring_buffer(self, center_point, radii):
        """Function to create non-intersecting concentric buffers as MultiPolygon"""
        buffers = [center_point.buffer(r) for r in radii]
        buffer0 = buffers[0]
        buffer1 = buffers[1].difference(buffers[0])
        buffer2 = buffers[2].difference(buffers[1])
        buffer3 = buffers[3].difference(buffers[2])
        ring_buffers = [buffer0, buffer1, buffer2, buffer3]
        return MultiPolygon(ring_buffers)
    
    def kilo2meter(self, km):
        """Function to return radius in meters"""
        return km*1000

