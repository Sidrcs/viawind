"""
Module Name: viawind
Description: Package to perform visual impact assessment (via)
@author: Siddharth Ramavajjala
"""

import os
import shutil
import warnings
import math

import geopandas as gpd
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from shapely.geometry import MultiPolygon

import numpy as np
from numpy import int16

import rioxarray as rxr
import rasterio as rio
from rasterio.plot import show
from rasterio.merge import merge
from rasterio.enums import Resampling, MergeAlg
from rasterio import features


class CalcVisualImpact:
    """Class to Perform via using wind turbine locations and Digital Elevation Model (DEM)"""
    def __init__(self, windturbine_fpath, dem_fpath, dir_path):
        """
        Constructs all the necessary attributes for the CalcVisualImpact object

        Parameters
        ----------
        Windturbine_fpath: str
            Complete Wind Turbine file path [.csv/.shp]
        dem_fpath: str
            raster file path [DEM/DSM]
        dir_path:str
          Directory path to store outputs
        """
        try:
            # Required datasets to create an instance
            self.windturbine_fpath = windturbine_fpath
            self.dem_fpath = dem_fpath
            self.dir_path = dir_path
        except FileNotFoundError as e:
            print(f"{str(e)}")
        except IOError as e:
            print(f"{str(e)}")
        finally:
            os.chdir(dir_path)
            print("- Suggested to wind turbine data as in US Wind Turbine database: https://eerscmap.usgs.gov/uswtdb/data/")
            print("- Suggested to use a Digital Surface Model (DSM) instead of a Digital Elevation Model (DEM) for better results")
            print("- Try to use 1-arc second or 1/3-arc second DEM from https://apps.nationalmap.gov/downloader/")
            warnings.simplefilter(action="ignore", category=UserWarning)


    def read_windturbine_file(self):
        """
        Function to read US Wind Turbine dataset

        Returns
        ----------
        Reprojected GeoDataFrame object
        """
        try:
            # Reads shapefile and reprojects it to EPSG:3857
            if self.windturbine_fpath.endswith(".shp"):
                gdf = gpd.read_file(f"{self.windturbine_fpath}")
                gdf = gdf.to_crs(3857)
                return gdf
            # Reads CSV and reprojects it to EPSG:3857
            if self.windturbine_fpath.endswith(".csv"):
                df = pd.read_csv(f"{self.windturbine_fpath}")
                # Extract column names as list from dataframe
                col_list = list(df.columns.values)
                # Remove pre-existing geometry column
                if "geometry" in col_list:
                    df = df.drop(labels=["geometry"], axis=1)
                if ["xlong","ylat"] not in col_list:
                    raise ValueError("Latitude, Longitude columns has to be renamed as xlong, ylat")
                if df["xlong"][0] not in range(-180,181):
                    raise ValueError("Lon/Lat CRS is not EPSG:4326. Please use EPSG:4326 only")
                if ["t_ttlh","t_hh","t_rsa"] not in col_list:
                    raise AttributeError("t_ttlh: Turbine total height from ground to tip of a blade at its apex in meters (m)\nt_hh: Turbine hub height in meters (m)\nt_rsa: Turbine rotor sweep area in square meters (m2) columns are mandatory for analysis. Refer - https://eerscmap.usgs.gov/uswtdb/api-doc/#keyValue")
                # Populate geometry column with lat, lon data from US Wind Turbine database
                gdf = gpd.GeoDataFrame(data=df, geometry=gpd.points_from_xy(df['xlong'], df['ylat']))
                # Set CRS to 3D coordinate systems ESPG:4326
                gdf = gdf.set_crs(4326)
                # Reproject to EPSG:3857
                gdf = gdf.to_crs(3857)
                # Convert the height columns (meters) to float objects
                convert_dict = {"t_ttlh": float, "t_hh": float, "t_rsa": float}
                gdf = gdf.astype(convert_dict)
                return gdf
        except FileNotFoundError:
            print(f"Wind Turbine file not found: {self.windturbine_fpath}")
        except IOError as e:
            print(f"{str(e)}")
        except ValueError as e:
            print(f"{str(e)}. Populate data as in US Wind Turbine database format")
    
    def read_dem(self):
        """
        Function to read, reproject and deflate input DEM/DSM raster

        Returns
        ----------
        Reprojected raster file path"""
        # Check if the input raster file exists
        try:
            if not os.path.exists(self.dem_fpath):
                raise FileNotFoundError(f"Raster file not found: {self.dem_fpath}")
            if not os.path.exists("datasets"):
                os.mkdir("datasets")
            dem = rxr.open_rasterio(f"{self.dem_fpath}")
            # Reprojects raster to EPSG:3857
            dem_reproj = dem.rio.reproject(3857)
            reproj_fpath = "datasets/dem_reproj.tif"
            # Output raster to reproj_fpath
            dem_reproj.rio.to_raster(reproj_fpath)
            deflate_fpath = "datasets/dem_deflate.tif"
            # Run GDAL deflate for viewshed creation and execute the command
            command = f"gdal_translate -co COMPRESS=DEFLATE {reproj_fpath} {deflate_fpath}"
            os.system(command)
            return deflate_fpath
        finally:
            if "dem" in locals() and dem is not None:
                dem = None
            if "dem_reproj" in locals() and dem_reproj is not None:
                dem_reproj = None

    def check_dir(self, output_dir):
        """Function to remove an existing directory and create a fresh directory on re-runs

        Parameters
        ----------
        output_dir: str
            Directory file path
        """
        try:
            # Removes a non-empty folder
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            # Create the output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        except OSError as e:
            print(f"{str(e)}")
    
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
            command = f"gdal_viewshed -oz {height} -ox {x} -oy {y} {input_dem} {output_filename}"
            os.system(command)
        fpath = os.path.join(os.getcwd(), output_dir)
        print(f'{height_name} viewsheds created for {len(gdf)} points and outputs saved to {fpath}')
        # Close the input and output datasets
        input_dem = None
        gdf = None

    def create_relative_turbine_viewsheds(self):
        """Function to compute viewsheds for Turbine Blade End, Hub, Rotor Sweep"""
        # Remove the directors Rasterio IOErrors in future
        for output_dir in ["viewsheds_blade_end", "viewsheds_hub", "viewsheds_rotor_sweep"]:
            self.check_dir(output_dir)
        gdf = self.read_windturbine_file()
        # Preferred file format: US Wind Turbine Database
        blade_end_height = gdf["t_ttlh"][0]
        hub_height = gdf["t_hh"][0]
        rsa = gdf["t_rsa"][0]
        rotor_sweep_height = math.sqrt(float(rsa)/math.pi)
        # Creates relative viewsheds in the following directories
        self.create_relative_viewshed("viewsheds_blade_end", blade_end_height, "blade")
        self.create_relative_viewshed("viewsheds_hub", hub_height, "hub")
        self.create_relative_viewshed("viewsheds_rotor_sweep", rotor_sweep_height, "sweep")
    
    def reclass_relative_viewsheds(self, viewshed_folder_path, new_value, original_value=255):
        """Function to reclassify relative viewshed as per Palmer 2022
        
        Parameters
        ----------
        viewshed_folder_path: str
            Relative folder path of viewsheds
        new_value: int
            New reclassed value of 255
                Default: 10 for blade end
                         20 for turbine hub
                         30 for rotor sweep
        original_value: int, optional
            Original value of raster visibility. 
            Default: 255
        """
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
        fpath = os.path.join(os.getcwd(), viewshed_folder_path)
        print(f"Reclassification completed and outputs saved to {fpath}")
    
    def reclass_relative_turbine_viewsheds(self):
        """Function to reclassify relative viewsheds - Blade End, Turbine, Rotor Sweep as per Palmer 2022"""
        self.reclass_relative_viewsheds("viewsheds_blade_end", new_value=10)
        self.reclass_relative_viewsheds("viewsheds_hub", new_value=20)
        self.reclass_relative_viewsheds("viewsheds_rotor_sweep",new_value=30)
    
    def merge_viewshed_rasters(self, input_file_paths, output_folder_path):
        """Function to merge reclassified viewshed rasters
        
        Parameters
        ----------
        input_file_paths: list
            Input file paths list contains

        """
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

    def perform_viewsheds_merge(self):
        """Function to merge - blade end, hub, rotor sweep reclassed rasters"""
        output_dir = "viewsheds_merged"
        self.check_dir(output_dir)
        # Creates a list of raster file paths from relative viewshed file directories
        viewsheds_blade_end = [file for file in os.listdir("viewsheds_blade_end") if file.endswith("blade_rc.tif")]
        viewsheds_hub = [file for file in os.listdir("viewsheds_hub") if file.endswith("hub_rc.tif")]
        viewsheds_rotor_sweep = [file for file in os.listdir("viewsheds_rotor_sweep") if file.endswith("sweep_rc.tif")]
        # Creates a sorted list of aforementioned raster file paths
        sorted_blade_end = sorted(viewsheds_blade_end, key=lambda x: int(x.split('_')[1]))
        sorted_hub = sorted(viewsheds_hub, key=lambda x: int(x.split('_')[1]))
        sorted_rotor_sweep = sorted(viewsheds_rotor_sweep, key=lambda x: int(x.split('_')[1]))
        # Perform raster merge for each turbine
        gdf = self.read_windturbine_file()
        for i in range(0,len(gdf)):
            input_paths = [os.path.join('viewsheds_blade_end',sorted_blade_end[i]), 
                        os.path.join('viewsheds_hub',sorted_hub[i]),
                        os.path.join('viewsheds_rotor_sweep',sorted_rotor_sweep[i])]
            output_path = f"{output_dir}/merged_{i+1}_viewshed.tif"
            self.merge_viewshed_rasters(input_paths, output_path)
        fpath = os.path.join(os.getcwd(), output_dir)
        print(f"Performed raster merge for {len(gdf)*3} rasters and outputs saved to {fpath} ")
        
    def create_multiring_buffer(self, center_point, radii):
        """Function to create non-intersecting concentric buffers as MultiPolygon"""
        buffers = [center_point.buffer(r) for r in radii]
        buffer0 = buffers[0]
        # Extracts the non-intersecting parts of buffers
        buffer1 = buffers[1].difference(buffers[0])
        buffer2 = buffers[2].difference(buffers[1])
        buffer3 = buffers[3].difference(buffers[2])
        ring_buffers = [buffer0, buffer1, buffer2, buffer3]
        # Creates a MultiPolygon object from non-intersecting concentric buffers
        return MultiPolygon(ring_buffers)
    
    def kilo2meter(self, km):
        """Function to return radius in meters"""
        return km*1000
    
    def rasterize_concentric_buffers(self, ref_raster, buffers, buffer_val_list, output_fpath):
        """Function to perform rasterization of MultiRingBuffer and output raster"""
        raster = rio.open(ref_raster)
        # Burns the values from the buffer value list to rasterized buffers
        geom_value = ((geom,value) for geom, value in zip(buffers.geoms, buffer_val_list))
        # Rasterize vector using the shape and coordinate system of the raster
        rasterized = features.rasterize(geom_value,
                                        out_shape = raster.shape,
                                        transform = raster.transform,
                                        all_touched = False,
                                        fill = 0,   # background value
                                        merge_alg = MergeAlg.replace,
                                        dtype = int16)
        with rio.open(output_fpath, "w", driver = "GTiff", crs = raster.crs, transform = raster.transform,
            dtype = rio.uint8,
            count = 1,
            width = raster.width,
            height = raster.height) as dst:
            dst.write(rasterized, indexes = 1)
        # Close dataset
        raster = None

    def create_turbine_multiringbuffer_raster(self, buffer_val_list=None):
        """Function create MultiRingBuffer rasters for wind turbine locations"""
        output_dir="raster_buffers"
        self.check_dir(output_dir)
        # Create an empty list to store raster file paths
        raster_flist = []
        # Read both shapefile and DEM raster
        gdf = self.read_windturbine_file()
        ref_raster = self.read_dem()
         # Burn the following values under each pixel of the MultiRing Buffer
        if buffer_val_list is None:
            buffer_val_list = [1,2,3,4]
        if buffer_val_list is not None:
            if not isinstance(buffer_val_list, list):
                raise TypeError("Buffer value list parameter has to be a list")
        # Create a list of radii used to create MultiRingBuffer
        radii = [self.kilo2meter(0.8), self.kilo2meter(3.2), self.kilo2meter(8), self.kilo2meter(16.1)]
        # Iterate through each row and index position of the geodataframe
        for index, row in enumerate(gdf.itertuples()):
            # Store geometry of each wind turbine location
            point_geom = row.geometry
            # Function to create a MultiPolygon buffer for the radii
            buffers = self.create_multiring_buffer(point_geom, radii)
            output_raster_path = os.path.join(output_dir, f"rasterized_{index+1}_turbine_buffer.tif")
            # Functiom to rasterize the MultiRing buffer
            self.rasterize_concentric_buffers(ref_raster, buffers, buffer_val_list, output_raster_path)
            raster_flist.append(output_raster_path)
            # Close datasets
            point_geom = None
            buffers = None
        fpath = os.path.join(os.getcwd(), output_dir)
        print(f"MultiRing Buffer rasters created for {len(raster_flist)} points and outputs saved to {fpath}")
        return raster_flist
    
    def create_turbine_viz_prominence(self, input_paths, output_path):
        """Function to merge distance zone and visual exposure"""
        try:
            files = [rxr.open_rasterio(path) for path in input_paths]
            raster_sum = files[0] + files[1]
            raster_sum.rio.to_raster(output_path)
        except IOError as e:
            print(f"Unable open/locate: {str(e)}")
        except TypeError as e:
            print(f"File path has to be a string: {str(e)}")

    def perform_viz_prominence(self):
        """Function to create visual prominence rasters for each turbine location"""
        output_dir="visual_exposure"
        self.check_dir(output_dir)
        # Creates a list of file paths for merged viewshed and raster buffers directories
        merged_viewsheds = [file for file in os.listdir("viewsheds_merged") if file.endswith("viewshed.tif")]
        distance_zone_rasters = [file for file in os.listdir("raster_buffers") if file.endswith("turbine_buffer.tif")]
        # Sorted list of aforementioned folders to help run the loop as per index value
        sorted_merged_viewsheds = sorted(merged_viewsheds, key=lambda x: int(x.split('_')[1]))
        sorted_distance_zone_rasters = sorted(distance_zone_rasters, key=lambda x: int(x.split('_')[1]))
        gdf = self.read_windturbine_file()
        # Loops and merges index position rasters from both lists
        for i in range(0,len(gdf)):
            input_paths = [os.path.join("viewsheds_merged",sorted_merged_viewsheds[i]), 
                        os.path.join("raster_buffers",sorted_distance_zone_rasters[i])]
            output_path = f"{output_dir}/merged_{i+1}_vizexp.tif"
            self.create_turbine_viz_prominence(input_paths, output_path) 
        fpath = os.path.join(os.getcwd(), output_dir)
        print(f"Performed raw visual exposure calculation and outputs saved to {fpath}")

    def reclass_viz_prominence_rasters(self):
        """Function reclassify visual pro rasters as per table 3 (Palmer 2022)"""
        output_dir="visual_exposure"
        for file in os.listdir(output_dir):
            input_raster = os.path.join(output_dir, file)
            # Opens each input raster
            if input_raster.endswith(".tif") and "rc" not in input_raster:
                with rio.open(input_raster) as src:
                    # Reads the raster data
                    data = src.read(1)
                    rc = data.copy()
                    # Reclassify the data as per key:value pairs
                    reclass_dict = {
                        0: [0, 1, 2, 3, 4, 10, 14, 20, 30],
                        4: [11, 22, 33],
                        2: [12, 23, 34],
                        1: [13, 24],
                        7: [21],
                        6: [32],
                        10: [31]
                    }
                    # Unpack reclass_dict to seperate key, value and iterate through each pair 
                    for output_value, input_values in reclass_dict.items():
                        rc[np.isin(rc, input_values)] = output_value
                    # Replace the file name to reflect reclassed visual exposure
                    output_raster = file.replace('.tif', '_rc.tif')
                    output_path = os.path.join(output_dir, output_raster)
                    # Write the reclassified data to the output raster
                    with rio.open(output_path, 'w', driver=src.driver, height=src.height, width=src.width,
                                count=src.count, crs=src.crs, transform=src.transform, dtype=data.dtype) as dst:
                        dst.write(rc, 1)

        print(f"Reclassification complete for visual prominence and outputs saved to {os.path.join(os.getcwd(), output_dir)}")

    def reclass_meaningful_visibility_rasters(self):
        """Function to reclass visual exposure rasters to reflect meaningful visibility"""
        output_dir="visual_exposure"
        for file in os.listdir(output_dir):
            input_raster = os.path.join(output_dir, file)
            # Opens each input raster
            if 'rc' not in input_raster:
                with rio.open(input_raster) as src:
                    # Read the raster data
                    data = src.read(1)
                    rc = data.copy()
                    # Reclassify the data as per key:value pairs
                    reclass_dict = {
                        0: [0, 1, 2, 3, 4, 10, 14, 20, 30],
                        1: [11, 12, 13, 21, 22, 23, 24, 31, 32, 33, 34]
                    }
                    # Unpack reclass_dict to seperate key, value and iterate through each pair 
                    for output_value, input_values in reclass_dict.items():
                        rc[np.isin(rc, input_values)] = output_value
                    # Replace the file name to reflect reclassed meaningful visibility
                    output_raster = file.replace("vizexp.tif", "mv_rc.tif")
                    output_path = os.path.join(output_dir, output_raster)
                    # Write the reclassified data to the output raster
                    with rio.open(output_path, 'w', driver=src.driver, height=src.height, width=src.width,
                                count=src.count, crs=src.crs, transform=src.transform, dtype=data.dtype) as dst:
                        dst.write(rc, 1)
        print(f"Reclassification complete for meaningful visibility and outputs saved to {output_dir}")

    def create_cumulative_rasters(self, input_paths, output_path):
        """Function to perform raster sum of input rasters"""
        # Initialize raster_sum
        raster_sum = None
        # Open each raster file and accumulate values
        for file in input_paths:
            raster = rxr.open_rasterio(file)
            if raster_sum is None:
                raster_sum = raster
            else:
                raster_sum += raster
        # Write the cumulative raster to the output path
        raster_sum.rio.to_raster(output_path)
        print(f"Created cumulative raster at {output_path}")

    def perform_cumulative_viz_prominence(self):
        """Function to merge and aggregate visual prominence rasters"""
        output_dir="cumulative_outputs"
        self.check_dir(output_dir)
        input_dir="visual_exposure"
        # Creates a list of all files that contains "rc" string in the input directory
        input_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith("vizexp_rc.tif")]
        output_path = f"{output_dir}/cumulative_visual_exposure.tif"
        # Function sums all the rastes in the input_paths
        self.create_cumulative_rasters(input_paths, output_path)
        return output_path

    def perform_cumulative_meaningful_viz(self):
        """Function to merge and aggregate meaningful visibility rasters"""
        output_dir="cumulative_outputs"
        input_dir="visual_exposure"
        # Creates a list of all files that contains "mv" string in the input directory
        input_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith("mv_rc.tif")]
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_path = f"{output_dir}/cumulative_meaningful_visibility.tif"
        # Function sums all the rasters in the input_paths
        self.create_cumulative_rasters(input_paths, output_path)
        return output_path
    
    def calc_mean_prominence(self, vizprom_fpath, mv_fpath):
        """Function to divide visual prominence and meaningful visibility"""
        output_dir="cumulative_outputs"
        with rio.open(vizprom_fpath) as src1, rio.open(mv_fpath) as src2:
            # Read the raster datasets
            data1 = src1.read(1)
            data2 = src2.read(1)
            # Perform division for all non ZeroDivisionError cases
            data1_copy = data1.astype("float32")
            data2_copy = data2.astype("float32")
            # Where denominator is zero, it fill the output array with zero (0/0 or 5/0 ~ 0)
            result = np.divide(data1_copy, data2_copy, out=np.zeros_like(data1_copy), where=data2_copy!=0)
            # Write the result to the output raster
            profile = src1.profile
            profile.update(dtype="float32", count=1)
            output_fpath = f"{output_dir}/cumulative_mean_prominence.tif"
            # Export the raster to the output directory
            with rio.open(output_fpath, "w", **profile) as dst:
                dst.write(result, 1)
        return output_fpath

    def perform_mean_prominence(self):
        """Function to perform mean visual prominence at each pixel location"""
        vizprom_fpath = self.perform_cumulative_viz_prominence()
        mv_fpath = self.perform_cumulative_meaningful_viz()
        fpath = self.calc_mean_prominence(vizprom_fpath, mv_fpath)
        return fpath
    
    def perform_visual_prominence_bind(self, fname="wind_turbine_prominence.shp"):
        """Function to bind visual prominence value to turbine location in geodataframe"""
        output_dir="cumulative_outputs"
        gdf = self.read_windturbine_file()
        prominence_raster = rio.open(self.perform_mean_prominence())
        prominence_data = prominence_raster.read(1)
        col_name = "mean_prominence"
        gdf[col_name] = None
        # Loop extracts the index position of each wind turbine location
        if gdf.crs.to_epsg() == prominence_raster.crs.to_epsg():
            for index, row in gdf.iterrows():
                lon, lat = row['geometry'].x, row['geometry'].y
                row, col = prominence_raster.index(lon,lat)
                mean_prominence = prominence_data[row,col]
                # Mean prominence value at the index is added to the mean prominence column
                gdf.at[index, col_name] = mean_prominence
        prominence_raster = None
        gdf[col_name] = gdf[col_name].astype(float)
        fpath = os.path.join(output_dir, fname)
        gdf.to_file(fpath)
        print(f"Created a shapefile with mean_prominence column and output saved to {fpath}")
        return gdf

    def visualize_mean_prominence(self, county_state_title):
        """To visualize wind turbine mean prominence"""
        # Read tuple with geodataframe, column name
        gdf = self.perform_visual_prominence_bind()
        col_name = "mean_prominence"
        # Marker sizes
        min_marker_size = 5
        max_marker_size = 25
        # Normalize the values in 'mean_prominence' to be within the specified range
        normalized_sizes = (gdf[col_name] - gdf[col_name].min()) / (
            gdf[col_name].max() - gdf[col_name].min())
        gdf["normalized_sizes"] = min_marker_size + (max_marker_size - min_marker_size) * normalized_sizes
        # Custom diviging color scheme from Color Brewer
        color_scheme = ["#1a9641", "#a6d96a","#ffffbf", "#fdae61", "#d7191c"]
        cmap = ListedColormap(color_scheme)
        # Plot the GeoDataFrame with proportional symbols
        ax = gdf.plot(column=col_name, markersize=gdf["normalized_sizes"]*20,
                      legend=True, cmap=cmap, scheme="NaturalBreaks", k=5,
                      edgecolor="#252525", aspect="equal", linewidth=0.7)
        # Set the number of legend classes
        if ax.get_legend():
            ax.get_legend().set_title("Mean Prominence")
            ax.get_legend().set_bbox_to_anchor((1, 1))
        # Set axes labels
        plt.title(f"Mean Visual Prominence of Wind Turbines in {county_state_title}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig("mean_turbine_prominence.png", dpi=300, bbox_inches="tight")
        plt.show()

    def perform_visual_impact(self):
        """Function to calculate visual impact through adjusted mean visual prominence"""
        print("Input one option from the following:\n1 - square root(n)\n2 - cube root(n)\n3 - log(n+1)\nWhere n is number of turbines")
        val = int(input("Please enter a valid value:"))
        gdf = self.read_windturbine_file()
        turbine_count = len(gdf)
        if val==1:
            adjustment = (pow(turbine_count, 1/2), "square root(n)")
        if val==2:
            adjustment = (pow(turbine_count, 1/3), "cube root(n)")
        if val==3:
            adjustment = (math.log(turbine_count+1), "log(n+1)")
        with rio.open("cumulative_outputs/cumulative_mean_prominence.tif") as src:
            data = src.read(1)
            data = data.astype("float32")
            result = data * adjustment[0]
            # Write the result to the output raster
            profile = src.profile
            profile.update(dtype="float32", count=1)
            # Export the raster to the output directory
            with rio.open("cumulative_outputs/adjusted_mean_prominence.tif", "w", **profile) as dst:
                dst.write(result, 1)
        with rio.open("cumulative_outputs/adjusted_mean_prominence.tif") as dst:
            raster_data = dst.read(1)
            # Define the number of classes
            num_classes = 5
            # Define the class boundaries
            class_boundaries = np.linspace(raster_data.min(), raster_data.max(), num_classes + 1)
            class_map = np.digitize(raster_data, class_boundaries)
            color_scheme = ListedColormap(['#2b83ba', '#abdda4','#ffffbf','#fdae61', '#d7191c'])
            cmap = plt.cm.get_cmap(color_scheme, num_classes)
            plt.imshow(class_map, cmap=cmap)
            plt.title(f"Visual Impact Map using {adjustment[1]} method with {num_classes} classes")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.colorbar()
            plt.savefig("visual_impact_map.png", dpi=300, bbox_inches="tight")
            plt.show()
            print("Saved visual impact map to the current directory")

    def visualize_dem(self, cmap="gist_earth", title="Digital Elevation Model"):
        """Function to visualize DEM"""
        try:
            dst = rio.open(self.read_dem())
            fig, ax = plt.subplots(figsize=(10,5))
            image_hidden = ax.imshow(dst.read()[0], cmap=cmap)
            fig.colorbar(image_hidden, ax=ax, cmap=cmap)
            show(dst,ax=ax, cmap=cmap)
            ax.set_title(f'{title} (EPSG:{dst.crs.to_epsg()})')
            plt.xlabel(f'Longitude ({dst.crs.units_factor[0]})')
            plt.ylabel(f'Latitude ({dst.crs.units_factor[0]})')
            plt.show()
        finally:
            if "dst" in locals() and dst is not None:
                dst.close()

    # Creates a plot for viewshed with legend
    def visualize_viewshed_windturbine(self, viewshed_fpath, turbine_index, cmap="gist_yarg", title="Viewshed for Wind Turbine"):
        """Function to create visualization of viewshed with legend"""
        try:
            gdf = self.read_windturbine_file()
            if not isinstance(turbine_index, int):
                raise TypeError("Turbine index has to be an integer")
            if turbine_index not in range(1, len(gdf)+1):
                raise ValueError(f"Turbine index has to be in range (1,{len(gdf)})")
            viewshed = rio.open(viewshed_fpath)
            fig, ax = plt.subplots(figsize=(10,5))
            image_hidden = ax.imshow(viewshed.read()[0], cmap=cmap)
            fig.colorbar(image_hidden, ax=ax, cmap=cmap)
            i = int(turbine_index)
            gdf.iloc[i-1:i].plot(ax=ax, marker="1", color="#31a354", markersize=1000)
            show(viewshed, ax=ax, cmap=cmap)
            ax.set_title(f"{title}  (EPSG:{viewshed.crs.to_epsg()})")
            plt.xlabel(f'Longitude ({viewshed.crs.units_factor[0]})')
            plt.ylabel(f'Latitude ({viewshed.crs.units_factor[0]})')
            plt.show()
            viewshed = None
        except TypeError as e:
            print(f"{str(e)}")
        except ValueError as e:
            print(f"{str(e)}")
        except IOError as e:
            print(f"File does not exist in path. {str(e)}")
        finally:
            if "viewshed" in locals() and viewshed is not None:
                viewshed.close()

    def explore_turbine_viewshed(self):
        """Function to interactively visualize concerned viewshed and wind turbine location"""
        gdf = self.read_windturbine_file()
        print("Please choose one viewshed to visualize\n1 - Blade end viewshed\n2 - Turbine hub viewshed\n3 - Rotor sweep viewshed,\n4 - merged viewshed")
        val = int(input("Input an option from above:"))
        turbine_index = int(input(f"Input a turbine index between (1,{len(gdf)}):"))
        if val == 1:
            viewshed_fpath = f"viewsheds_blade_end/viewshed_{turbine_index}_blade_rc.tif"
        if val == 2:
            viewshed_fpath = f"viewsheds_hub/viewshed_{turbine_index}_hub_rc.tif"
        if val == 3:
            viewshed_fpath = f"viewsheds_rotor_sweep/viewshed_{turbine_index}_sweep_rc.tif"
        if val == 4:
            viewshed_fpath = f"viewsheds_merged/merged_{turbine_index}_viewshed.tif"
        self.visualize_viewshed_windturbine(viewshed_fpath, turbine_index)

    def visualize_wind_turbines(self, title="Wind Turbine locations"):
        """Function to visualize wind turbine from geodataframe"""
        gdf = self.read_windturbine_file()
        gdf.plot(marker="1", color="#31a354")
        plt.title(title)
        plt.xlabel("Longitude (meter)")
        plt.ylabel("Latitude (meter)")
        plt.show()

    def run_via_pipeline(self, county_state_title):
        """Function to run complete VIA GIS pipeline"""
        try:
            print("VIA GIS Pipeline is getting started....")
            file_size = os.path.getsize(self.dem_fpath)
            fs = round(file_size*0.000001, 2)
            print(f"Raster file size in MegaBytes (MB) is around {fs}")
            print("Approximate compute time to run pipeline: 1-2 hrs (might vary)")
            print("........................................................")
            if fs>=1000:
                raise MemoryError("Please limit the file size to less than 1 GigaByte (GB)")
            self.create_relative_turbine_viewsheds()
            print("........................................................")
            self.reclass_relative_turbine_viewsheds()
            print("........................................................")
            self.perform_viewsheds_merge()
            self.create_turbine_multiringbuffer_raster()
            print("........................................................")
            self.perform_viz_prominence()
            self.reclass_viz_prominence_rasters()
            self.reclass_meaningful_visibility_rasters()
            print("........................................................")
            self.visualize_mean_prominence(county_state_title)
        except OSError as e:
            print(f"{str(e)}")





