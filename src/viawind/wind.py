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
from matplotlib.colors import ListedColormap

from shapely.geometry import MultiPolygon
from osgeo import gdal

import numpy as np
from numpy import int16

import rioxarray as rxr
import rasterio as rio
from rasterio.plot import show
from rasterio.merge import merge
from rasterio.enums import Resampling, MergeAlg
from rasterio import features


class CalcVisualImpact:
<<<<<<< Updated upstream:src/viawind/wind.py
    """Perform via using wind turbine locations and Digital Elevation Model (DEM)"""
    def __init__(self, windturbine_fpath, dem_fpath):
        # Required datasets to create an instance
        self.windturbine_fpath = windturbine_fpath
        self.dem_fpath = dem_fpath

    def read_windturbine_file(self):
        """Function to read US Wind Turbine dataset"""
=======
    """Class to Perform via using wind turbine locations and Digital Elevation or Surface Model (DEM/DSM)"""
    def __init__(self, windturbine_fpath, dem_fpath, dir_path):
        """Constructs all the necessary attributes for the CalcVisualImpact object

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
        """Function to read and reproject wind turbine dataset

        Returns
        ----------
        Reprojected GeoDataFrame object
        """
>>>>>>> Stashed changes:viawind/wind.py
        try:
            # Reads shapefile and reprojects it to EPSG:3857
            if self.windturbine_fpath.endswith(".shp"):
                gdf = gpd.read_file(f"{self.windturbine_fpath}")
                # Extract column names as list from geodataframe
                col_list = list(gdf.columns.values)
                # Checks if the geodataframe has following columns and raises an attribute error
                if not all(column in col_list for column in ["t_ttlh","t_hh","t_rsa"]):
                    raise AttributeError("t_ttlh: Turbine total height from ground to tip of a blade at its apex in meters (m)\nt_hh: Turbine hub height in meters (m)\nt_rsa: Turbine rotor sweep area in square meters (m2) columns are mandatory for analysis. Refer - https://eerscmap.usgs.gov/uswtdb/api-doc/#keyValue")
                if gdf.crs is None:
                    gdf = gdf.set_crs(4326)
                gdf = gdf.to_crs(3857)
                return gdf
            # Reads CSV and reprojects it to EPSG:3857
            if self.windturbine_fpath.endswith(".csv"):
                df = pd.read_csv(f"{self.windturbine_fpath}")
                # Extract column names as list from dataframe
                col_list = list(gdf.columns.values)
                # Remove pre-existing geometry column
                if "geometry" in col_list:
                    df = df.drop(labels=["geometry"], axis=1)
<<<<<<< Updated upstream:src/viawind/wind.py
                if "xlong" and "ylat" not in col_list:
                    raise ValueError("Latitude, Longitude columns has to be renamed as xlong, ylat")
                if df["xlong"] not in range(-180,181):
=======
                if not all(column in col_list for column in ["xlong", "ylat"]):
                    raise ValueError("Latitude, Longitude columns has to be renamed as xlong, ylat")
                if not all(column in col_list for column in ["t_ttlh","t_hh","t_rsa"]):
                    raise AttributeError("t_ttlh: Turbine total height from ground to tip of a blade at its apex in meters (m)\nt_hh: Turbine hub height in meters (m)\nt_rsa: Turbine rotor sweep area in square meters (m2) columns are mandatory for analysis. Refer - https://eerscmap.usgs.gov/uswtdb/api-doc/#keyValue")
                if not (-180 <= df["xlong"][0] <= 180) or not (-90 <= df["ylat"][0] <= 90):
>>>>>>> Stashed changes:viawind/wind.py
                    raise ValueError("Lon/Lat CRS is not EPSG:4326. Please use EPSG:4326 only")
                # Populate geometry column with lat, lon data from US Wind Turbine database
                gdf = gpd.GeoDataFrame(data=df, geometry=gpd.points_from_xy(df['xlong'], df['ylat']))
                # Set CRS to 3D coordinate systems ESPG:4326
                gdf = gdf.set_crs(4326)
                # Reproject to EPSG:3857
                gdf = gdf.to_crs(3857)
                return gdf
<<<<<<< Updated upstream:src/viawind/wind.py
=======
            return None
        except FileNotFoundError:
            print(f"Wind Turbine file not found: {self.windturbine_fpath}")
>>>>>>> Stashed changes:viawind/wind.py
        except IOError as e:
            print(f"File does not exist in the path and {str(e)}")
        except ValueError as e:
            print(f"{str(e)}")
    
    def read_dem(self):
<<<<<<< Updated upstream:src/viawind/wind.py
        """Function to read, reproject and returns reprojected raster file path"""
=======
        """Function to read, reproject and deflate input DEM/DSM raster

        Returns
        ----------
        Reprojected raster file path"""
>>>>>>> Stashed changes:viawind/wind.py
        # Check if the input raster file exists
        try:
            if not os.path.exists(self.dem_fpath):
                raise FileNotFoundError(f"Raster file not found: {self.dem_fpath}")
            dem = rxr.open_rasterio(f"{self.dem_fpath}")
            # Reprojects raster to EPSG:3857
            dem_reproj = dem.rio.reproject(3857)
            reproj_fpath = self.dem_fpath.replace(".tif", "_reproj.tif")
            # Output raster to reproj_fpath
            dem_reproj.to_raster(reproj_fpath)
            return reproj_fpath
        finally:
            if "dem" in locals() and dem is not None:
                dem = None
            if "dem_reproj" in locals() and dem_reproj is not None:
                dem_reproj = None
<<<<<<< Updated upstream:src/viawind/wind.py
=======

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
>>>>>>> Stashed changes:viawind/wind.py
    
    def create_relative_viewshed(self, output_dir, height, height_name):
        """Function to create relative viewsheds based on wind turbine height

        Parameters
        ----------
        output_dir: str
            Directory file path to store relative viewsheds output
        height: float
            Observer height to construct relative viewshed using gdal
        height_name:str
            Name of turbine height to create a uniform and sequential file names
        """
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

    def create_relative_turbine_viewsheds(self, blade_end="t_ttlh", hub="t_hh", rsa="t_rsa"):
        """Function to compute viewsheds for Turbine Blade End, Hub, Rotor Sweep """
        gdf = self.read_windturbine_file()
<<<<<<< Updated upstream:src/viawind/wind.py
        # Preferred file format from US Wind Turbine Database
        blade_end_height = gdf[blade_end][0]
        hub_height = gdf[hub][0]
        rsa = gdf[rsa][0]
=======
        # Extracts blade end turbine height from GeoDataFrame
        blade_end_height = gdf["t_ttlh"][0]
        # Extracts turbine hub height from GeoDataFrame
        hub_height = gdf["t_hh"][0]
        # Extracts rotor sweep area from GeoDataFrame
        rsa = gdf["t_rsa"][0]
>>>>>>> Stashed changes:viawind/wind.py
        rotor_sweep_height = math.sqrt(float(rsa)/math.pi)
        # Creates relative viewshed rasters
        self.create_relative_viewshed("viewsheds_blade_end", blade_end_height, "blade")
        self.create_relative_viewshed("viewsheds_hub", hub_height, "hub")
        self.create_relative_viewshed("viewsheds_rotor_sweep", rotor_sweep_height, "sweep")
    
    def reclass_relative_viewsheds(self, viewshed_folder_path, new_value, original_value=255):
        """Function to reclassify relative viewshed (Palmer 2022)
        Blade End [255:10], Hub [255:20], Rotor sweep [255:30]
        ...
        Parameters
        ----------
        viewshed_folder_path: str
        new_value: int 
        original_value: int
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
            print(f"Reclassification completed and outputs saved as {viewshed_folder_path}")
        return None
    
    def reclass_relative_turbine_viewsheds(self):
        """Function to reclassify relative viewsheds - Blade End, Turbine, Rotor Sweep (Palmer 2022)"""
        self.reclass_relative_viewsheds("viewsheds_blade_end", new_value=10)
        self.reclass_relative_viewsheds("viewsheds_hub", new_value=20)
        self.reclass_relative_viewsheds("viewsheds_rotor_sweep",new_value=30)
    
    def merge_viewshed_rasters(self, input_file_paths, output_folder_path):
        """Function to merge reclassified viewshed rasters

        Parameters
        ----------
        input_file_paths: str
        output_folder_path: str
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

    def perform_viewsheds_merge(self, output_dir="viewsheds_merged"):
        """Function to merge - blade end, hub, rotor sweep reclassed rasters"""
        # Creates a list of raster file paths from relative viewshed file directories
        viewsheds_blade_end = [file for file in os.listdir('viewsheds_blade_end') if "rc" in file]
        viewsheds_hub = [file for file in os.listdir('viewsheds_hub') if "rc" in file]
        viewsheds_rotor_sweep = [file for file in os.listdir('viewsheds_rotor_sweep') if "rc" in file]
        # Creates a sorted list of aforementioned raster file paths
        sorted_blade_end = sorted(viewsheds_blade_end, key=lambda x: int(x.split('_')[1]))
        sorted_hub = sorted(viewsheds_hub, key=lambda x: int(x.split('_')[1]))
        sorted_rotor_sweep = sorted(viewsheds_rotor_sweep, key=lambda x: int(x.split('_')[1]))
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        # Perform raster merge for each turbine
        for i in range(0,len(self.read_windturbine_file)):
            input_paths = [os.path.join('viewsheds_blade_end',sorted_blade_end[i]), 
                        os.path.join('viewsheds_hub',sorted_hub[i]),
                        os.path.join('viewsheds_rotor_sweep',sorted_rotor_sweep[i])]
            output_path = f"{output_dir}/merged_{i+1}_viewshed.tif"
            self.merge_viewshed_rasters(input_paths, output_path)
        fpath = os.path.join(os.getcwd(), output_dir)
        print(f"Performed raster merge for {len(self.read_windturbine_file)*3} rasters and save to {fpath} ")
        
    def create_multiring_buffer(self, center_point, radii):
        """Function to create non-intersecting concentric buffers as MultiPolygon

        Parameters
        ----------
        """
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

    def create_turbine_multiringbuffer_raster(self, output_dir="raster_buffers", buffer_val_list=None):
        """Function create MultiRingBuffer rasters for wind turbine locations"""
        if not isinstance(buffer_val_list, list):
            raise TypeError("Buffer value list has to be list type like [1,2,3,4]")
        # Create an empty list to store raster file paths
        raster_flist = []
        # Read both shapefile and DEM raster
        gdf = self.read_windturbine_file()
        ref_raster = self.read_dem()
        # Iterate through each row and index position of the geodataframe
        for index, row in enumerate(gdf.itertuples()):
            # Store geometry of each wind turbine location
            point_geom = row.geometry
            # Create a list of radii used to create MultiRingBuffer
            radii = [self.kilo2meter(0.8), self.kilo2meter(3.2), self.kilo2meter(8), self.kilo2meter(16.1)]
            # Function to create a MultiPolygon buffer for the radii
            buffers = self.create_multiring_buffer(point_geom, radii)
            output_raster_path = os.path.join(output_dir, f"rasterized_{index+1}_turbine_buffer.tif")
            # Burn the following values under each pixel of the MultiRing Buffer
            if buffer_val_list is None:
                buffer_val_list = [1,2,3,4]
            # Functiom to rasterize the MultiRing buffer
            self.rasterize_concentric_buffers(ref_raster, buffers, buffer_val_list, output_raster_path)
            raster_flist.append(output_raster_path)
            # Close datasets
            point_geom = None
            buffers = None
        return raster_flist
    
    def create_turbine_viz_prominence(self, input_paths, output_path):
        """Function to merge distance zone and visual exposure"""
        files = [rxr.open_rasterio(path) for path in input_paths]
        raster_sum = files[0] + files[1]
        raster_sum.rio.to_raster(output_path)

    def perform_viz_prominence(self, output_dir="visual_exposure"):
        """Function to create visual prominence rasters for each turbine location"""
        # Creates a list of file paths for merged viewshed and raster buffers directories
        merged_viewsheds = [file for file in os.listdir("viewsheds_merged")]
        distance_zone_rasters = [file for file in os.listdir("raster_buffers")]
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

<<<<<<< Updated upstream:src/viawind/wind.py
    def reclass_viz_prominence_rasters(self, output_dir="visual_exposure"):
        """Function reclassify visual pro rasters as per table 3 (Palmer 2022)"""
=======
    def reclass_viz_prominence_rasters(self):
        """Function reclassify visual prominence rasters as per table 3 (Palmer 2022)"""
        output_dir="visual_exposure"
>>>>>>> Stashed changes:viawind/wind.py
        for file in os.listdir(output_dir):
            input_raster = os.path.join(output_dir, file)
            # Opens each input raster
            if input_raster.endswith('.tif') and "rc" not in input_raster:
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

        print(f"Reclassification complete for visual prominence and output saved to {output_dir}")

    def reclass_meaningful_visibility_rasters(self, output_dir="visual_exposure"):
        """Function to reclass visual exposure rasters to reflect meaningful visibility"""
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
                    output_raster = file.replace('vizexp.tif', 'mv_rc.tif')
                    output_path = os.path.join(output_dir, output_raster)
                    # Write the reclassified data to the output raster
                    with rio.open(output_path, 'w', driver=src.driver, height=src.height, width=src.width,
                                count=src.count, crs=src.crs, transform=src.transform, dtype=data.dtype) as dst:
                        dst.write(rc, 1)
        print(f"Reclassification complete for meaningful visibility and output saved to {output_dir}")

    def create_cumulative_rasters(self, input_paths, output_path):
        """Function to perform raster sum of input rasters

         Parameters
        ----------
        input_paths: str
        output_path: str
        """
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

    def perform_cumulative_viz_prominence(self, input_dir="visual_exposure", output_dir="cumulative_outputs"):
        """Function to merge and aggregate visual prominence rasters"""
        # Creates a list of all files that contains "rc" string in the input directory
        input_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if "rc" in file]
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_path = f"{output_dir}/cumulative_visual_exposure.tif"
        # Function sums all the rastes in the input_paths
        self.create_cumulative_rasters(input_paths, output_path)
        return output_path

    def perform_cumulative_meaningful_viz(self, input_dir="visual_exposure", output_dir="cumulative_outputs"):
        """Function to merge and aggregate meaningful visibility rasters"""
        # Creates a list of all files that contains "mv" string in the input directory
        input_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if "mv" in file]
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_path = f"{output_dir}/cumulative_meaningful_visibility.tif"
        # Function sums all the rasters in the input_paths
        self.create_cumulative_rasters(input_paths, output_path)
        return output_path
    
<<<<<<< Updated upstream:src/viawind/wind.py
    def calc_mean_prominence(self, vizprom_fpath, mv_fpath, output_dir="cumulative_outputs"):
        """Function to divide visual prominence and meaningful visibility"""
=======
    def calc_mean_prominence(self, vizprom_fpath, mv_fpath):
        """Function to divide visual prominence and meaningful visibility

         Parameters
        ----------
        vizprom_fpath:str
        mv_fpath: str
        """
        output_dir="cumulative_outputs"
>>>>>>> Stashed changes:viawind/wind.py
        with rio.open(vizprom_fpath) as src1, rio.open(mv_fpath) as src2:
            # Read the raster datasets
            data1 = src1.read(1)
            data2 = src2.read(1)
            # Avoid division by zero
            data2_nonzero = data2.copy()
            data2_nonzero[data2 == 0] = 1
            # Perform raster division for non-zero case
            result = np.where(data2 == 0, np.nan, data1 / data2_nonzero)
            # Write the result to the output raster
            profile = src1.profile
            profile.update(dtype='float32', count=1)
            output_fpath = f"{output_dir}/cumulative_mean_prominence.tif"
            # Export the raster to the output directory
            with rio.open(output_fpath, 'w', **profile) as dst:
                dst.write(result, 1)
        return output_fpath

    def perform_mean_prominence(self):
        """Function to perform mean visual prominence at each pixel location"""
        vizprom_fpath = self.perform_cumulative_viz_prominence()
        mv_fpath = self.perform_cumulative_meaningful_viz()
        fpath = self.calc_mean_prominence(vizprom_fpath, mv_fpath)
        return fpath
    
<<<<<<< Updated upstream:src/viawind/wind.py
    def bind_mean_prominence_to_turbines(self, col_name="mean_prominence"):
        """Function to bind visual prominence value to turbine location in geodataframe"""
        gdf = self.read_windturbine_file()
        prominence_raster = rio.open(self.perform_mean_prominence())
        prominence_data = prominence_raster.read(1)
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
        return gdf
    
    def perform_visual_prominence_bind(self, output_dir="cumulative_outputs", col_name="mean_prominence", fname="wind_turbine_prominence.shp"):
        """Function to create shapefile with mean visual prominence column"""
        gdf = self.bind_mean_prominence_to_turbines(col_name)
        gdf[col_name] = gdf[col_name].astype(float)
        gdf.to_file(os.path.join(output_dir, fname))
        print(f"Created a visual prominence shapefile at {output_dir}")
        return (gdf,col_name)
=======
    def perform_visual_prominence_bind(self, fname="wind_turbine_prominence.shp"):
        """Function to bind visual prominence value to turbine location in geodataframe

        Parameters
        ----------
        fname: str
            Name of wind turbine shapefile to be saved (along with .shp extension)
        Returns
        ----------
        GeoDataFrame with mean_prominence column
        """
        if fname.endswith(".shp"):
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
        if not fname.endswith(".shp"):
            raise ValueError("Provide file name with .shp extension only")
>>>>>>> Stashed changes:viawind/wind.py

    def visualize_mean_prominence(self, county_state_title):
        """To visualize wind turbine mean prominence

        Parameters
        ----------
        county_state_title: str
            Title for mean visual prominence raster bubble plot (Project name, County, State)
        """
        # Read tuple with geodataframe, column name
        viz_tuple = self.perform_visual_prominence_bind()
        gdf = viz_tuple[0]
        col_name = viz_tuple[1]
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
        plt.savefig("mean_turbine_prominence.png", dpi=300)
        plt.show()

    def visualize_dem(self, cmap="gist_earth", title="Digital Elevation Model"):
        """Function to visualize DEM

        Parameters
        ----------
        cmap: str
        title: str
        """
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
            gdf.iloc[i:i+1].plot(ax=ax, marker="", color="gray")
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

<<<<<<< Updated upstream:src/viawind/wind.py
    def run_via_pipeline(self, county_state_title):
        """Function to run complete VIA GIS pipeline"""
        self.create_relative_turbine_viewsheds()
        self.reclass_relative_turbine_viewsheds()
        self.perform_viewsheds_merge()
        self.create_turbine_multiringbuffer_raster()
        self.perform_viz_prominence()
        self.reclass_viz_prominence_rasters()
        self.reclass_meaningful_visibility_rasters()
        vizprom_fpath = self.perform_cumulative_viz_prominence()
        mv_fpath = self.perform_cumulative_meaningful_viz()
        self.visualize_mean_prominence(county_state_title)
=======
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
        gdf.plot(marker="1", color="#31a354", markersize=500)
        plt.title(title)
        plt.xlabel("Longitude (meter)")
        plt.ylabel("Latitude (meter)")
        plt.savefig("wind_turbines.png", dpi=300, bbox_inches="tight")
        plt.show()

    def run_via_pipeline(self, county_state_title):
        """Function to run mean visual prominence geospatial pipeline pipeline

        Parameters
        ----------
        county_state_title: str
            Title for mean visual prominence raster bubble plot (Project name, County, State)
        """
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
>>>>>>> Stashed changes:viawind/wind.py





