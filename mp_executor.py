# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:49:11 2023

@author: Sidrcs
"""

import os, time
import geopandas as gpd
import rasterio as rio
from concurrent.futures import ThreadPoolExecutor

def check_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def create_viewshed_for_point(args):
    point, height, input_dem, output_filename = args
    x, y = point.xy
    x, y = x[0], y[0]
    command = f"gdal_viewshed -oz {height} -ox {x} -oy {y} {input_dem} {output_filename}"
    os.system(command)

def create_relative_viewshed(args):
    gdf, input_dem, output_dir, height, height_name = args
    check_dir(output_dir)

    with ThreadPoolExecutor() as executor:
        output_filenames = [
            os.path.join(output_dir, f'viewshed_{idx}_{height_name}.tif') for idx in range(1, len(gdf) + 1)
        ]
        executor.map(create_viewshed_for_point, zip(gdf['geometry'], [height]*len(gdf), [input_dem]*len(gdf), output_filenames))

    fpath = os.path.join(os.getcwd(), output_dir)
    print(f'{height_name} viewsheds created for {len(gdf)} points and outputs saved to {fpath}')

if __name__ == "__main__":
    os.chdir(r"F:\demo_folder")
    gdf = gpd.read_file("datasets/tx_aoi.shp")
    gdf = gdf.to_crs(3857)
    input_dem = "datasets/dem_deflate.tif"
    output_dir = "mp_viewsheds_blade_end"
    height = gdf["t_ttlh"][0]
    height_name = "blade"
    
    start = time.time()
    args = (gdf, input_dem, output_dir, height, height_name)
    create_relative_viewshed(args)
    end = time.time()
    print(f"Time elapsed in {end-start} seconds")
    # OUTPUT: Time elapsed in 56.30747747421265 seconds
