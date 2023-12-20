# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 23:03:55 2023

@author: Sidrcs
"""
import os, shutil, time
import multiprocessing as mp 
import geopandas as gpd


def check_dir(output_dir):
    try:
        # Removes a non-empty folder
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    except OSError as e:
        print(f"{str(e)}")
        
def create_relative_viewshed(gdf, input_dem, output_dir, height, height_name):
    """Function to create relative viewsheds based on wind turbine height"""
    # Create the output directory if it doesn't exist
    check_dir(output_dir)
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

if __name__ == "__main__":
    os.chdir("F:\demo_folder")
    print(f"Changed directory to {os.getcwd()}")
    gdf = gpd.read_file("datasets/tx_aoi.shp")
    gdf = gdf.to_crs(3857)
    input_dem = "datasets/dem_deflate.tif"
    output_dir = "mp_viewsheds_blade_end"
    height = gdf["t_ttlh"][0]
    height_name = "blade"
    
    start = time.time()
    p1 = mp.Process(target=create_relative_viewshed, args=(gdf, input_dem, output_dir, height, height_name))
    p1.start() 
    p1.join() 
    end = time.time()
    print(f"Process is completed in {(end-start)/60} minutes")
    
    output_dir = "test_viewsheds_blade_end"
    start = time.time()
    create_relative_viewshed(gdf, input_dem, output_dir, height, height_name)
    end = time.time()
    print(f"Process is completed in {(end-start)/60} minutes")
    