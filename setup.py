
"""Install viawind"""

import subprocess as sp
from setuptools import setup

def get_gdal_version():
    """Return system GDAL version"""
    process = sp.Popen(
        ["gdal-config", "--version"],
        stdout=sp.PIPE,
        stderr=sp.PIPE
    )

    sto, ste = process.communicate()
    if ste:
        raise OSError("GDAL is causing problems again. Make sure you can run "
                      "'gdal-config --version' successfully in your terminal")
    version = sto.decode().replace("\n", "")
    return version

def get_requirements():
    """Get requirements and update gdal version number"""
    with open("requirements.txt", encoding="utf-8") as file:
        reqs = file.readlines()
        gdal_version = get_gdal_version()
        gdal_line = [req for req in reqs if req.startswith("pygdal")][0]
        gdal_line = gdal_line[:-1]
        reqs = [req for req in reqs if req.startswith("pygdal")]
        gdal_line = f"{gdal_line}=={gdal_version}.*\n"
        reqs.append(gdal_line)
    return reqs

setup(
    name="viawind",
    version="0.0.1",
    packages=["viawind"],
    description=("Functions to help calculate Visual Impact Assessment (VIA) of onshore wind turbine projects"),
    author="Siddharth Ramavajjala",
    author_email="siddharthrcs@gmail.com",
    install_requires=get_requirements()
    )
