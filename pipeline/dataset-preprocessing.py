import sys
sys.path.append("../")

from functions.floodmodel_utils import get_basin_mask, get_river_mask, reshape_scalar_predictand, reshape_multiday_predictand
import xarray as xr
#Creating a Dask local cluster for parallel computing (making the computations later on much faster)
from dask.distributed import Client, LocalCluster

#LIBRARY IMPORTS
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client, LocalCluster


#Connecting to a cluster to be able to run the code locally/on the cloud
#cluster = LocalCluster()  # n_workers=10, threads_per_worker=1,
client = Client(processes=False)




period_train = dict(time=slice(None, '2005'))
period_valid = dict(time=slice('2006', '2011'))
period_test = dict(time=slice('2012', '2016'))

#Loading the transformed dataset
#glofas_loaded = xr.open_dataset("/tmp/flood_prediction/pipeline/data_download:upload/reshaped_glofas_2019.nc")
glofas_loaded = xr.open_mfdataset("/mnt/bucket/stuarts_files/glofas/*/*.nc", combine="by_coords")

era5_loaded = xr.open_mfdataset("/mnt/bucket/stuarts_files/Elbe/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_*.nc", combine="by_coords")


era5 = era5_loaded.copy()

glofas = glofas_loaded.rename({'lon' : 'longitude'})
glofas = glofas.rename({'lat': 'latitude'})


#Selecting region of interest by creating a mask
elbe_basin_mask = get_basin_mask(glofas['dis24'].isel(time=0), 'Elbe')

elbe_river_mask = get_river_mask(glofas['dis24'].isel(time=0))


glofas_masked = glofas.loc[period_test].where(elbe_basin_mask, drop=True).where(elbe_river_mask, drop=True)

#lofas_masked.to_netcdf('./glofas_masked_danube_2012-2016.nc')

y_orig = glofas_masked
#Making a copy because y will be transformed to represent the variation of discharge. The model will be predicting the variation of discharge, not the quantity of discharge itself
y = y_orig


#Reshape to align in coordinates
era5_masked = era5.loc[period_test].interp(latitude=glofas_masked.latitude, longitude=glofas_masked.longitude).where(elbe_basin_mask, drop=True)
X = era5_masked


#Downsampling our time from hourly to daily
X = X.resample(time='1D').mean()

# Reshape to align in time
from tqdm import tqdm

for i in tqdm(range(0, len(X.latitude))):

    for j in range(0, len(X.longitude)):

        if i == 0 and j == 0:
            Xda, yda = X.isel(latitude=i, longitude=j).to_array(dim='features').T, y.isel(latitude=i,
                                                                                          longitude=j).to_array(
                dim='features').T

        else:
            XdaNew, ydaNew = X.isel(latitude=i, longitude=j).to_array(dim='features').T, y.isel(latitude=i,
                                                                                                longitude=j).to_array(
                dim='features').T

            Xda = xr.concat([Xda, XdaNew], 'points')
            yda = xr.concat([yda, ydaNew], 'points')




#Xda.to_netcdf('./elbe.nc')

yda.to_netcdf('./elbeglofas2012-2016.nc')