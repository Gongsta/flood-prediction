#Classify glofas dataset to identify areas at timepoints with and without floods

import sys
sys.path.append("../")

#Creating a Dask local cluster for parallel computing (making the computations later on much faster)
from dask.distributed import Client, LocalCluster

import xarray as xr
glofas_loaded = xr.open_mfdataset("/mnt/bucket/stuarts_files/glofas/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc", combine="by_coords")

glofas = glofas_loaded

mean_discharge = glofas.mean(dim='time')['dis24']

#Mask to replace normal (i.e. not flooded events with NaN)
mask = glofas['dis24'] > (mean_discharge * 2)

mask2 = glofas['dis24'] > 10
#Mask to replace values with 0 by NaN
#mask2 = glofas['dis24'] != 0

ds2 = glofas['dis24'].where(mask).where(mask2)

import shutil

ds2.to_netcdf(('./classified_glofas_1999-2019.nc'))
#
# for i in range(2014, 2020, 1):
#     period=dict(time=slice(str(i)))
#     print(period)
#     ds2.loc[period].to_netcdf(('./reshaped_glofas_' + str(i) + '.nc'))
#     # shutil.move(('./reshaped_glofas_' + str(i) + '.nc'), ('/mnt/bucket/stuarts_files/thresholds_glofas/reshaped_glofas_' + str(i) + '.nc'))
#
#
#
