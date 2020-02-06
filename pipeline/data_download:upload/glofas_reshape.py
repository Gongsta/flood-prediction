import sys
sys.path.append("../")

import xarray as xr
#Creating a Dask local cluster for parallel computing (making the computations later on much faster)
from dask.distributed import Client, LocalCluster

import xarray as xr
glofas_loaded = xr.open_mfdataset("/mnt/bucket/stuarts_files/glofas/*/*.nc", combine="by_coords")

glofas = glofas_loaded

mean_discharge = glofas.mean(dim='time')['dis24']

#Mask to replace normal (i.e. not flooded events with NaN)
mask = glofas['dis24'] > (mean_discharge * 1.5)
#Mask to replace values with 0 by NaN
mask2 = glofas['dis24'] != 0

ds2 = glofas['dis24'].where(mask).where(mask2)

for i in range(1999, 2020, 1):
    period=dict(time=slice(str(i)))
    print(period)
    print('hello')
    ds2.loc[period].to_netcdf(('/mnt/bucket/stuarts_files/thresholds_glofas/ver4_reshaped_glofas_'+str(i)+'.nc'))
