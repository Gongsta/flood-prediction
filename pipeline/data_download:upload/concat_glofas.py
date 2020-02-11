import sys
sys.path.append("../")

import xarray as xr
#Creating a Dask local cluster for parallel computing (making the computations later on much faster)
from dask.distributed import Client, LocalCluster

import os

import os
import errno

filename = "/mnt/bucket/stuarts_files/monthly_glofas/"

for i in range(1999, 2020):
    for j in range(1,13):
        filename= ("/mnt/bucket/stuarts_files/monthly_glofas/" + str(i) + "/" + str(i) + str(j) + '.nc')

        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(filename, "w") as f:
            f.write("FOOBAR")import xarray as xr

for i in range(1999, 2020, 1):

    glofas_loaded = xr.open_mfdataset("/mnt/bucket/stuarts_files/glofas//*.nc", combine="by_coords")
