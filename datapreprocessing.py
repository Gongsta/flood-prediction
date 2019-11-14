import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

import sys
sys.path.append('./')

from functions.utils_floodmodel import get_mask_of_basin, add_shifted_variables, reshape_scalar_predictand

era5 = xr.open_dataset('../data/reanalysis-era5-pressure-levels_temperature_2000_01.nc')


glofas = xr.open_dataset('../data/dataset-cems-glofas-historical-fc9c62e9-1df3-4179-84dd-277f77e620fb/CEMS_ECMWF_dis24_20010101_glofas_v2.1.nc')

for i in range(2,30):
    if i<10:
        i = '0' + str(i)

    glofas = glofas.merge(xr.open_dataset('../data/dataset-cems-glofas-historical-fc9c62e9-1df3-4179-84dd-277f77e620fb/CEMS_ECMWF_dis24_200101'+ str(i) + '_glofas_v2.1.nc'))




glofas3 = glofas.merge(glofas2)

glofasplot = glofas3.isel(lat='', lon=2112)

glofasplot.to_dataframe().plot()


#29.5, 31.3