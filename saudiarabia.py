#Data extraction

#Saving our API key so that we can use Climate Data Store API to extract data
UID = '26343'
API_key = 'b61b1b28-a04b-4cb3-acb1-b48775702011'

import os
with open(os.path.join(os.path.expanduser('~'), '.cdsapirc2'), 'w') as f:
    f.write('url: https://cds.climate.copernicus.eu/api/v2\n')
    f.write(f'key: {UID}:{API_key}')


#Adding functions to our directory so we can use other functions
import sys
sys.path.append('./')
from functions.data_download import CDS_Dataset

ds = CDS_Dataset(dataset_name='reanalysis-era5-single-levels',
                 save_to_folder='../data/'  # path to where datasets shall be stored
                )

# define areas of interest (N/W/S/E)
saudi_arabia = [28,36,17,50]

# define time frame
year_start = 2005
year_end = 2010
month_start = 1
month_end = 12

# define requested variables
request = dict(product_type='reanalysis',
               format='netcdf',
               area=saudi_arabia,
               variable=['convective_precipitation','land_sea_mask','large_scale_precipitation',
            'runoff','slope_of_sub_gridscale_orography','soil_type',
            'total_column_water_vapour','volumetric_soil_water_layer_1','volumetric_soil_water_layer_2'],
               )

#Sending the request
ds.get(years = [str(y) for y in range(year_start, year_end+1)],
       months = [str(a).zfill(2) for a in range(month_start, month_end+1)],
       request = request,
       N_parallel_requests = 12)




#Model
era5 = xr.open_dataset('../data/reanalysis-era5-pressure-levels_temperature_2000_01.nc')

glofas = xr.open_dataset('../data/dataset-cems-glofas-historical-fc9c62e9-1df3-4179-84dd-277f77e620fb/CEMS_ECMWF_dis24_20010107_glofas_v2.1.nc')



