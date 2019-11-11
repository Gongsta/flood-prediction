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

ds = CDS_Dataset(dataset_name='reanalysis-era5-pressure-levels',
                 save_to_folder='./data/'  # path to where datasets shall be stored
                )

# define areas of interest
area_dict = dict(danube=[50, 7, 47, 20],
                 asia=[55, -140, 0, 35],
                 usa=[50, -125, 25, -70])

# define time frame
year_start = 2000
year_end = 2000
month_start = 1
month_end = 12

# define requested variables
request = dict(product_type='reanalysis',
               format='netcdf',
               area=area_dict['usa'],
               variable=['geopotential', 'temperature', 'specific_humidity'],
               pressure_level=['850', '700', '500'])

#Sending the request
ds.get(years = [str(y) for y in range(year_start, year_end+1)],
       months = [str(a).zfill(2) for a in range(month_start, month_end+1)],
       request = request,
       N_parallel_requests = 12)