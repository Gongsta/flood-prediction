#Saving our API key so that we can use Climate Data Store API to extract data
UID = '26343'
API_key = 'b61b1b28-a04b-4cb3-acb1-b48775702011'

import os
with open(os.path.join(os.path.expanduser('~'), '.cdsapirc2'), 'w') as f:
    f.write('url: https://cds.climate.copernicus.eu/api/v2\n')
    f.write(f'key: {UID}:{API_key}')


#Adding functions to our directory so we can use other functions
import sys
sys.path.append('../')
from functions.data_download import CDS_Dataset

ds = CDS_Dataset(dataset_name='reanalysis-era5-single-levels',
                 save_to_folder='/Volumes/Seagate Backup Plus Drive/data/Elbe'  # path to where datasets shall be stored
                )

# define areas of interest (a list of degrees latitude/longitude values for the northern, western, southern and eastern bounds of the area.)

# define time frame
year_start = 1999
year_end = 2019
month_start = 1
month_end = 12

# define requested variables
request = dict(product_type='reanalysis',
               format='netcdf',
                #[N,W,S,E]
               area=[54, 9, 48, 17],
               variable=['convective_precipitation', 'land_sea_mask', 'large_scale_precipitation',
                         'runoff', 'slope_of_sub_gridscale_orography', 'soil_type',
                         'total_column_water_vapour', 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2'])

#Sending the request
ds.get(years = [str(y) for y in range(year_start, year_end+1)],
       months = [str(a).zfill(2) for a in range(month_start, month_end+1)],
       request = request,
       N_parallel_requests = 12)


