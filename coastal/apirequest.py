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

ds = CDS_Dataset(dataset_name='satellite-sea-level-global',
                 save_to_folder='./data'  # path to where datasets shall be stored
                )

# define areas of interest (a list of degrees latitude/longitude values for the northern, western, southern and eastern bounds of the area.)


# define time frame
year_start = 2005
year_end = 2015
month_start = 1
month_end = 2

# define requested variables
request = dict(format='netcdf')

#Sending the request

#TODO: Check if this works since I'm not sure we can directly download it as a .nc file

ds.get(years = [str(y) for y in range(year_start, year_end+1)],
       months = [str(a).zfill(2) for a in range(month_start, month_end+1)],
       request = request,
       N_parallel_requests = 12)





import cdsapi

c = cdsapi.Client()

c.retrieve(
    'satellite-sea-level-global',
    {
        'year': '1995',
        'month': '01',
        'day': [
            '01', '02', '03',
        ],
        'variable': 'all',
        'format': 'zip',
    },
    'coastsample.zip')