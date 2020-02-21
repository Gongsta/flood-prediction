#Saving our API key so that we can use Climate Data Store API to extract data
UID = '26343'
API_key = 'b61b1b28-a04b-4cb3-acb1-b48775702011'

import os
with open(os.path.join(os.path.expanduser('~'), '.cdsapirc2'), 'w') as f:
    f.write('url: https://cds.climate.copernicus.eu/api/v2\n')
    f.write(f'key: {UID}:{API_key}')

# #Code for downloading Glofas Data
# import cdsapi
#
# c = cdsapi.Client()
#
# c.retrieve(
#     'cems-glofas-historical',
#     {
#         'format':'zip',
#         'year':[
#             '2005'
#         ],
#         'variable':'River discharge',
#         'month':[
#             '01','02','03', '04','05','06'
#         ],
#         'day':[
#             '01','02','03',
#             '04','05','06',
#             '07','08','09',
#             '10','11','12',
#             '13','14','15',
#             '16','17','18',
#             '19','20','21',
#             '22','23','24',
#             '25','26','27',
#             '28','29','30',
#             '31'
#         ],
#         'dataset':'Consolidated reanalysis',
#         'version':'2.1'
#     },
#     '2005.zip')

#'01','02','03', '04','05','06'
#'07','08','09','10','11','12'


import cdsapi

c = cdsapi.Client()

c.retrieve(
    'cems-glofas-historical',
    {
        'variable': 'River discharge',
        'dataset': 'Consolidated reanalysis',
        'version': '2.1',
        'year': '2007',
        'month': '01',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'format': 'zip',
    },
    'glofas.zip')