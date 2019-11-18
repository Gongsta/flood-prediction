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


#Downloading ERA5 Dataset

ds = CDS_Dataset(dataset_name='reanalysis-era5-single-levels',
                 save_to_folder='../data/'  # path to where datasets shall be stored
                )

# define areas of interest (N/W/S/E)
saudi_arabia = [28,36,17,50]

# define time frame of the downlaod
year_start = 2005
year_end = 2010
month_start = 1
month_end = 12

# define request variables
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

ds.get(years = ['2008'],
       months = ['5'],
       request = request,
       N_parallel_requests = 12)



#Code for downloading Glofas Data
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'cems-glofas-historical',
    {
        'format':'zip',
        'year':[
            '2005','2006','2007',
            '2008'
        ],
        'variable':'River discharge',
        'month':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12'
        ],
        'day':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12',
            '13','14','15',
            '16','17','18',
            '19','20','21',
            '22','23','24',
            '25','26','27',
            '28','29','30',
            '31'
        ],
        'area': ['28','36','17','50'],
        'dataset':'Consolidated reanalysis',
        'version':'2.1'
    },
    'download.zip')



import xarray as xr

#Data Preprocessing
glofas = xr.open_dataset('../data/dataset-cems-glofas-historical-fc9c62e9-1df3-4179-84dd-277f77e620fb/CEMS_ECMWF_dis24_20010101_glofas_v2.1.nc')

#Putting all of glofas files into a single file
for i in range(2,32):
    if i<10:
        i = '0' + str(i)

    glofas = glofas.merge(xr.open_dataset('../data/dataset-cems-glofas-historical-fc9c62e9-1df3-4179-84dd-277f77e620fb/CEMS_ECMWF_dis24_200101'+ str(i) + '_glofas_v2.1.nc'))

#Saving the 31 glofas to a single .nc file, stored in exterior folder not committed to github (the file is 670mb per month)
glofas.to_netcdf(path="../data/glofas2001.nc")

#Putting all of era5 into a single file
era5 = xr.open_dataset('../data/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_2005_01.nc')

for n in range(5,11):
    if n<10:
        n = '0'+ str(n)

    for i in range(2,13):
        if i<10:
            i = '0' + str(i)

         era5= era5.merge(xr.open_dataset('../data/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_20' + str(n) +'_' + str(i) + '.nc'))


era5.to_netcdf(path="../data/2005-2010-era5.nc")


glofas = xr.open_dataset("../data/glofas2001.nc")



#Creating the model

