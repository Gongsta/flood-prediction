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

ds = CDS_Dataset(dataset_name='reanalysis-era5-land-monthly-means',
                 save_to_folder='/Users/stevengong/Desktop/flood-prediction'  # path to where datasets shall be stored
                )

# define areas of interest (a list of degrees latitude/longitude values for the northern, western, southern and eastern bounds of the area.)


# define time frame
year_start = 2012
year_end = 2012
month_start = 1
month_end = 11

# define requested variables
request = dict(product_type='monthly_averaged_reanalysis',
               format='netcdf',
                #[N,W,S,E]

               #[54, 9, 48, 17]
               area=[52, 15, 51, 16],
               variable=['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
            '2m_temperature', 'evaporation_from_bare_soil', 'evaporation_from_open_water_surfaces_excluding_oceans',
            'evaporation_from_the_top_of_canopy', 'evaporation_from_vegetation_transpiration', 'evapotranspiration',
            'forecast_albedo', 'lake_bottom_temperature', 'lake_ice_depth',
            'lake_ice_temperature', 'lake_mix_layer_depth', 'lake_mix_layer_temperature',
            'lake_shape_factor', 'lake_total_layer_temperature', 'leaf_area_index_high_vegetation',
            'leaf_area_index_low_vegetation', 'potential_evaporation', 'runoff',
            'skin_reservoir_content', 'skin_temperature', 'snow_albedo',
            'snow_cover', 'snow_density', 'snow_depth',
            'snow_depth_water_equivalent', 'snow_evaporation', 'snowfall',
            'snowmelt', 'soil_temperature_level_1', 'soil_temperature_level_2',
            'soil_temperature_level_3', 'soil_temperature_level_4', 'sub_surface_runoff',
            'surface_latent_heat_flux', 'surface_net_solar_radiation', 'surface_net_thermal_radiation',
            'surface_pressure', 'surface_runoff', 'surface_sensible_heat_flux',
            'surface_solar_radiation_downwards', 'surface_thermal_radiation_downwards', 'temperature_of_snow_layer',
            'total_precipitation', 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
            'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',])

#Sending the request


ds.get(years = [str(y) for y in range(year_start, year_end+1)],
       months = [str(a).zfill(2) for a in range(month_start, month_end+1)],
       request = request,
       N_parallel_requests = 12)

