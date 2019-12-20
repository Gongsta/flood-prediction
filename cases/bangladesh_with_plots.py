#Bangladesh has the worst floods in the world, I really want to help this region out
import xarray as xr
from functions.utils_floodmodel import get_mask_of_basin

era5Loaded = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/Bangladesh/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_*_*.nc', combine='by_coords')
glofasLoaded = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')

era5 = era5Loaded
glofas = glofasLoaded

glofas = glofas.rename({'lon':'longitude'})
glofas = glofas.rename({'lat': 'latitude'})

bangladesh_catchment = get_mask_of_basin(glofas['dis24'].isel(time=0), 'Dhaka')
glofas = glofas.where(bangladesh_catchment, drop=True)