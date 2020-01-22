import xarray as xr

era5_loaded = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/Elbe/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_*_*.nc', combine='by_coords', chunks={'latitude':4, 'longitude':4,'time':4})
glofas_loaded = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')

era5_loaded = era5_loaded.chunk(25)

era5_loaded.to_zarr('/Volumes/Seagate Backup Plus Drive/weatherdata/Elbe', consolidated=True)
glofas_loaded.to_zarr('/Volumes/Seagate Backup Plus Drive/weatherdata/glofas', consolidated=True)


#Call this in console
#gcloud auth login
#gsutil -m cp -r /Volumes/portableHardDisk/weatherdata/Elbe/ gs://weather-data-copernicus/
#gsutil -m cp -r /Volumes/portableHardDisk/weatherdata/glofas/ gs://weather-data-copernicus/

#this is suggested in the pangeo documentation
import xarray as xr
import fsspec
ds = xr.open_zarr(fsspec.get_mapper('gcs://weather-data-copernicus/Elbe'))
#ds = xr.open_zarr('/Volumes/Seagate Backup Plus Drive/weatherdata/glofas')

#This is from the original xarray documentation

import gcsfs
from gcsfs import GCSFileSystem
gcs = GCSFileSystem(project="flood-prediction-263210", token='anon')
gcsmap = gcsfs.mapping.GCSMap('weather-data-copernicus', gcs=gcs, check=True, create=False)
ds_gcs = xr.open_zarr(gcsmap)


#dask_function(..., storage_options={'token': gcs.session.credentials})

file = gcs.open('gcs://weather-data-copernicus/data/Elbe/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_1999_01.nc')
#This allows me to read the file from the cloud
ds = xr.open_mfdataset(file, engine='h5netcdf')