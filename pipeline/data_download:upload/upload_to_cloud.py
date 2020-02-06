import xarray as xr
#SAVING THE DATASET
sample_dataset = xr.open_dataset('../../features_xy.nc')

sample_dataset['dis'].to_netcdf('/mnt/bucket/stuarts_files/sample_dataset.nc')

# #Call this in console
#gcloud auth login
#gsutil -m cp -r ../data/sample_dataset gs://weather-data-copernicus/

