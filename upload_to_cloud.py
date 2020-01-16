import xarray as xr
#SAVING THE DATASET
sample_dataset = xr.open_dataset('../data/features_xy.nc')


sample_dataset.to_zarr('../data/sample_dataset', consolidated=True)

# #Call this in console
#gcloud auth login
#gsutil -m cp -r ../data/sample_dataset gs://weather-data-copernicus/

