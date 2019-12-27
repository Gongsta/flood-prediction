#Bangladesh has the worst floods in the world, I really want to help this region out
import xarray as xr
from functions.floodmodel_utils import get_basin_mask
from dask.distributed import Client, LocalCluster
cluster = LocalCluster()
client = Client(cluster)
print(client.scheduler_info(['services']))

glofasLoaded = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')
era5Loaded = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/Bangladesh/*.nc', combine='by_coords')

glofas = glofasLoaded.copy()
era5 = era5Loaded.copy()

glofas = glofas.rename({'lat': 'latitude'})
glofas = glofas.rename({'lon': 'longitude'})

bangladesh_basin_mask = get_basin_mask(glofas['dis24'].isel(time=0), 'Syr Darya')
glofas = glofas.where(bangladesh_basin_mask, drop=True)

era5 = era5.interp(longitude=glofas.longitude, latitude=glofas.latitude).where(bangladesh_basin_mask, drop=True)



era5 = era5.mean(['latitude', 'longitude'])
glofas = glofas.mean(['latitude', 'longitude'])

from functions.floodmodel_utils import shift_and_aggregate

era5['lsp-4-11'] = shift_and_aggregate(era5['lsp'], shift=4, aggregate=8)
era5['lsp-12-25'] = shift_and_aggregate(era5['lsp'], shift=12, aggregate=14)
era5['lsp-26-55'] = shift_and_aggregate(era5['lsp'], shift=26, aggregate=30)
era5['lsp-56-180'] = shift_and_aggregate(era5['lsp'], shift=56, aggregate=125)



from functions.floodmodel_utils import reshape_scalar_predictand
X = era5
y_orig = glofas['dis24']
y = y_orig.copy()
y = y.diff('time', 1)

Xda, yda = reshape_scalar_predictand(X, y)