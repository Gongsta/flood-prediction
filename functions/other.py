"""Functions that are not being used but may be useful in the future"""
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas
from rasterio import features
from affine import Affine
from functions.utils import calc_area
np.seterr(divide='ignore', invalid='ignore')



"""
import shapefile

sf = shapefile.Reader("../basins/major_basins/Major_Basins_of_the_World.shp")

shapes = sf.shapes()

records = sf.records()

basin = "Elbe"

index = get_basin_index(basin, records)

points = shapes[index].points
bbox = shapes[index].bbox
lat, lon = createPointList(bbox[1], bbox[0], bbox[3], bbox[2], era5.latitude.values, era5.longitude.values)
glofasLat, glofasLon = createPointList(bbox[1], bbox[0], bbox[3], bbox[2], glofas.lat.values,glofas.lon.values)

#era5 = era5.sel(latitude=lat, longitude=lon)
glofas = glofas.sel(lat=glofasLat, lon=glofasLon)

"""


def createPointList(latMin, lonMin, latMax, lonMax, latList, lonList):
    """Deprecated, should not use this
            """
    lat = []
    lon = []

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]


    for i in latList:
        if i <= latMax and i >= latMin:
            lat.append(i)

    for i in lonList:
        if i <= lonMax and i >= lonMin:
            lon.append(i)


    if not lat:
        averageLat = (latMin + latMax)/2
        lat.append(find_nearest(latList, averageLat))


    if not lon:
        averageLon = (lonMin + lonMax)/2
        lon.append(find_nearest(lonList, averageLon))



    return lat, lon



def get_basin_index(basin, records):
    """
    Deprecated, a function that returns the index of the basins. Use rather get_basin_mask

    :param basin: str
    :param records:
    :return:
    """

    for i in range(len(records)):
        if basin == records[i][3]:

            return i


def select_upstream(mask_river_in_catchment, lat, lon, basin='Danube'):
    """Return a mask containing upstream river gridpoints.

    Arguments
    ----------
    mask_river_in_catchment : xr.DataArray
        array that is True only for river gridpoints within a certain catchment
        coords: only latitude and longitute

    lat, lon : float
        latitude and longitude of the considered point

    basin : str
        identifier of the basin in the basins dataset

    Returns
    -------
    xr.DataArray
        0/1 mask array with (latitude, longitude) as coordinates
    """

    # this condition should be replaced with a terrain dependent mask
    # but generally speaking, there will always be some points returned that
    # do not influence the downstream point;
    # the statistical model should ignore those points as learned from the dataset
    da = mask_river_in_catchment.load()
    is_west = (~np.isnan(da.where(da.longitude <= lon))).astype(bool)

    mask_basin = get_basin_mask(da, basin_name=basin)

    nearby_mask = da*0.
    nearby_mask.loc[dict(latitude=slice(lat+1.5, lat-1.5),
                         longitude=slice(lon-1.5, lon+1.5))] = 1.
    nearby_mask = nearby_mask.astype(bool)

    mask = mask_basin & nearby_mask & is_west & mask_river_in_catchment

    if 'basins' in mask.coords:
        mask = mask.drop('basins')
    if 'time' in mask.coords:
        mask = mask.drop('time')  # time and basins dimension make no sense here
    return mask


def add_shifted_variables(ds, shifts, variables='all'):
    """Adds additional variables to an array which are shifted in time.

    Parameters
    ----------
    ds : xr.Dataset
    shifts : list(int, )
        e.g. range(1,4); shift=1 means having the value x(t=0) at t=1
    variables : str or list
        e.g. ['lsp', 'cp']

    Returns
    -------
    xr.Dataset
        the input Dataset with the shifted timeseries added as additional variable
    """
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()  # enforce input type

    if variables == 'all':
        variables = ds.data_vars

    for var in variables:
        for i in shifts:
            if i == 0:
                continue  # zero-shift is the original timeseries
            if i > 0:
                sign = '-'
            else:
                sign = '+'
            newvar = var+sign+str(i)
            ds[newvar] = ds[var].shift(time=i)
    return ds



def aggregate_clustersum(ds, cluster, clusterdim):
    """Aggregate a 3-dimensional array over certain points (latitude, longitude).

    Parameters
    ----------
    ds : xr.Dataset
        the array to aggregate (collapse) spatially
    cluster : xr.DataArray
        3-dimensional array (clusterdim, latitude, longitude),
        `clusterdim` contains the True/False mask of points to aggregate over
        e.g. len(clusterdim)=4 means you have 4 clusters
    clusterdim : str
        dimension name to access the different True/False masks

    Returns
    -------
    xr.DataArray
        1-dimensional
    """
    out = xr.Dataset()

    # enforce same coordinates
    interp = True
    if (len(ds.latitude.values) == len(cluster.latitude.values) and
            len(ds.longitude.values) == len(cluster.longitude.values)):
        if (np.allclose(ds.latitude.values, cluster.latitude.values) and
                np.allclose(ds.longitude.values, cluster.longitude.values)):
            interp = False
    if interp:
        ds = ds.interp(latitude=cluster.latitude, longitude=cluster.longitude)
    area_per_gridpoint = calc_area(ds.isel(time=0))

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()

    for var in ds:
        for cl in cluster.coords[clusterdim]:
            newname = var+'_cluster'+str(cl.values)
            this_cluster = cluster.sel({clusterdim: cl})

            da = ds[var].where(this_cluster, 0.)  # no contribution from outside cluster
            out[newname] = xr.dot(da, area_per_gridpoint)
    return out.drop(clusterdim)


def cluster_by_discharge(dis_2d, bin_edges):
    """Custom clustering by discharge.
    """
    cluster = dict()
    for i in range(len(bin_edges)-1):
        cluster[str(i)] = (dis_2d >= bin_edges[i]) & (dis_2d < bin_edges[i+1])
        cluster[str(i)].attrs['units'] = None

    return xr.Dataset(cluster,
                      coords=dict(clusterId=('clusterId', range(len(bin_edges))),
                                  latitude=('latitude', dis_2d.latitude),
                                  longitude=('longitude', dis_2d.longitude)))



def multiday_prediction_to_timeseries(prediction):
    """Convert a 2-dimensional xarray to 1-dimensional with nonunique time-index.

    Parameters
    ----------
    xar : xr.DataArray
        2-dimensional xarray (init_time, forecast_day)

    Returns
    -------
    xr.DataArray
        1-dimensional (time) array with nonunique time index
    """
    forecast_days = len(prediction.forecast_day)
    inits = np.array(prediction.init_time)[:, np.newaxis]

    # repeat the initial time for every forecast day in a column
    times = np.repeat(inits, forecast_days, axis=1)

    # add the forecast day to each column
    for i, day in enumerate(prediction.forecast_day.values):
        times[:, i] += np.timedelta64(day, 'D')

    times = times.ravel()
    data = prediction.values.ravel()
    return pd.Series(data, index=times)



def add_valid_time(pred):
    """Add a another time coordinate giving the valid time of a forecast.

    Parameters
    ----------
    pred : xr.DataArray
        2-dimensional (init_time, forecast_day)

    Returns
    -------
    xr.DataArray
        with an additional 'time' coordinate of forecast validity.
    """
    validtime = np.zeros((len(pred.init_time), len(pred.forecast_day)))
    fcst_days = pred.forecast_day.values

    # iterate over columns and add the respective number of days
    for i, fcst_day in enumerate(fcst_days):
        validtime[:, i] = pred.init_time.values + np.timedelta64(fcst_day, 'D')

    pred.coords['time'] = (('init_time', 'forecast_day'),
                           validtime.astype(np.datetime64))
    return pred


def add_future_precip(X, future_days=13):
    """Add shifted precipitation variables.

    Parameters
    ----------
    X : xr.Dataset
        containing 'lsp' and 'cp' variables
    future_days : int
        create variables that are shifted by 1 up to `future_days`-days

    Returns
    -------
    xr.Dataset
        with additional shifted variables
    """
    for var in ['lsp', 'cp']:
        for i in range(1, future_days+1):
            newvar = var+'+'+str(i)
            X[newvar] = X[var].shift(time=-i)  # future precip as current day variable


def add_future_vars(X, future_days=13):

    """Add shifted variables (from future time points) to the dataset
    for multi-day forecasts.

    Parameters
    ----------
    X : xr.Dataset
        variables: time shifted features
        coords: time
    future_days : int
    """
    if isinstance(X, xr.Dataset):
        for var in X.variables:
            if var not in 'time':
                for i in range(1, future_days+1):
                    newvar = var+'+'+str(i)
                    # future precip as current day variable
                    X[newvar] = X[var].shift(time=-i)
    else:
        raise TypeError('Input type has to be a xr.Dataset!')
    return X

def remove_outlier(x):
    """Removes outliers under, over 1th, 99th percentile of the input pandas series.

    Parameters
    ----------
    x : pd.Series
    """
    x99 = x.quantile(0.99)
    x01 = x.quantile(0.01)
    x = x.where(x > x01).dropna()
    x = x.where(x < x99).dropna()
    return x
