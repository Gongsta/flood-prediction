"""Contains utilities related to the flood model (data processing,saving prediction, generating prediction, etc.) """

import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas
from rasterio import features
from affine import Affine
from functions.utils import calc_area
np.seterr(divide='ignore', invalid='ignore')


def get_river_mask(dis):
    """Returns a DataArray where all values that are of a discharge of 10 or under is false (hence all areas that are a river returns true).
    In other words, a mask of the river in the area.

        Parameters:
        -----------
        dis : xr.DataArray
            A GloFAS DataArray with a discharge dimension contains the 'latitude' and 'longitude' coordinates, 'time' coordinates are optional. I would suggest
            to remove the 'time' coordinates when plotting the river by taking the mean time.

        Returns
        -------
        newDis : xr.DataArray
            a new DataArray where all points that have a discharge of 10 or under is returned as false.

        See Also
        --------
        get_basin_mask : Function which returns the mask of a basin in the xr.DataArray form.

        Examples
        --------
        # Obtaining the Elbe river area
        >> glofas = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')
        >> elbe_river_mask = get_mask_river(glofas['dis24'].mean('time'))
        #Note that if glofas is downloaded as containing all the global coordinates, the function will return all rivers in the world
        >> glofas = glofas.where(elbe_river_mask, drop=True)

        """
    newDis = dis > 10

    return (newDis)


def get_basin_mask(da, basin_name):

    """Returns a mask where all points outside the selected basin are False.

    Parameters:
    -----------
    da : xr.DataArray
        contains the coordinates
    kw_basins : str
        Name of the basin in the basins dataset shapefile

    Returns
    -------
    da : xr.DataArray
        the transformed dataArray with all points outside of the basin identified as False.

    Examples
    --------
    # Obtaining the mask of the Elbe basin
    >> glofas = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')
    >> elbe_basin_mask = get_basin_mask(glofas['dis24'].isel(time=0), 'Elbe')
    >> elbe_basin_mask

    <xarray.DataArray 'basins' (latitude: 1500, longitude: 3600)>
array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]])
Coordinates:
  * longitude  (longitude) float64 -179.9 -179.8 -179.8 ... 179.8 179.9 180.0
  * latitude   (latitude) float64 89.95 89.85 89.75 ... -59.75 -59.85 -59.95
    time       datetime64[ns] 1999-01-01


    #Applying the mask of the basin to the GloFAS dataset by dropping all datasets outside of the basin
    >> glofas = glofas.where(elbe_basin_mask, drop=True)

    #Applying the mask of the basin to the Era5 dataset by dropping all datasets outside of the basin
    >> era5 = era5.interp(latitude=glofas.latitude, longitude=glofas.longitude).where(elbe_basin_mask, drop=True)



    """
    def transform_from_latlon(lat, lon):
        """
        Performing affine transformation, for more information look here: https://pypi.org/project/affine/
        :param lat: xr.DataArray of latitude points
        :param lon: xr.DataArray of longitude points
        :return:
        """
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        trans = Affine.translation(lon[0], lat[0])
        scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
        return trans * scale

    def rasterize(shapes, coords, fill=np.nan, **kwargs):
        """Rasterize a list of (geometry, fill_value) tuples onto the given
        xray coordinates. This only works for 1d latitude and longitude
        arrays.
        """
        transform = transform_from_latlon(coords['latitude'], coords['longitude'])
        out_shape = (len(coords['latitude']), len(coords['longitude']))
        #Performing rasterization using the Rasterio library
        raster = features.rasterize(shapes, out_shape=out_shape,
                                    fill=fill, transform=transform,
                                    dtype=float, **kwargs)
        return xr.DataArray(raster, coords=coords, dims=('latitude', 'longitude'))


    shp2 = "./basins/major_basins/Major_Basins_of_the_World.shp"
    basins = geopandas.read_file(shp2)
    single_basin = basins.query("NAME == '"+ basin_name +"'").reset_index(drop=True)
    shapes = [(shape, n) for n, shape in enumerate(single_basin.geometry)]

    da['basins'] = rasterize(shapes, da.coords)
    da = da.basins == 0
    return da.drop('basins')  # the basins coordinate is not used anymore from here on



def shift_and_aggregate(da, shift, aggregate):
    """

    Parameters
    ----------
    da : xr.DataArray with the feature to shift and aggregate
    shift : int
    aggregate : int

    Returns
    -------
    shifted_and_aggregated : xr.DataArray
        the xr.DataArray with the new shifted and aggregated variable


    Examples
    --------
    # Creating a new predictor from [lsp(t-11), ...., lsp(t-4)]
    >> era5['lsp-4-11'] = shift_and_aggregate(era5['lsp'], shift=4, aggregate=8)


    """
    timeShifted = da.shift(time=shift)
    #Taking the average of the aggregated values
    shifted_and_aggregated = timeShifted.rolling(time=aggregate).sum()/aggregate
    return shifted_and_aggregated



def reshape_scalar_predictand(X_dis, y):
    """Reshape, merge predictor/predictand in time, drop nans.

    Parameters
    ----------
    X_dis : xr.Dataset
        variables: time shifted predictors
        coords: time, latitude, longitude
    y : xr.DataArray
        coords: time


    Returns
    -------
    Xda : xr.DataArray
    yda : xr.DataArray


    Examples
    --------
    #Reshaping X and y
    >> Xda, yda = reshape_scalar_predictand(X, y)


    """
    if isinstance(X_dis, xr.Dataset):
        X_dis = X_dis.to_array(dim='var_dimension')

    # stack -> seen as one dimension for the model
    stack_dims = [a for a in X_dis.dims if a != 'time']  # all except time
    X_dis = X_dis.stack(features=stack_dims)
    Xar = X_dis.dropna('features', how='all')  # drop features that only contain NaN

    if isinstance(y, xr.Dataset):
        if len(y.data_vars) > 1:
            warnings.warn('Supplied `y` with more than one variable.'
                          'Which is the predictand? Supply only one!')
        for v in y:
            y = y[v]  # use the first
            break

    yar = y
    if len(yar.dims) > 1:
        raise NotImplementedError('y.dims: '+str(yar.dims) +
                                  ' Supply only one predictand dimension, e.g. `time`!')

    # to be sure that these dims are not in the output
    for coord in ['latitude', 'longitude']:
        if coord in yar.coords:
            yar = yar.drop(coord)

    # merge times
    yar.coords['features'] = 'predictand'
    Xy = xr.concat([Xar, yar], dim='features')  # maybe merge instead concat?
    Xyt = Xy.dropna('time', how='any')  # drop rows with nan values

    Xda = Xyt[:, :-1]  # last column is predictand
    yda = Xyt[:, -1].drop('features')  # features was only needed in merge
    return Xda, yda





def reshape_multiday_predictand(X_dis, y):
    """Reshape, merge predictor/predictand in time, drop nans.

    Parameters
    ----------
    X_dis : xr.Dataset
        variables: time shifted predictors (name irrelevant)
        coords: time, latitude, longitude
    y : xr.DataArray (multiple variables, multiple timesteps)
        coords: time, forecast_day
    """
    if isinstance(X_dis, xr.Dataset):
        X_dis = X_dis.to_array(dim='var_dimension')

    # stack -> seen as one dimension for the model
    stack_dims = [a for a in X_dis.dims if a != 'time']  # all except time
    X_dis = X_dis.stack(features=stack_dims)
    Xar = X_dis.dropna('features', how='all')  # drop features that only contain NaN

    if not isinstance(y, xr.DataArray):
        raise TypeError('Supply `y` as xr.DataArray.'
                        'with coords (time, forecast_day)!')

    # to be sure that these dims are not in the output
    for coord in ['latitude', 'longitude']:
        if coord in y.coords:
            y = y.drop(coord)

    out_dim = len(y.forecast_day)
    y = y.rename(dict(forecast_day='features'))  # rename temporarily
    Xy = xr.concat([Xar, y], dim='features')  # maybe merge instead concat?
    Xyt = Xy.dropna('time', how='any')  # drop rows with nan values

    Xda = Xyt[:, :-out_dim]  # last column is predictand
    yda = Xyt[:, -out_dim:]  # features was only needed in merge
    yda = yda.rename(dict(features='forecast_day'))  # change renaming back to original
    return Xda, yda



def add_time(vector, time, name=None):
    """Converts input vector to xarray.DataArray with the corresponding input time coordinate.

    Parameters
    ----------
    vector : numpy.array
    time   : xr.DataArray
    name   : str
    """
    return xr.DataArray(vector, dims=('time'), coords={'time': time}, name=name)


def generate_prediction_array(y_pred, y_reana, forecast_range=14):
    """Convenience function to generate a [number of forecasts, forecast range] shaped xr.DataArray from the one
    dimensional xr.DataArray input prediction and converts the predicted discharge change into absolute values,
    starting from t=t0 with the reanalysis value for each forecast.

    Parameters
    ----------
    y_pred          : xr.DataArray
    y_reana         : xr.DataArray
    forecast_range  : int
    """
    # reorganize data into the shape [forecast_range, number_of_forecasts]
    # add +1 to forecast range to include the init state in the length
    num_forecasts = int(np.floor(y_pred.shape[0]/(forecast_range+1)))
    full_forecast_len = num_forecasts*(forecast_range+1)
    new_pred = y_pred[:full_forecast_len].copy()
    time_new = y_pred.time[:full_forecast_len].copy()
    time_new_data = time_new.values.reshape([num_forecasts, (forecast_range+1)])
    pred_multif_data = new_pred.values.reshape([num_forecasts, (forecast_range+1)])
    # set init to reanalysis value
    pred_multif_data[:, 0] = y_reana.where(new_pred)[0::(forecast_range+1)].values
    # cumulative sum to accumulate the forecasted change
    pred_multif_data_fin = np.cumsum(pred_multif_data, axis=1)

    pred_multif = xr.DataArray(pred_multif_data_fin,
                       coords={'num_of_forecast': range(1, num_forecasts+1),
                               'forecast_day': range(0, forecast_range+1),
                               'time': (('num_of_forecast', 'forecast_day'), time_new_data)},
                       dims=['num_of_forecast', 'forecast_day'],
                       name='prediction')
    return pred_multif




def multi_forecast_case_study(pipe_case, x, y):
    """
    Convenience function for predicting discharge via the pre-trained input pipe.
    Loads glofas forecast_rerun data from a in-function set path, used to evaluate
    the model predictions.
    Outputs are 3 xr.DataArrays: One for the model forecast, one for the forecast reruns,
                                 one for the reanalysis.

    Parameters
    ----------
        pipe_case : trainer ML pipe ready for prediction
        x         : xr.DataArray
        y         : xr.DataArray

    Returns
    -------
    xr.DataArray (3 times)
    """
    y_2013 = y
    X_2013 = x
    
    multif_list = []
    multifrerun_list = []
    for forecast in range(1, 5):
        if forecast == 1:
            date_init = '2013-05-18'
            date_end = '2013-06-17'
            fr_dir = '2013051800'
        elif forecast == 2:
            date_init = '2013-05-22'
            date_end = '2013-06-21'
            fr_dir = '2013052200'
        elif forecast == 3:
            date_init = '2013-05-25'
            date_end = '2013-06-24'
            fr_dir = '2013052500'
        elif forecast == 4:
            date_init = '2013-05-29'
            date_end = '2013-06-28'
            fr_dir = '2013052900'

        X_case = X_2013.sel(time=slice(date_init, date_end)).copy()

        # not needed with the new dataset containing 1981-2016
        # X_case = X_case.drop(dim='features', labels='lsp-56-180')
        # y_case = y_2013.sel(time=slice(date_init, date_end)).copy()

        # prediction start from every nth day
        # if in doubt, leave n = 1 !!!
        n = 1
        X_pred = X_case[::n].copy()
        y_pred = pipe_case.predict(X_pred)
        y_pred = add_time(y_pred, X_pred.time, name='forecast')

        multif_case = generate_prediction_array(y_pred, y_2013, forecast_range=30)
        multif_case.num_of_forecast.values = [forecast]
        multif_list.append(multif_case)

        # add glofas forecast rerun data
        # glofas forecast rerun data
        frerun = xr.open_mfdataset(f'../../data/glofas-freruns/{fr_dir}/glof*', combine='by_coords')
        poi = dict(lat=48.35, lon=13.95)
        fr = frerun['dis'].sel(lon=slice(13.9, 14.), lat=slice(48.4, 48.3)).compute()
        fr = fr.where(~np.isnan(fr), 0).drop(labels=['lat', 'lon']).squeeze()
        multifrerun_list.append(fr)

    # merge forecasts into one big array
    date_init = '2013-05-18'
    date_end = '2013-06-28'
    y_case_fin = y_2013.sel(time=slice(date_init, date_end)).copy()
    X_case_multi_core = X_2013.sel(time=slice(date_init, date_end)
                                   ).isel(features=1).copy().drop('features')*np.nan

    X_list = []
    for fc in multif_list:
        X_iter = X_case_multi_core.copy()
        X_iter.loc[{'time': fc.time.values.ravel()}] = fc.values[0]
        X_list.append(X_iter)
    X_multif_fin = xr.concat(X_list, dim='num_of_forecast')
    X_multif_fin.name = 'prediction'

    X_list = []
    for frr in multifrerun_list:
        X_iter = X_case_multi_core.copy()
        ens_list = []
        for fr_num in frr.ensemble:
            fr_iter = frr.sel(ensemble=fr_num)
            X_ens_iter = X_iter.copy()
            X_ens_iter.loc[{'time': frr.time.values}] = fr_iter.values
            ens_list.append(X_ens_iter)
        ens_da = xr.concat(ens_list, dim='ensemble')
        X_list.append(ens_da)
    X_multifr_fin = xr.concat(X_list, dim='num_of_forecast')
    X_multifr_fin.name = 'forecast rerun'
    return X_multif_fin, X_multifr_fin, y_case_fin


def multi_forecast_case_study_tdnn(pipe_case):

    # THIS CODE NEEDS FIXING

    """
    Convenience function for predicting discharge via the pre-trained input pipe.
    Loads glofas forecast_rerun data from a in-function set path, used to evaluate
    the model predictions.
    Outputs are 3 xr.DataArrays: One for the model forecast, one for the forecast reruns,
                                 one for the truth/reanalysis.

    Parameters
    ----------
    pipe_case : trainer ML pipe ready for prediction

    Returns
    -------
    xr.DataArray (3 times)
    """

    features_2013 = xr.open_dataset('../data/features_xy.nc')
    y_orig = features_2013['dis']
    y = y_orig.copy()
    X = features_2013.drop(['dis', 'dis_diff'])

    #Try removing this code and see if it still works?
    y = y.diff('time', 1)

    Xda, yda = reshape_scalar_predictand(X, y)


    multif_list = []
    multifrerun_list = []
    for forecast in range(1, 5):
        if forecast == 1:
            date_init = '2013-05-18'
            date_end = '2013-06-17'
            fr_dir = '2013051800'
        elif forecast == 2:
            date_init = '2013-05-22'
            date_end = '2013-06-21'
            fr_dir = '2013052200'
        elif forecast == 3:
            date_init = '2013-05-25'
            date_end = '2013-06-24'
            fr_dir = '2013052500'
        elif forecast == 4:
            date_init = '2013-05-29'
            date_end = '2013-06-28'
            fr_dir = '2013052900'

        X_case = Xda.sel(time=slice(date_init, date_end)).copy()

        # prediction start from every nth day
        # if in doubt, leave n = 1 !!!
        n = 1
        X_pred = X_case[::n].copy()
        y_pred = pipe_case.predict(X_pred)

        multif_case = generate_prediction_array(y_pred, y_orig, forecast_range=30)
        multif_case.num_of_forecast.values = [forecast]
        multif_list.append(multif_case)

        # add glofas forecast rerun data
        # glofas forecast rerun data
        # frerun = xr.open_dataset('/Users/stevengong/Desktop/data/glofas_freruns_case_study.nc')
        # fr = frerun['dis'].sel(lon=slice(13.9, 14.), lat=slice(48.4, 48.3))
        # fr = fr.drop(labels=['lat', 'lon']).squeeze()
        # multifrerun_list.append(fr)

    # merge forecasts into one big array
    date_init = '2013-05-18'
    date_end = '2013-06-28'
    y_case_fin = yda.sel(time=slice(date_init, date_end)).copy()
    X_case_multi_core = Xda.sel(time=slice(date_init, date_end)
                              ).isel(features=1).copy().drop('features')*np.nan

    X_list = []
    for fc in multif_list:
        X_iter = X_case_multi_core.copy()
        X_iter.loc[{'time': fc.time.values.ravel()}] = fc.values[0]
        X_list.append(X_iter)
    X_multif_fin = xr.concat(X_list, dim='num_of_forecast')
    X_multif_fin.name = 'prediction'

    X_list = []
    # for frr in multifrerun_list:
    #     X_iter = X_case_multi_core.copy()
    #     ens_list = []
    #     for fr_num in frr.ensemble:
    #         fr_iter = frr.sel(ensemble=fr_num)
    #         X_ens_iter = X_iter.copy()
    #         X_ens_iter.loc[{'time': frr.time.values}] = fr_iter.values
    #         ens_list.append(X_ens_iter)
    #     ens_da = xr.concat(ens_list, dim='ensemble')
    #     X_list.append(ens_da)
    # X_multifr_fin = xr.concat(X_list, dim='num_of_forecast')
    # X_multifr_fin.name = 'forecast rerun'
    return X_multif_fin, y_case_fin
