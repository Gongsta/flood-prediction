import os
import warnings
import numpy as np
import datetime as dt
import pandas as pd

import matplotlib.pyplot as plt
from dask import delayed
import xarray as xr


np.seterr(divide='ignore', invalid='ignore')


def reshape_scalar_predictand(X_dis, y):
    """Reshape, merge predictor/predictand in time, drop nans.

    Parameters
    ----------
    X_dis : xr.Dataset
        variables: time shifted predictors (name irrelevant)
        coords: time, latitude, longitude
    y : xr.DataArray
        coords: time
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


def add_time(vector, time, name=None):
    """Converts numpy arrays to xarrays with a time coordinate.

    Parameters
    ----------
    vector : np.array
        1-dimensional array of predictions
    time : xr.DataArray
        the return value of `Xda.time`

    Returns
    -------
    xr.DataArray
    """
    return xr.DataArray(vector, dims=('time'), coords={'time': time}, name=name)


def shift_input(X_train_scaled, y_train_scaled, days_intake_length, forecast_day=0):

    """Shifts the datasets in time in order to be able to be fitted the data into the LSTM. This step is done after feature scaling has been applied

        Parameters
        ----------
        X_train_scaled : np.array
            2-dimensional array of shape (number of timestamps, number of features)
        y_train_scaled : np.array
            1-dimensional array of shape (number of timestamps)

        Returns
        -------
        X_train: np.array
        3-dimensional array of shape (number of timestamps, length of days per timesteps, number of features)

        y_train: np.array
        2-dimensional array of shape (number of timestamps, length of days per timesteps i.e. number of forecasted days)
        """

    X_train = []
    y_train = []

    for i in range(X_train_scaled.shape[1]):
        feature_array = []

        for j in range(days_intake_length, len(X_train_scaled)-forecast_day):
            feature_array.append(X_train_scaled[j - days_intake_length:j, i])

        X_train.append(feature_array)

    y_feature_array = []

    for i in range(days_intake_length, len(y_train_scaled)-forecast_day):
        y_feature_array.append(y_train_scaled[i - days_intake_length:i, 0])

    X_train.append(y_feature_array)

    # Creating the transformed y_train array which is shifted by 60 days
    for i in range(days_intake_length, len(y_train_scaled)-forecast_day):
        y_train.append(y_train_scaled[i:i+forecast_day, 0])

    # Transforming the list into numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping new_X_train to be supported as an input format for the LSTM
    X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[2], X_train.shape[0]))  #(batch_size, time_steps, seq_len)

    return X_train, y_train

def shift_test_inputs(X_train, X_valid, y_train, y_valid, days_intake_length, forecast_day, sc, sc2, regressor):

    """Shifts the datasets in time in order to be able to be fitted the data into the pre-trained LSTM. This step is done after
    the model has been trained

        Parameters
        ----------
        X_train_scaled : np.array
            2-dimensional array of shape (number of timestamps, number of features)
        y_train_scaled : np.array
            1-dimensional array of shape (number of timestamps)

        Returns
        -------
        X_train: np.array
        3-dimensional array of shape (number of timestamps, length of days per timesteps, number of features)

        y_train: np.array
        2-dimensional array of shape (number of timestamps, length of days per timesteps i.e. number of forecasted days)
        """
    # Fitting the test values on the model
    X_total = np.concatenate((X_train, X_valid))
    # To test our model on the test set, we will need to use part of the training set. More specifically, since our model has been trained on the
    # 60 previous days, we will need exactly 60 days out of the training set, in addition to all of the test set.
    X_inputs = X_total[len(X_total) - len(X_valid) - days_intake_length:]

    # Scaling the input data
    X_inputs = sc.transform(X_inputs)

    y_total = np.concatenate((y_train, y_valid))
    y_inputs = y_total[len(y_total) - len(y_valid) - days_intake_length:]

    y_inputs = sc2.transform(y_inputs.reshape(-1, 1))
    # Empty array in which we will append values

    X_valid, y_valid = shift_input(X_inputs, y_inputs, days_intake_length, forecast_day)
    y_pred_valid = regressor.predict(X_valid)
    # y_pred_valid = sc.inverse_transform(y_pred_valid)
    y_pred_valid = sc2.inverse_transform(y_pred_valid)

    return y_pred_valid