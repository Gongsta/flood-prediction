# Creating the model
import numpy as np
import matplotlib.pyplot as plt

from dask.distributed import Client, LocalCluster

cluster = LocalCluster(processes=True)  # n_workers=10, threads_per_worker=1,
client = Client(cluster)  # memory_limit='16GB',

import xarray as xr

y = glofas['dis24']
X = era5

from functions.utils_floodmodel import reshape_scalar_predictand

X, y = reshape_scalar_predictand(X, y)

"""
def feature_preproc(era5, glofas, timeinit, timeend):

    era5_features = era5

    # interpolate to glofas grid
    era5_features = era5_features.interp(latitude=glofas.latitude,
                                         longitude=glofas.longitude)
    # time subset
    era5_features = era5_features.sel(time=slice(timeinit, timeend))
    glofas = glofas.sel(time=slice(timeinit, timeend))

    # select the point of interest
    # poi = dict(latitude=48.403, longitude=15.615)  # krems (lower austria), outside the test dataset
    poi = dict(latitude=48.35, longitude=13.95)  # point in upper austria

    dummy = glofas['dis'].isel(time=0)
    danube_catchment = get_mask_of_basin(dummy, kw_basins='Danube')
    X = era5_features.where(danube_catchment).mean(['latitude', 'longitude'])

    # select area of interest and average over space for all features
    dis = glofas.interp(poi)
    y = dis.diff('time', 1)  # compare predictors to change in discharge

    shifts = range(1, 3)
    notshift_vars = ['swvl1', 'tcwv', 'rtp_500-850']
    shift_vars = [v for v in X.data_vars if not v in notshift_vars]

    X = add_shifted_variables(X, shifts, variables=shift_vars)

    Xda, yda = reshape_scalar_predictand(X, y)  # reshape into dimensions (time, feature)
    return Xda, yda


X, y = feature_preproc(era5, glofas, '2005', '2010')
"""
"""PROBLEM"""
# need to fix dimensionality problem, since GloFAS is a daily dataset, whereas ERA5 iss an hourly dataset. I would want to uppscale instead of downscale.
X.features

# Splitting the dataset into training, test, and validation

period_train = dict(time=slice('2005', '2012'))
period_valid = dict(time=slice('2014', '2015'))
period_test = dict(time=slice('2015', '2016'))

X_train, y_train = X.loc[period_train], y.loc[period_train]
X_valid, y_valid = X.loc[period_valid], y.loc[period_valid]
X_test, y_test = X.loc[period_test], y.loc[period_test]

""""""
# Visualizing the distribution of discharge
import seaborn as sns

sns.distplot(y)
plt.ylabel('density')
plt.xlim([0, 150])
plt.title('distribution of discharge')
plt.plot()
plt.savefig('../images/generic/distribution_dis.png', dpi=600, bbox_inches='tight')

from sklearn.preprocessing import StandardScaler
import keras
from keras.layers.core import Dropout


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


class DenseNN(object):
    def __init__(self, **kwargs):
        self.output_dim = 1
        self.xscaler = StandardScaler()
        self.yscaler = StandardScaler()

        model = keras.models.Sequential()
        self.cfg = kwargs
        hidden_nodes = self.cfg.get('hidden_nodes')

        model.add(keras.layers.Dense(hidden_nodes[0],
                                     activation='tanh'))
        model.add(keras.layers.BatchNormalization())
        model.add(Dropout(self.cfg.get('dropout', None)))

        for n in hidden_nodes[1:]:
            model.add(keras.layers.Dense(n, activation='tanh'))
            model.add(keras.layers.BatchNormalization())
            model.add(Dropout(self.cfg.get('dropout', None)))
        model.add(keras.layers.Dense(self.output_dim,
                                     activation='linear'))
        opt = keras.optimizers.Adam()

        model.compile(loss=self.cfg.get('loss'), optimizer=opt)
        self.model = model

        self.callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        min_delta=1e-2, patience=100, verbose=0, mode='auto',
                                                        baseline=None, restore_best_weights=True), ]

    def score_func(self, X, y):
        """Calculate the RMS error

        Parameters
        ----------
        xr.DataArrays
        """
        ypred = self.predict(X)
        err_pred = ypred - y

        # NaNs do not contribute to error
        err_pred = err_pred.where(~np.isnan(err_pred), 0.)
        return float(np.sqrt(xr.dot(err_pred, err_pred)))

    def predict(self, Xda, name=None):
        """Input and Output: xr.DataArray

        Parameters
        ----------
        Xda : xr.DataArray
            with coordinates (time,)
        """
        X = self.xscaler.transform(Xda.values)
        y = self.model.predict(X).squeeze()
        y = self.yscaler.inverse_transform(y)

        y = add_time(y, Xda.time, name=name)
        return y

    def fit(self, X_train, y_train, X_valid, y_valid, **kwargs):
        """
        Input: xr.DataArray
        Output: None
        """

        print(X_train.shape)
        X_train = self.xscaler.fit_transform(X_train.values)
        y_train = self.yscaler.fit_transform(
            y_train.values.reshape(-1, self.output_dim))

        X_valid = self.xscaler.transform(X_valid.values)
        y_valid = self.yscaler.transform(
            y_valid.values.reshape(-1, self.output_dim))

        return self.model.fit(X_train, y_train,
                              validation_data=(X_valid, y_valid),
                              epochs=self.cfg.get('epochs', 1000),
                              batch_size=self.cfg.get('batch_size'),
                              callbacks=self.callbacks,
                              verbose=0, **kwargs)


config = dict(hidden_nodes=(64,),
              dropout=0.25,
              epochs=300,
              batch_size=50,
              loss='mse')

m = DenseNN(**config)

hist = m.fit(X_train, y_train, X_valid, y_valid)

# Summary of Model
m.model.summary()

m.model.save('modeltest.h5')
# save model
from keras.utils import plot_model

# plot Graph of Network
from keras.utils import plot_model

plot_model(m.model, to_file='./images/generic/model.png', show_shapes=True)

h = hist.model.history

# Plot training & validation loss value
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(h.history['loss'], label='loss')
ax.plot(h.history['val_loss'], label='val_loss')
plt.title('Learning curve')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
plt.legend(['Training', 'Validation'])
ax.set_yscale('log')
plt.savefig('./images/danube/learningcurve.png', dpi=600, bbox_inches='tight')

yaml_string = m.model.to_yaml()

with open('./models/keras-config.yml', 'w') as f:
    yaml.dump(yaml_string, f)

with open('./models/model-config.yml', 'w') as f:
    yaml.dump(config, f, indent=4)

from contextlib import redirect_stdout

with open('./models/summary.txt', "w") as f:
    with redirect_stdout(f):
        m.model.summary()

from functions.plot import plot_multif_prediction

from functions.utils_floodmodel import generate_prediction_array

y_pred_train = m.predict(X_train)
y_pred_train = generate_prediction_array(y_pred_train, y_train, forecast_range=14)

plt.plot(y_train, y_pred_train)

y_pred_valid = m.predict(X_valid)
y_pred_valid = generate_prediction_array(y_pred_valid, y_orig, forecast_range=14)

y_pred_test = m.predict(X_test)
y_pred_test = generate_prediction_array(y_pred_test, y_test, forecast_range=14)

from functions.plot import plot_multif_prediction

title = 'Setting: Time-Delay Neural Net: 64 hidden nodes, dropout 0.25'
plot_multif_prediction(y_pred_test, y_test, forecast_range=14, title=title)

# Update shapefile with Shapefile in order to include the predictions for each individual basin
