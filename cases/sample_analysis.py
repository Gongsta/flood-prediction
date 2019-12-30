import sys
sys.path.append('/Users/stevengong/Desktop/flood-prediction')
from functions.floodmodel_utils import get_basin_mask, shift_and_aggregate, generate_prediction_array, reshape_scalar_predictand
import xarray as xr

# Creating the model
import numpy as np

from dask.distributed import Client, LocalCluster

cluster = LocalCluster()  # n_workers=10, threads_per_worker=1,
client = Client()
print(client.scheduler_info()['services'])

#Connecting my client to the cluster does not work :((
#client = Client("tcp://192.168.0.112:8786")  # memory_limit='16GB',


#Loading our data
#The open_mfdataset function automatically combines the many .nc files, the * represents the value that varies
ds = xr.open_dataset('../data/features_xy.nc')


y_orig = ds['dis']
y = y_orig.copy()
X = ds.drop(['dis', 'dis_diff'])

#try removing this and see if it yields the same results. No removing it will cause the plot at ./images/test/failed_model.png
y = y.diff('time', 1)

Xda, yda = reshape_scalar_predictand(X, y)
#Xda is an array of features

period_train = dict(time=slice(None, '2005'))
period_valid = dict(time=slice('2006', '2011'))
period_test = dict(time=slice('2012', '2016'))



X_train, y_train = Xda.loc[period_train], yda.loc[period_train]
X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]
X_test, y_test = Xda.loc[period_test], yda.loc[period_test]



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.core import Dropout
from keras.constraints import MinMaxNorm, nonneg
from sklearn.externals.joblib import dump, load

from functions.floodmodel_utils import add_time

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
              batch_size=90,
              loss='mse')

m = DenseNN(**config)
hist = m.fit(X_train, y_train, X_valid, y_valid)

#Saving the feature scaling settings for later
dump(m.xscaler, './models/test/x_std_scaler.bin', compress=True)
dump(m.yscaler, './models/test/y_std_scaler.bin', compress=True)

import matplotlib.pyplot as plt
h = hist.model.history

# Plot training & validation loss value
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(h.history['loss'], label='loss')
ax.plot(h.history['val_loss'], label='val_loss')
plt.title('Learning curve')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
plt.legend(['Training', 'Validation'])
ax.set_yscale('log')


#serialize model to yaml file
model_yaml = m.model.to_yaml()
with open('./models/test/testmodel.yml', 'w') as yaml_file:
    yaml_file.write(model_yaml)

#Serializing weights to h5 file
m.model.save_weights('./models/test/weights.h5')


#LATER ON... LOADING THE WEIGHTS
yaml_model = open('./models/test/testmodel.yml', 'r').read()
from keras.models import model_from_yaml
loaded_model = model_from_yaml(yaml_model)
loaded_model.load_weights('./models/test/weights.h5')
m.model = loaded_model
m.xscaler.fit_transform(X_train.values)
m.yscaler.fit_transform(y_train.values.reshape(-1, m.output_dim))

y_pred_train = m.predict(X_train)
y_pred_valid = m.predict(X_valid)
y_pred_test = m.predict(X_test)
y_pred_test = np.concatenate(([y_orig.sel(time='2012-01-01').values], y_pred_test)).cumsum()

#Trying to remove this generate_prediction_array function, add a function where we loop through the function and predict the future values
#We have to concatenate and call the cumulative sum in order to revert to the orginal function as we have been predicting the variation of discharge
plt.plot(y_pred_test)
plt.plot(np.concatenate(([y_orig.sel(time='2012-01-01').values], y_test)).cumsum())




date_init = '2013-05-29'
            date_end = '2013-06-28'
            fr_dir = '2013052900'

"""
from functions.plot import plot_multif_prediction
title='Setting: Time-Delay Neural Net: 64 hidden nodes, dropout 0.25'
plot_multif_prediction(y_pred_test, y_orig, forecast_range=30, title=title)
plt.savefig('./images/test/multif_prediction.png', dpi=600)

"""





