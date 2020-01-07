#Same LSTM, but directly predicting the discharge value rather than the variation of discharge.


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


Xda, yda = reshape_scalar_predictand(X, y)
#Xda is an array of features

period_train = dict(time=slice(None, '2005'))
period_valid = dict(time=slice('2006', '2011'))
period_test = dict(time=slice('2012', '2016'))



dataset_train = yda.loc[period_train]
dataset_valid = yda.loc[period_valid]
dataset_test = yda.loc[period_test]
"""
I'm trying to reshape by preserving the xarray format, but its not working well so far... the method below uses np arrays as input for the model.


new = xr.DataArray()
new = new.combine_first(X_train[0:60, 0])
new = xr.concat([new, X_train[1:61, 0]], 'new_time')
new = new.dropna('time')

for i in range(60, 150):
    new = xr.concat([new, X_train[i-60:i, 0]], 'new_time')
    new = new.dropna('time')


for i in range (60, 150):
    test = xr.concat([test, ])

new = xr.merge(new, X_train[0])
new = new.combine_first(X_train[0])


"""

#Trying with 1 feature for now, reshaping the format to be supported by LSTM

#Using exclusively the discharge to predict the discharge
#We do this transforming the dataset, inputting 60 previous days for X_train
import numpy as np
new_X_train = []
new_y_train = []

#Applying feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
#y_train = sc.fit_transform(y_train)

for i in range(60, len(X_train)):
    #OR DO
    #    new_X_train.append(X_train[i-60:i,0]) ? It keeps the xarrays

    new_X_train.append(y_train.values[i-60:i])
    new_y_train.append(y_train.values[i])

new_X_train, new_y_train = np.array(new_X_train), np.array(new_y_train)

#Reshaping new_X_train to be supported as an input format for the LSTM
new_X_train = np.reshape(new_X_train, (new_X_train.shape[0], new_X_train.shape[1], 1))




"""This is being tested"""

from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.core import Dropout
from sklearn.externals.joblib import dump, load
from functions.floodmodel_utils import add_time


regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences= True, input_shape=(new_X_train.shape[1], 1)))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.1))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

#regressor.fit(X_train.values, y_train.values, epochs=100, batch_size=32)

regressor.fit(new_X_train, new_y_train, epochs=100, batch_size=32)


#Fitting the test values on the model
dataset_train = y_train.values
dataset_test = y_test.values
dataset_total = np.concatenate((dataset_train, dataset_test))

#To test our model on the test set, we will need to use part of the training set. More specifically, since our model has been trained on the
#60 previous days, we will need exactly 60 days out of the training set, in addition to all of the test set.
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:]
new_y_test = []
new_X_test = []

for i in range(60, len(inputs)):
    new_X_test.append(inputs[i-60:i])
    new_y_test.append(inputs[i])


new_X_test, new_y_test = np.array(new_X_test), np.array(new_y_test)
new_X_test = np.reshape(new_X_test, (new_X_test.shape[0], new_X_test.shape[1], 1))

y_pred_test = regressor.predict(new_X_test)

#Trying to remove this generate_prediction_array function, add a function where we loop through the function and predict the future values
#Plotting the predicted values
y_pred_test = np.concatenate(([y_orig.sel(time='2012-01-01').values], y_pred_test.reshape(-1))).cumsum()
#Plotting the real values
new_y_test = np.concatenate(([y_orig.sel(time='2012-01-01').values], new_y_test)).cumsum()


import matplotlib.pyplot as plt

plt.plot(y_pred_test)
plt.plot(new_y_test)


#Remember that y_train is a DataArray of the difference of discharge values, y_train_original returns the original values of discharge
y_train_original = np.concatenate(([y_orig[0].values], y_train.values)).cumsum()

#Viewing the original discharge rather than the differenrce of discharge.
import matplotlib.pyplot as plt
plt.plot(y_train_original)

#Maybe try this?
y_train_original_xr = xr.DataArray(y_train_original, dims=('time'), coords={'time': np.append(np.datetime64('1981-06-29T00:00:00'), y_train.time.values)})
y_train_original_xr.plot()
#
# class LSTM(object):
#     def __init__(self, **kwargs):
#         self.output_dim = 1
#         self.xscaler = StandardScaler()
#
#         self.yscaler = StandardScaler()
#
#
#         model = Sequential()
#
#         self.cfg = kwargs
#         hidden_nodes = self.cfg.get('hidden_nodes')
#
#         model.add(LSTM(units=50, return_sequences=True, input_shape=2))
#         model.add(keras.layers.BatchNormalization())
#         model.add(Dropout(self.cfg.get('dropout', None)))
#
#         for n in hidden_nodes[1:]:
#             model.add(keras.layers.Dense(n, activation='tanh'))
#             model.add(keras.layers.BatchNormalization())
#             model.add(Dropout(self.cfg.get('dropout', None)))
#         model.add(keras.layers.Dense(self.output_dim,
#                                      activation='linear'))
#         opt = keras.optimizers.Adam()
#
#         model.compile(loss=self.cfg.get('loss'), optimizer=opt)
#         self.model = model
#
#         # A callback is a set of functions to be applied at given stages of the training procedure.
#         # You can use callbacks to get a view on internal states and statistics of the model during training.
#
#         self.callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
#                                                         min_delta=1e-2, patience=100, verbose=0, mode='auto',
#                                                         baseline=None, restore_best_weights=True), ]
#
#     def score_func(self, X, y):
#         """Calculate the RMS error
#
#         Parameters
#         ----------
#         xr.DataArrays
#         """
#         ypred = self.predict(X)
#         err_pred = ypred - y
#
#         # NaNs do not contribute to error
#         err_pred = err_pred.where(~np.isnan(err_pred), 0.)
#         return float(np.sqrt(xr.dot(err_pred, err_pred)))
#
#     def predict(self, Xda, name=None):
#         """Input and Output: xr.DataArray
#
#         Parameters
#         ----------
#         Xda : xr.DataArray
#             with coordinates (time,)
#         """
#         X = self.xscaler.transform(Xda.values)
#         y = self.model.predict(X).squeeze()
#         y = self.yscaler.inverse_transform(y)
#
#         y = add_time(y, Xda.time, name=name)
#         return y
#
#     def fit(self, X_train, y_train, X_valid, y_valid, **kwargs):
#         """
#         Input: xr.DataArray
#         Output: None
#         """
#
#         print(X_train.shape)
#         X_train = self.xscaler.fit_transform(X_train.values)
#         y_train = self.yscaler.fit_transform(
#             y_train.values.reshape(-1, self.output_dim))
#
#         X_valid = self.xscaler.transform(X_valid.values)
#         y_valid = self.yscaler.transform(
#             y_valid.values.reshape(-1, self.output_dim))
#
#         return self.model.fit(X_train, y_train,
#                               validation_data=(X_valid, y_valid),
#                               epochs=self.cfg.get('epochs', 1000),
#                               batch_size=self.cfg.get('batch_size'),
#                               callbacks=self.callbacks,
#                               verbose=0, **kwargs)
#
#
# config = dict(hidden_nodes=(64,),
#               dropout=0.1,
#               epochs=300,
#               batch_size=50,
#               loss='mse')
#
# m = LSTM(**config)
# hist = m.fit(X_train, y_train, X_valid, y_valid)
#
# #Saving the feature scaling settings for later
# dump(m.xscaler, './models/test/x_std_scaler.bin', compress=True)
# dump(m.yscaler, './models/test/y_std_scaler.bin', compress=True)
#
# import matplotlib.pyplot as plt
# h = hist.model.history
#
# # Plot training & validation loss value
# fig, ax = plt.subplots(figsize=(8,4))
# ax.plot(h.history['loss'], label='loss')
# ax.plot(h.history['val_loss'], label='val_loss')
# plt.title('Learning curve')
# ax.set_ylabel('Loss')
# ax.set_xlabel('Epoch')
# plt.legend(['Training', 'Validation'])
# ax.set_yscale('log')
#
#
# #serialize model to yaml file
# model_yaml = m.model.to_yaml()
# with open('./models/test/testmodel.yml', 'w') as yaml_file:
#     yaml_file.write(model_yaml)
#
# #Serializing weights to h5 file
# m.model.save_weights('./models/test/weights.h5')
#
#
# #LATER ON... LOADING THE WEIGHTS
# yaml_model = open('./models/test/testmodel.yml', 'r').read()
# from keras.models import model_from_yaml
# loaded_model = model_from_yaml(yaml_model)
# loaded_model.load_weights('./models/test/weights.h5')
# m.model = loaded_model
# m.xscaler.fit_transform(X_train.values)
# m.yscaler.fit_transform(y_train.values.reshape(-1, m.output_dim))
#
# y_pred_train = m.predict(X_train)
# y_pred_valid = m.predict(X_valid)
# y_pred_test = m.predict(X_test)
#
#






