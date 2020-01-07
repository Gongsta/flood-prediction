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

#We lose one value, 1981-01-02, meaning the array is shifted towards the right
y = y.diff('time', 1)

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
X_train = []
y_train = []

#Applying feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
dataset_train_scaled = sc.fit_transform(dataset_train.values.reshape(-1, 1))

for i in range(60, len(dataset_train)):
    #OR DO
    #    new_X_train.append(X_train[i-60:i,0]) ? It keeps the xarrays

    X_train.append(dataset_train_scaled[i-60:i,0])
    y_train.append(dataset_train_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping new_X_train to be supported as an input format for the LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


"""This is being tested"""

from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.core import Dropout
from sklearn.externals.joblib import dump, load
from functions.floodmodel_utils import add_time


regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences= True, input_shape=(X_train.shape[1], 1)))
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

regressor.fit(X_train, y_train, epochs=100, batch_size=32)


#Making the predictions on the validation set
dataset_total = np.concatenate((dataset_train, dataset_valid))

#To test our model on the test set, we will need to use part of the training set. More specifically, since our model has been trained on the
#60 previous days, we will need exactly 60 days out of the training set, in addition to all of the test set.
inputs = dataset_total[len(dataset_total)-len(dataset_valid)-60:]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
y_valid = []
X_valid = []

for i in range(60, len(inputs)):
    X_valid.append(inputs[i-60:i, 0])
    y_valid.append(inputs[i, 0])


X_valid, y_valid = np.array(X_valid), np.array(y_valid)

#This reshape is necessary since this is the input format the model expects
X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

y_pred_valid = regressor.predict(X_valid)
y_pred_valid = sc.inverse_transform(y_pred_valid)


#Making the predictions on the test set (where there was a flood event)
dataset_total_2 = np.concatenate((dataset_valid, dataset_test))

input = dataset_total_2[len(dataset_total_2)-len(dataset_test)-60:]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
y_test = []
X_test = []

for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
    y_test.append(inputs[i,0])

X_test, y_test = np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

y_pred_test = regressor.predict(X_test)
y_pred_test = sc.inverse_transform(y_pred_test)


#See your results!
import matplotlib.pyplot as plt

#Plotting the validation predicted values
#Since the current predicted y values are the difference of discharge, we will call the cumsum() function to revert to the original discharge values. We will need the first previous
#day's value, i.e. 2005-12-31, but we will remove this value in the next line of code
y_pred_valid = np.concatenate(([y_orig.sel(time='2005-12-31').values], y_pred_valid.reshape(-1))).cumsum()
#We delete the value at the first index of the array since that value represents the value at time 2005-12-31, which we are not interested in.
y_pred_valid = np.delete(y_pred_valid, 0)
y_pred_valid_xr = xr.DataArray(y_pred_valid, dims=('time'), coords={'time': dataset_valid.time.values})
y_pred_valid_xr.plot()

#Plotting the real values
y_valid = np.concatenate(([y_orig.sel(time='2005-12-31').values], y_valid)).cumsum()
y_valid = np.delete(y_valid, 0)
y_valid_xr = xr.DataArray(y_valid, dims=('time'), coords={'time': dataset_valid.time.values})
y_valid_xr.plot()
plt.title('LSTM model prediction trained on time values from 1981-2005')
plt.savefig('./images/sampleanalysis/LSTM_difference_of_discharge_validationdata.png', dpi=600)


#Plotting the test predicated values
y_pred_test = np.concatenate(([y_orig.sel(time='2011-12-31').values], y_pred_test.reshape(-1))).cumsum()
#We delete the value at the first index of the array since that value represents the value at time 2005-12-31, which we are not interested in.
y_pred_test = np.delete(y_pred_test, 0)
y_pred_test_xr = xr.DataArray(y_pred_test, dims=('time'), coords={'time': dataset_test.time.values})
y_pred_test_xr.plot()

#Plotting the real values
y_test = np.concatenate(([y_orig.sel(time='2011-12-31').values], y_test)).cumsum()
y_test = np.delete(y_test, 0)
y_test_xr = xr.DataArray(y_test, dims=('time'), coords={'time': dataset_test.time.values})
y_test_xr.plot()
plt.title('LSTM model prediction trained on time values from 1981-2005')
plt.savefig('./images/sampleanalysis/LSTM_difference_of_discharge_testdata.png', dpi=600)



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






