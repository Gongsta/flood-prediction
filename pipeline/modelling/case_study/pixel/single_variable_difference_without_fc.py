#This model computes the variation of discharge without feature scaling. It was the first one I created
#Somehow, applying feature scaling makes it overfit...
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


#Loading our data
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



#Using exclusively the discharge to predict the discharge
#We do this transforming the dataset, inputting 60 previous days for X_train
import numpy as np
X_train = []
y_train = []


for i in range(60, len(dataset_train)):
    #OR DO
    #    new_X_train.append(X_train[i-60:i,0]) ? It keeps the xarrays

    X_train.append(dataset_train[i-60:i])
    y_train.append(dataset_train[i])

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
y_valid = []
X_valid = []

for i in range(60, len(inputs)):
    X_valid.append(inputs[i-60:i])
    y_valid.append(inputs[i])


X_valid, y_valid = np.array(X_valid), np.array(y_valid)

#This reshape is necessary since this is the input format the model expects
X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

y_pred_valid = regressor.predict(X_valid)


#Making the predictions on the test set (where there was a flood event)
dataset_total_2 = np.concatenate((dataset_valid, dataset_test))

inputs = dataset_total_2[len(dataset_total_2)-len(dataset_test)-60:]
y_test = []
X_test = []

for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
    y_test.append(inputs[i,0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape
#(

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

y_pred_test = regressor.predict(X_test)


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
y_valid = sc.inverse_transform(y_valid.reshape(-1,1))
y_valid = np.concatenate(([y_orig.sel(time='2005-12-31').values], y_valid.reshape(-1))).cumsum()
y_valid = np.delete(y_valid, 0)
y_valid_xr = xr.DataArray(y_valid, dims=('time'), coords={'time': dataset_valid.time.values})
y_valid_xr.plot()
plt.title('LSTM model prediction trained on time values from 1981-2005 with feature scaling')
plt.savefig('./images/sampleanalysis/LSTM_difference_of_discharge_validationdata.png', bbox_inches='tight', dpi=600)


#Plotting the test predicated values
y_pred_test = np.concatenate(([y_orig.sel(time='2011-12-31').values], y_pred_test.reshape(-1))).cumsum()
#We delete the value at the first index of the array since that value represents the value at time 2005-12-31, which we are not interested in.
y_pred_test = np.delete(y_pred_test, 0)
y_pred_test_xr = xr.DataArray(y_pred_test, dims=('time'), coords={'time': dataset_test.time.values})
y_pred_test_xr.plot()

#Plotting the real values
y_test = sc.inverse_transform(y_test.reshape(-1,1))
y_test = np.concatenate(([y_orig.sel(time='2011-12-31').values], y_test.reshape(-1))).cumsum()
y_test = np.delete(y_test, 0)
y_test_xr = xr.DataArray(y_test, dims=('time'), coords={'time': dataset_test.time.values})
y_test_xr.plot()
plt.title('LSTM model prediction trained on time values from 1981-2005 with feature scaling')
plt.savefig('./images/sampleanalysis/LSTM_difference_of_discharge_testdata.png', dpi=600)






