#This 3rd version of the LSTM incorporates the many other variables such as precipitation into the model

import sys
sys.path.append('/Users/stevengong/Desktop/flood-prediction')
from functions.floodmodel_utils import reshape_scalar_predictand
import xarray as xr
from dask.distributed import Client, LocalCluster

#Connecting to a local cluster
# cluster = LocalCluster()  # n_workers=10, threads_per_worker=1,
client = Client("tcp://169.45.50.121:8786")

#Connecting to an online cluster
#client = Client("tcp://169.45.50.121:8786")  # memory_limit='16GB',

print(client.scheduler_info()['services'])


# #Loading our data locally
# ds = xr.open_dataset('../data/features_xy.nc')

#Loading our data on the cloud
import gcsfs
from gcsfs import GCSFileSystem
fs = GCSFileSystem(project="flood-prediction-263210", token='anon')
gcsmapds = gcsfs.mapping.GCSMap('weather-data-copernicus/sample_dataset', gcs=fs, check=True, create=False)
ds = xr.open_zarr(gcsmapds)

y_orig = ds['dis']
y = y_orig.copy()
X = ds.drop(['dis', 'dis_diff'])

#We lose one value, 1981-01-02, meaning the array is shifted towards the right
#y = y.diff('time', 1)

Xda, yda = reshape_scalar_predictand(X, y)
#Xda is an array of features

period_train = dict(time=slice(None, '2005'))
period_valid = dict(time=slice('2006', '2011'))
period_test = dict(time=slice('2012', '2016'))



X_train, y_train = Xda.loc[period_train], yda.loc[period_train]
X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]
X_test, y_test = Xda.loc[period_test], yda.loc[period_test]

# #I am merging X_train and y_train again after reshaping to apply the standard scaling methodology on the entire dataset
# dataset_train = xr.concat(X_train, y_train)
# dataset_valid = xr.concat(X_valid, y_valid)
# dataset_test = xr.concat(X_test, y_test)


#Using exclusively the discharge to predict the discharge
#We do this transforming the dataset, inputting 60 previous days for X_train
import numpy as np

#Applying feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X_train_scaled = sc.fit_transform(X_train)
sc2 = MinMaxScaler(feature_range=(0,1))
y_train_scaled = sc2.fit_transform(y_train.values.reshape(-1,1))
X_train = []
y_train_array = []

#This is not computed in parallel which causes some problem
#Iterating through each feature, shifting the time for each feature, and appending the time-shifted feature array to X_train for a total of 16 times
for n in range(16):
    feature_array = []

    for i in range(60, len(X_train_scaled)):
        feature_array.append(X_train_scaled[i - 60:i, n])




    X_train.append(feature_array)

y_feature_array = []

for i in range(60, len(y_train_scaled)):
    y_feature_array.append(y_train_scaled[i-60:i,0])

X_train.append(y_feature_array)

#Creating the transformed y_train array which is shifted by 60 days
for i in range(60, len(y_train_scaled)):
    y_train_array.append(y_train_scaled[i,0])

#Transforming the list into numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train_array)

X_train.shape
# (17, 8862, 60)

#Reshaping new_X_train to be supported as an input format for the LSTM
X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[2], X_train.shape[0]))
X_train.shape
#(8862, 60, 17) --> The input of the LSTM is always is a 3D array in the form (batch_size, time_steps, seq_len)

"""This is being tested"""

from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.core import Dropout
from sklearn.externals.joblib import dump, load
from functions.floodmodel_utils import add_time


regressor = Sequential()

#input_shape is (time_steps, input units)
regressor.add(LSTM(units=800, return_sequences= True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=800, return_sequences= True))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=800, return_sequences= True))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=800))
regressor.add(Dropout(0.1))

regressor.add(Dense(units=30))

regressor.compile(optimizer='adam', loss='mean_squared_error')

history = regressor.fit(X_train, y_train, epochs=100, batch_size=32)




# serialize model to YAML
regressor_yaml = regressor.to_yaml()
with open("./models/sample-analysis/LSTM3.yaml", "w") as yaml_file:
    yaml_file.write(regressor_yaml)
# serialize weights to HDF5
regressor.save_weights("./models/sample-analysis/LSTM3.h5")
#Seialize feature scaling weights


#LATER ON... LOADING THE WEIGHTS
regressor_model = open('./models/sample-analysis/LSTM3.yaml', 'r').read()
from keras.models import model_from_yaml
loaded_regressor = model_from_yaml(regressor_model)
loaded_regressor.load_weights('./models/sample-analysis/LSTM3.h5')
regressor = loaded_regressor






#I am calling this again because I will require the original datasets that have not been manipulated in order to apply transformation again
X_train, y_train = Xda.loc[period_train], yda.loc[period_train]
X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]
X_test, y_test = Xda.loc[period_test], yda.loc[period_test]

#Fitting the test values on the model
X_total = np.concatenate((X_train, X_valid))
#To test our model on the test set, we will need to use part of the training set. More specifically, since our model has been trained on the
#60 previous days, we will need exactly 60 days out of the training set, in addition to all of the test set.
X_inputs = X_total[len(X_total)-len(X_valid)-60:]

#Scaling the input data
X_inputs = sc.transform(X_inputs)


y_total = np.concatenate((y_train, y_valid))
y_inputs = y_total[len(y_total)-len(y_valid)-60:]

y_inputs = sc2.transform(y_inputs.reshape(-1,1))
#Empty array in which we will append values
X_valid = []
y_valid_feature_array =[]

#Iterating through the 16 features and appending them to X_valid
for n in range(16):
    feature_array = []
    for i in range(60, len(X_inputs)):
        #Creates a 2d list with each list inside the list representing a 60 day time interval of the variable
        feature_array.append(X_inputs[i-60:i, n])

    #Appending each feature to the final X_valid list
    X_valid.append(feature_array)



for i in range(60, len(X_inputs)):
    y_valid_feature_array.append(y_inputs[i-60:i,0])


X_valid.append(y_valid_feature_array)

X_valid = np.array(X_valid)

#This reshape is necessary since this is the input format the model expects
X_valid = np.reshape(X_valid, (X_valid.shape[1], X_valid.shape[2], X_valid.shape[0]))

y_pred_valid = regressor.predict(X_valid)
# y_pred_valid = sc.inverse_transform(y_pred_valid)
y_pred_valid = sc2.inverse_transform(y_pred_valid)



#I am calling this again because I will require the original datasets that have not been manipulated in order to apply transformation again
X_train, y_train = Xda.loc[period_train], yda.loc[period_train]
X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]
X_test, y_test = Xda.loc[period_test], yda.loc[period_test]

#Making the predictions on the test set (where there was a flood event)
X_total = np.concatenate((X_valid, X_test))
#To test our model on the test set, we will need to use part of the training set. More specifically, since our model has been trained on the
#60 previous days, we will need exactly 60 days out of the training set, in addition to all of the test set.
X_inputs = X_total[len(X_total)-len(X_test)-60:]

#Scaling the input data
X_inputs = sc.transform(X_inputs)

#Empty array in which we will append values
X_test = []

#Iterating through the 16 features and appending them to X_valid
for n in range(16):
    feature_array = []
    for i in range(60, len(X_inputs)):
        #Creates a 2d list with each list inside the list representing a 60 day time interval of the variable
        feature_array.append(X_inputs[i-60:i, n])

    #Appending each feature to the final X_valid list
    X_test.append(feature_array)


X_test = np.array(X_test)

#This reshape is necessary since this is the input format the model expects
X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[2], X_test.shape[0]))

y_pred_test = regressor.predict(X_test)


#See your results!

import matplotlib.pyplot as plt

#Plotting the validation predicted values
#Since the current predicted y values are the difference of discharge, we will call the cumsum() function to revert to the original discharge values. We will need the first previous
#day's value, i.e. 2005-12-31, but we will remove this value in the next line of code

X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]

#y_pred_valid = np.concatenate(([y_orig.sel(time='2005-12-31').values], y_pred_valid.reshape(-1))).cumsum()
#We delete the value at the first index of the array since that value represents the value at time 2005-12-31, which we are not interested in.
#y_pred_valid = np.delete(y_pred_valid, 0)
y_pred_valid_xr = xr.DataArray(y_pred_valid.reshape(-1), dims=('time'), coords={'time': X_valid.time.values})
y_pred_valid_xr.plot(label = "Predicted discharge")

#Plotting the real validation values
#y_valid = np.concatenate(([y_orig.sel(time='2005-12-31').values], y_valid.values)).cumsum()
#y_valid = np.delete(y_valid, 0)
y_valid_xr = xr.DataArray(y_valid, dims=('time'), coords={'time': X_valid.time.values})
y_valid_xr.plot(label="True discharge")
plt.title('LSTM model prediction trained on time values from 1981-2005')
plt.legend(loc='upper left')
plt.savefig('./images/sampleanalysis/LSTM_ver3_multivariate_discharge_validationdata.png', dpi=600)




#Plotting the test predicated values

X_test, y_test = Xda.loc[period_test], yda.loc[period_test]

#y_pred_test = np.concatenate(([y_orig.sel(time='2011-12-31').values], y_pred_test.reshape(-1))).cumsum()
#We delete the value at the first index of the array since that value represents the value at time 2005-12-31, which we are not interested in.
#y_pred_test = np.delete(y_pred_test, 0)
y_pred_test_xr = xr.DataArray(y_pred_test, dims=('time'), coords={'time': X_test.time.values})
y_pred_test_xr.plot(label="Predicted discharge")

#Plotting the real test values
#y_test = np.concatenate(([y_orig.sel(time='2011-12-31').values], y_test)).cumsum()
#y_test = np.delete(y_test, 0)
y_test_xr = xr.DataArray(y_test, dims=('time'), coords={'time': X_test.time.values})
y_test_xr.plot(label="True discharge")
plt.title('LSTM model prediction trained on time values from 1981-2005')
plt.legend(loc='upper left')
plt.savefig('./images/sampleanalysis/LSTM_ver3_multivariate_discharge_testdata.png', dpi=600)

