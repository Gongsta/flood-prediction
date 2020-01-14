#THe purpose of this code is to be able to predict to coastal floods based on the data available of current water levels.

import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

days_intake_length = 60
forecast_day = 14

dataset = xr.open_mfdataset("coastal/coastsample/*", combine='by_coords')

#Visualizing the sea elevations across the world

plt.figure(figsize=(10,5))
ax = plt.axes(projection=ccrs.PlateCarree())
dataset['adt'].isel(time=0).plot(ax=ax,transform=ccrs.PlateCarree())
plt.savefig('./images/coasts/absolutedynamictopography1995.png')

plt.figure(figsize=(10,5))
ax = plt.axes(projection=ccrs.PlateCarree())
dataset['sla'].isel(time=0).plot(ax=ax, transform = ccrs.PlateCarree())
plt.savefig('./images/coasts/sealevelanomaly1995.png')


#These are the two parts of the dataset that I plan on using
mean_anomaly = dataset['sla']
elevation = dataset['adt']


#Selecting area of interest (roi)

#For now, i deal with the coordinate problem by taking the mean of the worldwide data
mean_anomaly = dataset['sla'].mean(['latitude', 'longitude'])
elevation = dataset['adt'].mean(['latitude', 'longitude'])


#Train-test split
period_train = dict(time=slice(None, '2005'))
period_valid = dict(time=slice('2006', '2011'))
period_test = dict(time=slice('2012', '2016'))


dataset_train = elevation.loc[period_train]
dataset_valid = elevation.loc[period_valid]
dataset_test = elevation.loc[period_test]




X_train = []
y_train = []

#Applying feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
dataset_train_scaled = sc.fit_transform(dataset_train.values.reshape(-1,1))

for i in range(days_intake_length, len(dataset_train)-forecast_day):
    #OR DO
    #    new_X_train.append(X_train[i-60:i,0]) ? It keeps the xarrays

    X_train.append(dataset_train_scaled[i-days_intake_length:i, 0])
    y_train.append(dataset_train_scaled[i:i+forecast_day, 0])

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

regressor.add(Dense(units=forecast_day))

regressor.compile(optimizer='adam', loss='mean_squared_error')

#regressor.fit(X_train.values, y_train.values, epochs=100, batch_size=32)

regressor.fit(X_train, y_train, epochs=100, batch_size=32)




# serialize model to YAML
regressor_yaml = regressor.to_yaml()
with open("./models/coastal/LSTMsample.yaml", "w") as yaml_file:
    yaml_file.write(regressor_yaml)
# serialize weights to HDF5
regressor.save_weights("./models/coastal/LSTMsample.h5")
#Seialize feature scaling weights


#LATER ON... LOADING THE WEIGHTS
regressor_model = open('./models/coastal/LSTMsample.yaml', 'r').read()
from keras.models import model_from_yaml
loaded_regressor = model_from_yaml(regressor_model)
loaded_regressor.load_weights('./models/coastal/LSTMsample.h5')
regressor = loaded_regressor




#Fitting the test values on the model
dataset_total = np.concatenate((dataset_train, dataset_valid))

#To test our model on the test set, we will need to use part of the training set. More specifically, since our model has been trained on the
#60 previous days, we will need exactly 60 days out of the training set, in addition to all of the test set.
inputs = dataset_total[len(dataset_total)-len(dataset_valid)-60:]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
y_valid = []
X_valid = []

for i in range(days_intake_length, len(inputs)-forecast_day+1):
    X_valid.append(inputs[i-days_intake_length:i, 0])
    y_valid.append(inputs[i:i+forecast_day, 0])


X_valid, y_valid = np.array(X_valid), np.array(y_valid)

#This reshape is necessary since this is the input format the model expects
X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

y_pred_valid = regressor.predict(X_valid)
y_pred_valid = sc.inverse_transform(y_pred_valid)
y_valid = sc.inverse_transform(y_valid)


#Making the predictions on the test set (where there was a flood event)
dataset_total_2 = np.concatenate((dataset_valid, dataset_test))

inputs = dataset_total_2[len(dataset_total_2)-len(dataset_test)-60:]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
y_test = []
X_test = []

for i in range(days_intake_length, len(inputs)-forecast_day+1):
    X_test.append(inputs[i-60:i, 0])
    y_test.append(inputs[i:i+forecast_day,0])

X_test, y_test = np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

y_pred_test = regressor.predict(X_test)
y_pred_test = sc.inverse_transform(y_pred_test)
y_test = sc.inverse_transform(y_test.reshape(-1,1))


#See your results!

#TODO, create a function for the time shifted array of forecasts

#NOTE: In this lSTM model, the plots will look different from the other plots. This is because the model returns an array of
#forecasts rather than a single value. THe LSTM model is trained not only on predicting a single day, but multiple days.

import matplotlib.pyplot as plt

for i in range(0, len(y_valid), 14):
    plt.plot(y_pred_valid[i])

import matplotlib.pyplot as plt

#Plotting the real validation values
dataset_valid.plot(label='True discharge', figsize=(15,5))

#Plotting the validation predicted values
for i in range(0, len(y_pred_valid), forecast_day):
    y_pred_valid_xr = xr.DataArray(y_pred_valid[i], dims=('time'), coords={'time': dataset_valid.time.values[i:i+forecast_day]})
    y_pred_valid_xr.plot()


plt.title('LSTM model 14-day forecasts with 60-day timesteps')
plt.legend(loc="upper left")
plt.savefig('./images/sampleanalysis/LSTM_ver5_discharge_validationdata.png', dpi=600)


#Plotting the real test values
#This line of code is useless, you can just call dataset_test.plot()
#y_test_xr = xr.DataArray(y_test.reshape(-1), dims=('time'), coords={'time': dataset_test.time.values[forecast_day:]})
dataset_test.plot(label="True discharge", figsize=(15,5))
#Plotting the test predicated values

for i in range(0, len(y_pred_test), forecast_day):
    y_pred_test_xr = xr.DataArray(y_pred_test[i], dims=('time'), coords={'time': dataset_test.time.values[i:i+forecast_day]})
    y_pred_test_xr.plot()


plt.title('LSTM model 14-day forecasts with 60-day timesteps')
plt.legend(loc='upper left')
plt.savefig('./images/sampleanalysis/LSTM_ver5_discharge_testdata.png', dpi=600)





#Plotting the forecast of the 14th day of the test set
dataset_test.plot(label="True discharge", figsize=(15,5))
#Plotting the test predicated values

forecast_predictions = []
for i in range(len(y_pred_test)):
    forecast_predictions.append(y_pred_test[i][13])

forecast_predictions = np.array(forecast_predictions)

forecast_predictions_xr = xr.DataArray(forecast_predictions, dims=('time'), coords={'time': dataset_test.time.values[13:]})
forecast_predictions_xr.plot(label="14th day predicted discharge")


plt.title('LSTM model 14th-day forecasts with 60-day timesteps')
plt.legend(loc='upper left')
plt.savefig('./images/sampleanalysis/LSTM_ver5_14thday_discharge_testdata.png', dpi=600)






