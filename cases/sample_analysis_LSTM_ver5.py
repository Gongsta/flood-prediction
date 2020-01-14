#So far, the previous versions have been predicting a day ahead based on the 60 previous days. Now, let's try to predict 2,3,4 or more days ahead
#based on the 60 previous days and see how the model performs
#Since I had really good results using exclusively discharge values, I will be trying the same again, but with longer day forecasts

#I want to try 2 things: 1. Forecast a single day. The model returns a numerical value. ex: 5002
#                        2. Forecast an array of days. THe model returns an array of predictions: [1 day forecast, 2 day forecast, 3 day forecast, 4 day forecast, etc.]. I will be doing this is version 5

#This version of the model returns an array of predictions rather than a single prediction
days_intake_length = 60
forecast_day = 14
#This second version directly calculates the discharge instead of the variation of discharge
import sys
sys.path.append('/Users/stevengong/Desktop/flood-prediction')


from functions.floodmodel_utils import get_basin_mask, shift_and_aggregate, generate_prediction_array, reshape_scalar_predictand
import xarray as xr
import matplotlib.pyplot as plt


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
#y = y.diff('time', 1)

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

regressor.add(Dense(units=14))

regressor.compile(optimizer='adam', loss='mean_squared_error')

#regressor.fit(X_train.values, y_train.values, epochs=100, batch_size=32)

history = regressor.fit(X_train, y_train, epochs=100, batch_size=32)


#TODO: Save the loss function plot. Test if this works
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history['loss'], label='loss')
ax.plot(history['val_loss'], label='val_loss')
plt.title('Learning curve')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
plt.legend(['Training', 'Validation'])
ax.set_yscale('log')
plt.savefig('./images/sampleanalysis/LSTM5learningcurve.png', dpi=600, bbox_inches='tight')

#TODO: Save the model architecture
from keras.utils import plot_model

plot_model(regressor, to_file='./images/sampleanalysis/model_architecture/LSTM5.png', show_shapes=True)


# serialize model to YAML
regressor_yaml = regressor.to_yaml()
with open("./models/sample-analysis/LSTM5.yaml", "w") as yaml_file:
    yaml_file.write(regressor_yaml)
# serialize weights to HDF5
regressor.save_weights("./models/sample-analysis/LSTM5.h5")
#Seialize feature scaling weights


#LATER ON... LOADING THE WEIGHTS
regressor_model = open('./models/sample-analysis/LSTM5.yaml', 'r').read()
from keras.models import model_from_yaml
loaded_regressor = model_from_yaml(regressor_model)
loaded_regressor.load_weights('./models/sample-analysis/LSTM5.h5')
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




