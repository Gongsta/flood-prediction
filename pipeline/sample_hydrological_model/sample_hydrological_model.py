#THIS TRAINS A HYDROLOGICAL MODEL LOCALLY, USING EXCUSIVELY DISCHARGE DATA TO PREDICT DISCHARGE DATA


#HYPERPARAMETERS
days_intake_length = 60
forecast_day = 14


#LIBRARY IMPORTS
from functions.floodmodel_utils import reshape_scalar_predictand
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client, LocalCluster


#Connecting to a cluster to be able to run the code locally/on the cloud
cluster = LocalCluster()  # n_workers=10, threads_per_worker=1,
client = Client()
#Use this line of code if you want to run the code on the cluster
# client = Client("tcp://169.45.50.121:8786")


#This will tell you where you dashboard will be so you can visualize your model being run
print(client.scheduler_info()['services'])


#Loading our data

#Loading locally
ds = xr.open_dataset('../data/features_xy.nc')


#Loading on the cloud (Uncomment if you want to do so)
"""
import gcsfs
from gcsfs import GCSFileSystem
fs = GCSFileSystem(project="flood-prediction-263210", token='anon')
gcsmapds = gcsfs.mapping.GCSMap('weather-data-copernicus/sample_dataset', gcs=fs, check=True, create=False)
ds = xr.open_zarr(gcsmapds)
"""

#Selecting our X and y values from the dataset. Y has the river discharge values
y_orig = ds['dis']

#y_orig will be used later as y will be applied standard scaling
y = y_orig.copy()

#Selecting the periods for each dataset
period_train = dict(time=slice(None, '2005'))
period_valid = dict(time=slice('2006', '2011'))
period_test = dict(time=slice('2012', '2016'))

#Train, valid and test split
dataset_train = y.loc[period_train]
dataset_valid = y.loc[period_valid]
dataset_test = y.loc[period_test]



X_train = []
y_train = []

#Applying feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
dataset_train_scaled = sc.fit_transform(dataset_train.values.reshape(-1,1))

#Loop that appends an array of discharge values for a given range of days
for i in range(days_intake_length, len(dataset_train)-forecast_day):

    X_train.append(dataset_train_scaled[i-days_intake_length:i, 0])
    y_train.append(dataset_train_scaled[i:i+forecast_day, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping new_X_train to be supported as an input format for the LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



#LSTM MODEL

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.core import Dropout


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

history = regressor.fit(X_train, y_train, epochs=100, batch_size=32)


#Visualizing the loss functions
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history['loss'], label='loss')
ax.plot(history['val_loss'], label='val_loss')
plt.title('Learning curve')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
plt.legend(['Training', 'Validation'])
ax.set_yscale('log')
# plt.savefig('./images/sampleanalysis/LSTM5learningcurve.png', dpi=600, bbox_inches='tight')

#TODO: Save the model architecture
from keras.utils import plot_model

plot_model(regressor, to_file='./images/sampleanalysis/model_architecture/LSTM5.png', show_shapes=True)


#SAVING THE MODEL
# serialize model to YML
regressor_yaml = regressor.to_yaml()
with open("./models/sample-analysis/LSTM5.yaml", "w") as yaml_file:
    yaml_file.write(regressor_yaml)
# serialize weights to HDF5
regressor.save_weights("./models/sample-analysis/LSTM5.h5")
#Seialize feature scaling weights


#LATER ON...LOADING THE MODEL
regressor_model = open('./models/sample-analysis/LSTM5.yaml', 'r').read()
from keras.models import model_from_yaml
loaded_regressor = model_from_yaml(regressor_model)
loaded_regressor.load_weights('./models/sample-analysis/LSTM5.h5')
regressor = loaded_regressor



#TESTING OUR MODEL ON FUTURE DATA


#Making predictions on validation data
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





#Visualize your results!

#PLOT 1 (discontinued lines validation set)


dataset_valid.plot(label='True discharge', figsize=(15,5))


for i in range(0, len(y_pred_valid), forecast_day):
    y_pred_valid_xr = xr.DataArray(y_pred_valid[i], dims=('time'), coords={'time': dataset_valid.time.values[i:i+forecast_day]})
    y_pred_valid_xr.plot()

plt.title('LSTM model' + str(forecast_day) +'-day forecasts with' + str(days_intake_length)+ '-day timesteps')
plt.legend(loc="upper left")
plt.savefig('./images/sampleanalysis/LSTM_ver5_discharge_validationdata.png', dpi=600)



#PLOT 2 (discontinued lines on test set)
dataset_test.plot(label="True discharge", figsize=(15,5))

for i in range(0, len(y_pred_test), forecast_day):
    y_pred_test_xr = xr.DataArray(y_pred_test[i], dims=('time'), coords={'time': dataset_test.time.values[i:i+forecast_day]})
    y_pred_test_xr.plot()

plt.title('LSTM model ' + str(forecast_day) +'-day forecasts with 60-day timesteps')
plt.legend(loc='upper left')
plt.savefig('./images/sampleanalysis/LSTM_ver5_discharge_testdata.png', dpi=600)



#Plot 3 (Continuous values)
dataset_test.plot(label="True discharge", figsize=(15,5))
#Plotting the test predicated values

forecast_predictions = []
for i in range(len(y_pred_test)):
    forecast_predictions.append(y_pred_test[i][forecast_day-1])

forecast_predictions = np.array(forecast_predictions)

forecast_predictions_xr = xr.DataArray(forecast_predictions, dims=('time'), coords={'time': dataset_test.time.values[forecast_day-1:]})
forecast_predictions_xr.plot(label="14th day predicted discharge")


plt.title('LSTM model '+ str(forecast_day) +'th-day forecasts with 60-day timesteps')
plt.legend(loc='upper left')
plt.savefig('./images/sampleanalysis/LSTM_ver5_14thday_discharge_testdata.png', dpi=600)




