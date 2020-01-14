#Appending the path to the system so that the program can recognize the files
import sys
sys.path.append('/Users/stevengong/Desktop/flood-prediction')
from functions.floodmodel_utils import get_basin_mask, shift_and_aggregate, add_time
import xarray as xr
import numpy as np
#Creating a Dask local cluster for parallel computing (making the computations later on much faster)
from dask.distributed import Client, LocalCluster

#cluster = LocalCluster()  #processes=4 threads=4, memory=8.59 GB
client = Client("tcp://169.45.50.121:8786")
print(client.scheduler_info()['services'])


days_intake_length = 60
forecast_day = 14

# #SAVING THE DATASET
# era5_loaded = xr.open_mfdataset('/Volumes/portableHardDisk/data/Elbe/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_*_*.nc', combine='by_coords', chunks={'latitude':4, 'longitude':4,'time':4})
# glofas_loaded = xr.open_mfdataset('/Volumes/portableHardDisk/data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')
#
# era5_loaded = era5_loaded.chunk(25)
#
# era5_loaded.to_zarr('/Volumes/Seagate Backup Plus Drive/weatherdata/Elbe', consolidated=True)
# glofas_loaded.to_zarr('/Volumes/Seagate Backup Plus Drive/weatherdata/glofas', consolidated=True)
#
# #Call this in console
# #gcloud auth login
# #gsutil -m cp -r /Volumes/portableHardDisk/weatherdata/Elbe/ gs://weather-data-copernicus/
# #gsutil -m cp -r /Volumes/portableHardDisk/weatherdata/glofas/ gs://weather-data-copernicus/
#
# If the command fails during the upload, and you've waited for 4 hours or more.., you can upload simply the missing files by calling
# gsutil rsync /Volumes/portableHardDisk/weatherdata/Elbe/ gs://weather-data-copernicus/ . Found the information here https://readthedocs.org/projects/gcsfs/downloads/pdf/stable/
#
#


#Loading the dataset

#Does not work, takes too much time to load
"""
#this is suggested in the pangeo documentation
import xarray as xr
import fsspec
ds = xr.open_zarr(fsspec.get_mapper('gcs://weather-data-copernicus/Elbe'))
#ds = xr.open_zarr('/Volumes/Seagate Backup Plus Drive/weatherdata/glofas')
"""

#TODO: Fix this part of the code to make google work

import fsspec
ds = xr.open_zarr(fsspec.get_mapper('gcs://pangeo-data/mydataset'))

import gcsfs
from gcsfs import GCSFileSystem
fs = GCSFileSystem(project="flood-prediction-263210", token='cache')
gcsmapglofas = gcsfs.mapping.GCSMap('weather-data-copernicus/glofas/', gcs=fs, check=True, create=False)
glofas_loaded = xr.open_zarr(gcsmapglofas)
gcsmapElbe = gcsfs.mapping.GCSMap('weather-data-copernicus/Elbe', gcs=fs, check=True, create=False)
era5_loaded = xr.open_zarr(gcsmapElbe)


era5_loaded = client.persist(era5_loaded)
glofas_loaded = client.persist(glofas_loaded)
#DATA PREPROCESSING

#Doing this for debugging purposes
glofas = glofas_loaded.copy()
era5 = era5_loaded.copy()

#WHICH ONE TO USE? client.persist(glofas) or glofas,persist()?
glofas.persist()
era5.persist()
#Renaming the glofas coordinates from 'lon' to 'longitude' so that it is identical with era5's coordinates, which are spelled 'longitude' and 'latitude'
glofas = glofas.rename({'lon' : 'longitude'})
glofas = glofas.rename({'lat': 'latitude'})


"""At this point, the loaded GloFAS dataset includes all coordinates from around the world. We are only interested in looking at data from the Elbe basin, hence 
we use the get_mask_of_basin function to drop all coordinates outside of the Elbe basin. Similarly, when era5 is downloaded, it is a square/rectangular area, so we need
 to remove all data that is not part of the basin so we can use relevant and predictive data."""
elbe_basin_mask = get_basin_mask(glofas['dis24'].isel(time=0), 'Elbe')
glofas = glofas.where(elbe_basin_mask, drop=True)

#We first need to interpolate the data because there are different dimension sizes (era5 is 6x6 gridpoints whereas glofas provides 15x15 gridpoints
#The following code would return an error:  >> era5 = era5.where(elbe_area, drop=True)
#IN CASE YOU DON'T KNOW: #nterpolation is an estimation of a value within two known values in a sequence of values.
#Polynomial interpolation is a method of estimating values between known data points
era5 = era5.interp(latitude=glofas.latitude, longitude=glofas.longitude).where(elbe_basin_mask, drop=True)


# Taking the average latitude and longitude of the area to reduce the dimensionality of our dataset
era5 = era5.mean(['latitude', 'longitude'])
glofas = glofas.mean(['latitude', 'longitude'])

#Adding additional parameters so that our model can get better at predicting
era5['lsp-4-11'] = shift_and_aggregate(era5['lsp'], shift=4, aggregate=8)
era5['lsp-12-25'] = shift_and_aggregate(era5['lsp'], shift=12, aggregate=14)
era5['lsp-26-55'] = shift_and_aggregate(era5['lsp'], shift=26, aggregate=30)
era5['lsp-56-180'] = shift_and_aggregate(era5['lsp'], shift=56, aggregate=125)



y_orig = glofas['dis24']
#Making a copy because y will be transformed to represent the variation of discharge. The model will be predicting the variation of discharge, not the quantity of discharge itself
y = y_orig.copy()
y = y.diff('time', 1)

#Era5 will be the predictor dataset
X = era5

y.compute()
X.compute()

from functions.floodmodel_utils import reshape_scalar_predictand
#merges both values in time, since GLOFAS is daily while era5 is hourly data
#TRY TO RESHAPE WITHOUT TAKING THE MEAN LATITUDE AND LONGITUDE OF X
Xda, yda = reshape_scalar_predictand(X, y)


"""PROBLEM"""
# need to fix dimensionality problem, since GloFAS is a daily dataset, whereas ERA5 iss an hourly dataset. I would want to uppscale instead of downscale.
X.features

# Splitting the dataset into training, test, and validation
period_train = dict(time=slice('1999', '2005'))
period_valid = dict(time=slice('2006', '2011'))
period_test = dict(time=slice('2012', '2016'))

#Train-test split
# X_train, y_train = Xda.loc[period_train], yda.loc[period_train]
# X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]
# X_test, y_test = Xda.loc[period_test], yda.loc[period_test]

dataset_train = yda.loc[period_train]
dataset_valid = yda.loc[period_valid]
dataset_test = yda.loc[period_test]

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
plt.savefig('./images/Elbe/elbeLSTMlearningcurve.png', dpi=600, bbox_inches='tight')

#TODO: Save the model architecture
from keras.utils import plot_model

plot_model(regressor, to_file='./images/Elbe/model_architecture/elbeLSTM.png', show_shapes=True)


# serialize model to YAML
regressor_yaml = regressor.to_yaml()
with open("./models/sample-analysis/elbe_LSTM.yaml", "w") as yaml_file:
    yaml_file.write(regressor_yaml)
# serialize weights to HDF5
regressor.save_weights("./models/elbe_LSTM.h5")
#Seialize feature scaling weights


#LATER ON... LOADING THE WEIGHTS
regressor_model = open('./models/elbe_LSTM.yaml', 'r').read()
from keras.models import model_from_yaml
loaded_regressor = model_from_yaml(regressor_model)
loaded_regressor.load_weights('./models/elbe_LSTM.h5')
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
plt.savefig('./images/Elbe/elbe_LSTM_discharge_validationdata.png', dpi=600)


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
plt.savefig('./images/Elbe/elbe_LSTM_ver5_discharge_testdata.png', dpi=600)





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
plt.savefig('./images/Elbe/elbeLSTM_ver5_14thday_discharge_testdata.png', dpi=600)




