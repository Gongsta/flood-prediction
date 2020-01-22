"""Here is the intuition behind this model. I will be making predictions for each pixel of my basin. The input will be a group of pixels whereas
the output will be the prediction of the discharge value at that pixel."""


from functions.floodmodel_utils import get_basin_mask, get_river_mask, reshape_scalar_predictand, reshape_multiday_predictand
import xarray as xr
#Creating a Dask local cluster for parallel computing (making the computations later on much faster)
from dask.distributed import Client, LocalCluster



#HYPERPARAMETERS
days_intake_length = 60
forecast_day = 14


#LIBRARY IMPORTS
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



# No AWS keys required
client = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))


import boto3
client = boto3.client('s3') #low-level functional API
glofas = client.get_object(Bucket='flood-prediction', Key='glofas/2002/CEMS_ECMWF_dis24_20020620_glofas_v2.1.nc')



#DATA PREPROCESSING

glofas_loaded = xr.open_mfdataset("/Volumes/portableHardDisk/data/glofas/*/*.nc", combine="by_coords")
era5_loaded = xr.open_mfdataset("/Volumes/portableHardDisk/data/Elbe/*.nc", combine="by_coords")
glofas = glofas_loaded.copy()
era5 = era5_loaded.copy()


#Renaming the glofas coordinates from 'lon' to 'longitude' so that it is identical with era5's coordinates, which are spelled 'longitude' and 'latitude'
glofas = glofas.rename({'lon' : 'longitude'})
glofas = glofas.rename({'lat': 'latitude'})


"""At this point, the loaded GloFAS dataset includes all coordinates from around the world. We are only interested in looking at data from the Elbe basin, hence 
we use the get_mask_of_basin function to drop all coordinates outside of the Elbe basin. Similarly, when era5 is downloaded, it is a square/rectangular area, so we need
 to remove all data that is not part of the basin so we can use relevant and predictive data."""
elbe_basin_mask = get_basin_mask(glofas['dis24'].isel(time=0), 'Elbe')
elbe_river_mask = get_river_mask(glofas['dis24'])

glofas = glofas.where(elbe_basin_mask, drop=True).where(elbe_river_mask, drop=True)



y_orig = glofas['dis24']
#Making a copy because y will be transformed to represent the variation of discharge. The model will be predicting the variation of discharge, not the quantity of discharge itself
y = y_orig.copy()

#Era5 will be the predictor dataset
X = era5

Xda, yda = reshape_scalar_predictand(X, y)
Xda, yda = reshape_multiday_predictand(X, y)



period_train = dict(time=slice(None, '2005'))
period_valid = dict(time=slice('2006', '2011'))
period_test = dict(time=slice('2012', '2016'))


X_train, y_train = Xda.loc[period_train], yda.loc[period_train]
X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]
X_test, y_test = Xda.loc[period_test], yda.loc[period_test]


#TODO: X_train and y_train still should have still have dimensions (latitude, longitude, time) at this point

#Using exclusively the discharge to predict the discharge
#We do this transforming the dataset, inputting 60 previous days for X_train

#Applying feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X_train_scaled = sc.fit_transform(X_train)
y_train_scaled = y_train
#I'm hesitating whether I should apply standard scaling on the output
# sc2 = MinMaxScaler(feature_range=(0,1))
# y_train_scaled = sc2.fit_transform(y_train.values.reshape(-1,1))
X_train = []
y_train_array = []

#Iterating through each feature, shifting the time for each feature, and appending the time-shifted feature array to X_train for a total of 16 times
for n in range(len(X_train.data_vars)):
    feature_array = []

    for i in range()

    for i in range(60, len(X_train_scaled)):
        feature_array.append(X_train_scaled[i - days_intake_length:i, n])


    X_train.append(feature_array)

y_feature_array = []

for i in range(days_intake_length, len(y_train_scaled)):
    y_feature_array.append(y_train_scaled[i-days_intake_length:i,0])

X_train.append(y_feature_array)

#Creating the transformed y_train array which is shifted by 60 days
for i in range(days_intake_length, len(y_train_scaled)):
    y_train_array.append(y_train_scaled[i,0])

#Transforming the list into numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train_array)

X_train.shape
# (17, 8862, 60)

#Reshaping new_X_train to be supported as an input format for the LSTM
X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[2], X_train.shape[0]))
X_train.shape




#Reshape the model to be able to be fitted in the model
X_train = np.reshape(X_train, (X_train[0], X_train[1], -1))


#LSTM MODEL
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.core import Dropout


regressor = Sequential()

regressor.add(LSTM(units=800, return_sequences= True, input_shape=(X_train.shape[0], X_train.shape[1], X_train.shape[2]))) #(n_sample, n_timesteps, n_features)
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=800, return_sequences= True))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=800, return_sequences= True))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=800))
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

plot_model(regressor, to_file='./images/sampleanalysis/model_architecture/sample_LSTM2.png', show_shapes=True)


#SAVING THE MODEL
# serialize model to YML
regressor_yaml = regressor.to_yaml()
with open("./models/sample-analysis/sample_LSTM2.yaml", "w") as yaml_file:
    yaml_file.write(regressor_yaml)
# serialize weights to HDF5
regressor.save_weights("./models/sample-analysis/sample_LSTM2.h5")
#Seialize feature scaling weights


#LATER ON...LOADING THE MODEL
regressor_model = open('./models/sample-analysis/sample_LSTM2.yaml', 'r').read()
from keras.models import model_from_yaml
loaded_regressor = model_from_yaml(regressor_model)
loaded_regressor.load_weights('./models/sample-analysis/sample_LSTM2.h5')
regressor = loaded_regressor



#TESTING OUR MODEL ON FUTURE DATA


#Predicting on validation data
X_train, y_train = Xda.loc[period_train], yda.loc[period_train]
X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]
X_test, y_test = Xda.loc[period_test], yda.loc[period_test]

#Fitting the test values on the model
X_total = np.concatenate((X_train, X_valid))
#To test our model on the test set, we will need to use part of the training set. More specifically, since our model has been trained on the
#60 previous days, we will need exactly 60 days out of the training set, in addition to all of the test set.
X_inputs = X_total[len(X_total)-len(X_valid)-days_intake_length:]

#Scaling the input data
X_inputs = sc.transform(X_inputs)


y_total = np.concatenate((y_train, y_valid))
y_inputs = y_total[len(y_total)-len(y_valid)-days_intake_length:]

y_inputs = sc2.transform(y_inputs.reshape(-1,1))
#Empty array in which we will append values
X_valid = []
y_valid_feature_array =[]

#Iterating through the 16 features and appending them to X_valid
for n in range(16):
    feature_array = []
    for i in range(days_intake_length, len(X_inputs)):
        #Creates a 2d list with each list inside the list representing a 60 day time interval of the variable
        feature_array.append(X_inputs[i-days_intake_length:i, n])

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




#Model predicting on test data

#I am calling this again because I will require the original datasets that have not been manipulated in order to apply transformation again
X_train, y_train = Xda.loc[period_train], yda.loc[period_train]
X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]
X_test, y_test = Xda.loc[period_test], yda.loc[period_test]

#Making the predictions on the test set (where there was a flood event)
X_total = np.concatenate((X_valid, X_test))
#To test our model on the test set, we will need to use part of the training set. More specifically, since our model has been trained on the
#60 previous days, we will need exactly 60 days out of the training set, in addition to all of the test set.
X_inputs = X_total[len(X_total)-len(X_test)-days_intake_length:]

#Scaling the input data
X_inputs = sc.transform(X_inputs)

#Empty array in which we will append values
X_test = []

#Iterating through the 16 features and appending them to X_valid
for n in range(16):
    feature_array = []
    for i in range(days_intake_length, len(X_inputs)):
        #Creates a 2d list with each list inside the list representing a 60 day time interval of the variable
        feature_array.append(X_inputs[i-days_intake_length:i, n])

    #Appending each feature to the final X_valid list
    X_test.append(feature_array)


X_test = np.array(X_test)

#This reshape is necessary since this is the input format the model expects
X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[2], X_test.shape[0]))

y_pred_test = regressor.predict(X_test)


#See your results!

import matplotlib.pyplot as plt

#Plotting the validation predicted values
X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]
y_pred_valid_xr = xr.DataArray(y_pred_valid.reshape(-1), dims=('time'), coords={'time': X_valid.time.values})
y_pred_valid_xr.plot(label = "Predicted discharge")
#Plotting the real validation values
y_valid_xr = xr.DataArray(y_valid, dims=('time'), coords={'time': X_valid.time.values})
y_valid_xr.plot(label="True discharge")
plt.title('LSTM model prediction trained on time values from 1981-2005')
plt.legend(loc='upper left')
# plt.savefig('./images/sampleanalysis/LSTM_ver3_multivariate_discharge_validationdata.png', dpi=600)






#Plotting the test predicated values

X_test, y_test = Xda.loc[period_test], yda.loc[period_test]

#y_pred_test = np.concatenate(([y_orig.sel(time='2011-12-31').values], y_pred_test.reshape(-1))).cumsum()
y_pred_test_xr = xr.DataArray(y_pred_test, dims=('time'), coords={'time': X_test.time.values})
y_pred_test_xr.plot(label="Predicted discharge")

#Plotting the real test values
y_test_xr = xr.DataArray(y_test, dims=('time'), coords={'time': X_test.time.values})
y_test_xr.plot(label="True discharge")
plt.title('LSTM model prediction trained on time values from 1981-2005')
plt.legend(loc='upper left')
# plt.savefig('./images/sampleanalysis/LSTM_ver3_multivariate_discharge_testdata.png', dpi=600)

