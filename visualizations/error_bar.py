#THIS WAS DONE WITH  SAMPLE ANALYSIS LSTM VER5
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




"""
Demo of errorbar function with different ways of specifying error bars.

Errors can be specified as a constant value (as shown in `errorbar_demo.py`),
or as demonstrated in this example, they can be specified by an N x 1 or 2 x N,
where N is the number of data points.

N x 1:
    Error varies for each point, but the error values are symmetric (i.e. the
    lower and upper values are equal).

2 x N:
    Error varies for each point, and the lower and upper limits (in that order)
    are different (asymmetric case)

In addition, this example demonstrates how to use log scale with errorbar.
"""
import numpy as np
import matplotlib.pyplot as plt

# example data
x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)
# example error bar values that vary with x-position
error = 0.1 + 0.2 * x
# error bar values w/ different -/+ errors
lower_error = 0.4 * error
upper_error = error
asymmetric_error = [lower_error, upper_error]

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
ax0.errorbar(x, y, yerr=error, fmt='-o')
ax0.set_title('variable, symmetric error')

ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')
ax1.set_title('variable, asymmetric error')
ax1.set_yscale('log')
plt.show()




#Calculating the error
from sklearn.metrics import mean_squared_error
MSE= mean_squared_error(dataset_test[18:63], forecast_predictions_xr.values[5:50])

#The error per data point
RMSE = MSE**(0.5)/len(dataset_test[18:63])

#TODO: Determine what metric you should be using, MSE, standard deviation, etc.
#Plotting the  error
fig, ax = plt.subplots(sharex=True, figsize=(15,5))
ax.errorbar(forecast_predictions_xr.time.values[5:50], forecast_predictions_xr.values[5:50], yerr =RMSE,fmt='-o', capsize=5)
ax.set_title('variable, symmetric error')
plt.show()