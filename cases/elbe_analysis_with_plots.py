#Appending the path to the system so that the program can recognize the files
import sys
sys.path.append('/Users/stevengong/Desktop/flood-prediction')
from functions.floodmodel_utils import get_basin_mask, shift_and_aggregate, add_time, get_river_mask
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
#Creating a Dask local cluster for parallel computing (making the computations later on much faster)
from dask.distributed import Client, LocalCluster

cluster = LocalCluster()  #processes=4 threads=4, memory=8.59 GB
client = Client(cluster)
print(client.scheduler_info()['services'])

#This line of code connects the client to a remote cluster
#client = Client("tcp://192.168.0.112:8786")  # memory_limit='16GB',


#Loading our data located in a remote disk
#The open_mfdataset function automatically combines the many .nc files thanks to the power of Dask, which opens the files in parallel, the file aren't loaded until .compute() is called
era5_loaded = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/Elbe/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_*_*.nc', combine='by_coords')
glofas_loaded = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')


#DATA PREPROCESSING

#Doing this for debugging purposes
glofas = glofas_loaded.copy()
era5 = era5_loaded.copy()

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



#River selection (Optional) --> ***NOT IMPLEMENTED YET IN THE MODEL PIPELINE***
elbe_river_mask = get_river_mask(glofas['dis24'].mean('time'))
glofas_river = glofas
glofas_river = glofas_river.where(elbe_river_mask, drop=True)
glofas_river.isel(time=0).plot()
#plt.imshow(elbe_river_mask.astype(int))
plt.title('Elbe River')
plt.savefig('./images/Elbe/Elbe_river', dpi=600)

#Visualizing the region in 1999 after selecting the pertinent area
glofas['dis24'].isel(time=1).plot()
plt.savefig('./images/Elbe/1999_glofas_Elbe_basin', dpi=600)


era5['lsp'].isel(time=1).plot()
plt.savefig('./images/Elbe/era5_Elbe_basin', dpi=600)

#Visualizing the region from 1999-2019
dis_mean = glofas['dis24'].mean('time')
dis_mean.plot()
plt.title('Mean discharge in Elbe from 1999-2019')
plt.savefig('./images/Elbe/average_discharge_map', dpi=600)

# Taking the average latitude and longitude if necessary
era5 = era5.mean(['latitude', 'longitude'])
glofas = glofas.mean(['latitude', 'longitude'])

era5['lsp-4-11'] = shift_and_aggregate(era5['lsp'], shift=4, aggregate=8)
era5['lsp-12-25'] = shift_and_aggregate(era5['lsp'], shift=12, aggregate=14)
era5['lsp-26-55'] = shift_and_aggregate(era5['lsp'], shift=26, aggregate=30)
era5['lsp-56-180'] = shift_and_aggregate(era5['lsp'], shift=56, aggregate=125)

# Visualizing the features
# Converting to a dataarray
era5visualization = era5.to_array(dim='features').T
glofasvisualization = glofas.to_array(dim='features').T

import matplotlib.pyplot as plt

for f in era5visualization.features:
    plt.figure(figsize=(15, 5))
    era5visualization.sel(features=f).plot(ax=plt.gca())
    plt.savefig('./images/Elbe/' + str(f) + 'era5' + '.png', dpi=600, bbox_inches='tight')

for f in glofasvisualization.features:
    plt.figure(figsize=(15, 5))
    glofasvisualization.sel(features=f).plot(ax=plt.gca())
    plt.savefig('./images/Elbe/waterdischarge.png', dpi=600, bbox_inches='tight')


# Visualizing the distribution of discharge
import seaborn as sns

sns.distplot(glofas['dis24'])
plt.ylabel('density')
plt.xlim([0, 150])
plt.title('distribution of discharge')
plt.imshow()
plt.savefig('./images/Elbe/distribution_of_dis.png', dpi=600, bbox_inches='tight')
plt.close()
#For a Specific time period in Glofas
#glofas['dis24'].sel(time=slice('2013-5', '2013-6')).plot()


y_orig = glofas['dis24']
#Making a copy because y will be transformed to represent the variation of discharge. The model will be predicting the variation of discharge, not the quantity of discharge itself
y = y_orig.copy()
y = y.diff('time', 1)

X = era5

from functions.floodmodel_utils import reshape_scalar_predictand
#merges both values in time, since GLOFAS is daily while era5 is hourly data
Xda, yda = reshape_scalar_predictand(X, y)


"""PROBLEM"""
# need to fix dimensionality problem, since GloFAS is a daily dataset, whereas ERA5 iss an hourly dataset. I would want to uppscale instead of downscale.
X.features

# Splitting the dataset into training, test, and validation
period_train = dict(time=slice('1999', '2005'))
period_valid = dict(time=slice('2006', '2011'))
period_test = dict(time=slice('2012', '2016'))

#Train-test split
X_train, y_train = Xda.loc[period_train], yda.loc[period_train]
X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]
X_test, y_test = Xda.loc[period_test], yda.loc[period_test]


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import keras
from keras.layers.core import Dropout
from keras.constraints import MinMaxNorm, nonneg


class DenseNN(object):
    def __init__(self, **kwargs):
        self.output_dim = 1
        self.xscaler = StandardScaler()
        self.yscaler = StandardScaler()

        model = keras.models.Sequential()
        self.cfg = kwargs
        hidden_nodes = self.cfg.get('hidden_nodes')

        model.add(keras.layers.Dense(hidden_nodes[0],
                                     activation='tanh'))
        model.add(keras.layers.BatchNormalization())
        model.add(Dropout(self.cfg.get('dropout', None)))

        for n in hidden_nodes[1:]:
            model.add(keras.layers.Dense(n, activation='tanh'))
            model.add(keras.layers.BatchNormalization())
            model.add(Dropout(self.cfg.get('dropout', None)))
        model.add(keras.layers.Dense(self.output_dim,
                                     activation='linear'))
        opt = keras.optimizers.Adam()

        model.compile(loss=self.cfg.get('loss'), optimizer=opt)
        self.model = model

        self.callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        min_delta=1e-2, patience=100, verbose=0, mode='auto',
                                                        baseline=None, restore_best_weights=True), ]

    def score_func(self, X, y):
        """Calculate the RMS error

        Parameters
        ----------
        xr.DataArrays
        """
        ypred = self.predict(X)
        err_pred = ypred - y

        # NaNs do not contribute to error
        err_pred = err_pred.where(~np.isnan(err_pred), 0.)
        return float(np.sqrt(xr.dot(err_pred, err_pred)))

    def predict(self, Xda, name=None):
        """Input and Output: xr.DataArray

        Parameters
        ----------
        Xda : xr.DataArray
            with coordinates (time,)
        """
        X = self.xscaler.transform(Xda.values)
        y = self.model.predict(X).squeeze()
        y = self.yscaler.inverse_transform(y)

        y = add_time(y, Xda.time, name=name)
        return y

    def fit(self, X_train, y_train, X_valid, y_valid, **kwargs):
        """
        Input: xr.DataArray
        Output: None
        """

        print(X_train.shape)
        X_train = self.xscaler.fit_transform(X_train.values)
        y_train = self.yscaler.fit_transform(
            y_train.values.reshape(-1, self.output_dim))

        X_valid = self.xscaler.transform(X_valid.values)
        y_valid = self.yscaler.transform(
            y_valid.values.reshape(-1, self.output_dim))

        return self.model.fit(X_train, y_train,
                              validation_data=(X_valid, y_valid),
                              epochs=self.cfg.get('epochs', 1000),
                              batch_size=self.cfg.get('batch_size'),
                              callbacks=self.callbacks,
                              verbose=0, **kwargs)


config = dict(hidden_nodes=(64,),
              dropout=0.25,
              epochs=300,
              batch_size=50,
              loss='mse')

m = DenseNN(**config)

hist = m.fit(X_train, y_train, X_valid, y_valid)

# Summary of Model
m.model.summary()

m.model.save('../models/elbemodel.h5')
# save model
from keras.utils import plot_model

# plot Graph of Network

h = hist.model.history

# Plot training & validation loss value
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(h.history['loss'], label='loss')
ax.plot(h.history['val_loss'], label='val_loss')
plt.title('Learning curve')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
plt.legend(['Training', 'Validation'])
ax.set_yscale('log')
plt.savefig('../images/Elbe/ElbeNNlearningcurve.png', dpi=600, bbox_inches='tight')

yaml_string = m.model.to_yaml()
import yaml

with open('../models/keras-config.yml', 'w') as f:
    yaml.dump(yaml_string, f)

with open('../models/model-config.yml', 'w') as f:
    yaml.dump(config, f, indent=4)

from contextlib import redirect_stdout

with open('./models/ElbeNNsummary.txt', "w") as f:
    with redirect_stdout(f):
        m.model.summary()

from keras.utils import plot_model

plot_model(m.model, to_file='./images/Elbe/ElbeNNmodel.png', show_shapes=True)

from functions.plot import plot_multif_prediction

from functions.utils_floodmodel import generate_prediction_array

y_pred_train = m.predict(X_train)
y_pred_train = generate_prediction_array(y_pred_train, y_train, forecast_range=14)

plt.plot(y_train, y_pred_train)

y_pred_valid = m.predict(X_valid)
y_pred_valid = generate_prediction_array(y_pred_valid, y_valid, forecast_range=14)

y_pred_test = m.predict(X_test)
y_pred_test = generate_prediction_array(y_pred_test, y_test, forecast_range=14)

from functions.plot import plot_multif_prediction

title = 'Setting: Time-Delay Neural Net: 64 hidden nodes, dropout 0.25'
plot_multif_prediction(y_pred_test, y_test, forecast_range=14, title=title)





#Test case of Elbe basin during end of May-Beginning of June 2013 (flood happened in June 4th) to verify how well my model can predict flooding
import pandas as pd
time_range = pd.date_range('2013-05-15', periods=30)
y_case = y.loc[time_range]

#Plotting the real time plot
fig, ax = plt.subplots(figsize=(15, 5))
plt.title('Case study Elbe Basin May/June 2013')
ax.set_ylabel('river discharge [m$^3$/s]')
y_case.to_pandas().plot(ax=ax, label='reanalysis', lw=4)

legendlabels = ['reanalysis', 'neural net']
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='firebrick', lw=2)]

ax.legend(custom_lines, legendlabels, fontsize=11)

