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
X_train, y_train = Xda.loc[period_train], yda.loc[period_train]
X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]
X_test, y_test = Xda.loc[period_test], yda.loc[period_test]

from sklearn.preprocessing import StandardScaler
import keras
from keras.layers.core import Dropout


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
        #Calling the squeeze does what??
        y = self.model.predict(X).squeeze()
        #Since we applied feature scaling to y, we need to revert back to obtain the original value
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
              batch_size=90,
              loss='mse')

m = DenseNN(**config)
hist = m.fit(X_train, y_train, X_valid, y_valid)


# Summary of Model
m.model.summary()



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
plt.savefig('./images/Elbe/ElbeNNlearningcurve.png', dpi=600, bbox_inches='tight')

# serialize model to YAML
model_yaml = m.model.to_yaml()
with open("./models/elbe-model1.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
m.model.save_weights("./models/elbe-model1.h5")
#Seialize feature scaling weights



#LATER ON... LOADING THE WEIGHTS
yaml_model = open('./models/elbe-model1.yaml', 'r').read()
from keras.models import model_from_yaml
loaded_model = model_from_yaml(yaml_model)
loaded_model.load_weights('./models/elbe-model1.h5')
m.model = loaded_model
m.xscaler.fit_transform(X_train.values)
m.yscaler.fit_transform(y_train.values.reshape(-1, m.output_dim))

from contextlib import redirect_stdout

with open('./models/ElbeNNsummary.txt', "w") as f:
    with redirect_stdout(f):
        m.model.summary()

from keras.utils import plot_model

plot_model(m.model, to_file='./images/Elbe/ElbeNNmodel.png', show_shapes=True)

from functions.plot import plot_multif_prediction

from functions.floodmodel_utils import generate_prediction_array

y_pred_train = m.predict(X_train)
y_pred_train = generate_prediction_array(y_pred_train, y_orig, forecast_range=14)

y_pred_valid = m.predict(X_valid)
y_pred_valid = generate_prediction_array(y_pred_valid, y_orig, forecast_range=14)

y_pred_test = m.predict(X_test)
y_pred_test = generate_prediction_array(y_pred_test, y_orig, forecast_range=14)

from functions.plot import plot_multif_prediction
title='Setting: Time-Delay Neural Net: 64 hidden nodes, dropout 0.25'
plot_multif_prediction(y_pred_test, y_orig, forecast_range=14, title=title)
plt.savefig('./images/Elbe/multif_prediction.png', dpi=600)


#Test case of Elbe basin during end of May-Beginning of June 2013 (flood happened in June 4th) to verify how well my model can predict flooding
import pandas as pd
time_range = pd.date_range('2013-05-15', periods=30)
y_case = y_orig.loc[time_range]

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

