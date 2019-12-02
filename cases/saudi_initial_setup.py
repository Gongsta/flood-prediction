"""Code to run for people trying to replicate this"""

#Data extraction


#Saving our API key so that we can use Climate Data Store API to extract data
UID = '26343'
API_key = 'b61b1b28-a04b-4cb3-acb1-b48775702011'

import os
with open(os.path.join(os.path.expanduser('~'), '.cdsapirc2'), 'w') as f:
    f.write('url: https://cds.climate.copernicus.eu/api/v2\n')
    f.write(f'key: {UID}:{API_key}')


#Adding functions to our directory so we can use other functions
import sys
sys.path.append('./')
from functions.data_download import CDS_Dataset


#Downloading ERA5 Dataset

ds = CDS_Dataset(dataset_name='reanalysis-era5-single-levels',
                 save_to_folder='../data/'  # path to where datasets shall be stored
                )

# define areas of interest (N/W/S/E)
saudi_arabia = [28,36,17,50]

# define time frame of the downlaod
year_start = 2005
year_end = 2016
month_start = 1
month_end = 12

# define request variables
request = dict(product_type='reanalysis',
               format='netcdf',
               area=saudi_arabia,
               variable=['convective_precipitation','land_sea_mask','large_scale_precipitation',
            'runoff','slope_of_sub_gridscale_orography','soil_type',
            'total_column_water_vapour','volumetric_soil_water_layer_1','volumetric_soil_water_layer_2'],
               )

#Sending the request
ds.get(years = [str(y) for y in range(year_start, year_end+1)],
       months = [str(a).zfill(2) for a in range(month_start, month_end+1)],
       request = request,
       N_parallel_requests = 12)



#Code for downloading Glofas Data
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'cems-glofas-historical',
    {
        'format':'zip',
        'year':[
            '2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016'
        ],
        'variable':'River discharge',
        'month':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12'
        ],
        'day':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12',
            '13','14','15',
            '16','17','18',
            '19','20','21',
            '22','23','24',
            '25','26','27',
            '28','29','30',
            '31'
        ],
        'dataset':'Consolidated reanalysis',
        'version':'2.1'
    },
    'download.zip')



import xarray as xr


#Data Preprocessing

import xarray as xr
#The open_mfdataset function automatically combines the many .nc files, the * represents the value that varies
era5 = xr.open_mfdataset('../data/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_*_*.nc', combine='by_coords')

glofas = xr.open_mfdataset('../data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')


#To read a single shape by calling its index use the shape() method. The index is the shape's count from 0.
# So to read the 8th shape record you would use its index which is 7.
import shapefile

sf = shapefile.Reader("../data/Saudi_bassins/saudi_arabia_bassins.shp")

shapes = sf.shapes()

"""
#View the characteristics of the shape file
for name in dir(shapes[3]):
    if not name.startswith('_'):
        print(name)

"""
#The shapefile we are using has 5 attributes:
#'bbox'
#'parts'
#'points'
#'shapeType'
#'shapeTypeName'
#Read the following documentation to learn more

#Creating an array of all the basins in the world
basins = []
for n in range(len(shapes)):
    basins.append(shapes[n].bbox)


latitudeList = []
longitudeList = []
for point in basins:
    latitudeList.append(point[0])
    longitudeList.append(point[1])


from functions.utils_floodmodel import get_mask_of_basin, add_shifted_variables, reshape_scalar_predictand

'''
#Get mask of basin Return a mask where all points outside the selected basin are False.
danube_catchment = get_mask_of_basin(glofas['dis'].isel(time=0))
dis = glofas['dis'].where(danube_catchment)
'''

era5 = era5.sel(latitude=['19,75','20','20.25','20.5'], longitude=['45.75','46'])


"""Problem where I have to hand manually type this with the many integers"""
#This doesnt work:
#glofas = glofas.sel(lat='89.95', lon='45.75')
glofas = glofas.sel(lat=['20.25','19.75'], lon=['45.75000000000003','46.05000000000001'])

#Taking the average latitude and longitude if necessary
era5 = era5.mean(['latitude','longitude'])
glofas = glofas.mean(['lat','lon'])

#Visualizing the features
#Converting to a dataarray
era5visualization = era5.to_array(dim='features').T
glofasvisualization = glofas.to_array(dim='features').T

import matplotlib.pyplot as plt
for f in era5visualization.features:
    plt.figure(figsize=(15,5))
    era5visualization.sel(features=f).plot(ax=plt.gca())
    plt.savefig('./images/'+str(f)+ 'era5'+'.png', dpi=600, bbox_inches='tight')


for f in glofasvisualization.features:
    plt.figure(figsize=(15,5))
    glofasvisualization.sel(features=f).plot(ax=plt.gca())
    plt.savefig('./images/glofasvisualization'+str(f)+'.png', dpi=600, bbox_inches='tight')


#Creating the model
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dask
from dask.distributed import Client, LocalCluster
cluster = LocalCluster(processes=True) #n_workers=10, threads_per_worker=1,
client = Client(cluster)  # memory_limit='16GB',

import xarray as xr
from dask.diagnostics import ProgressBar

y = glofas['dis24']
X = era5

from functions.utils_floodmodel import reshape_scalar_predictand
X, y = reshape_scalar_predictand(X, y)

"""PROBLEM"""
#need to fix dimensionality problem, since GloFAS is a daily dataset, whereas ERA5 iss an hourly dataset. I would want to uppscale instead of downscale.
X.features

#Splitting the dataset into training, test, and validation

period_train = dict(time=slice('2005', '2008'))
period_valid = dict(time=slice('2008', '2009'))
period_test = dict(time=slice('2009', '2010'))

X_train, y_train = X.loc[period_train], y.loc[period_train]
X_valid, y_valid = X.loc[period_valid], y.loc[period_valid]
X_test, y_test = X.loc[period_test], y.loc[period_test]


""""""
#Visualizing the distribution of discharge
import seaborn as sns
sns.distplot(y)
plt.ylabel('density')
plt.xlim([0, 0.001])
plt.title('distribution of discharge')
plt.plot()
#plt.savefig('distribution_dis.png', dpi=600, bbox_inches='tight')


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import keras
from keras.layers.core import Dropout
from keras.constraints import MinMaxNorm, nonneg


def add_time(vector, time, name=None):
    """Converts numpy arrays to xarrays with a time coordinate.

    Parameters
    ----------
    vector : np.array
        1-dimensional array of predictions
    time : xr.DataArray
        the return value of `Xda.time`

    Returns
    -------
    xr.DataArray
    """
    return xr.DataArray(vector, dims=('time'), coords={'time': time}, name=name)


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

#Summary of Model
m.model.summary()

m.model.save('modeltest.h5')
#save model
from keras.utils import plot_model

"""
#plot Graph of Network
from keras.utils import plot_model
plot_model(m.model, to_file='./images/model.png', show_shapes=True)


h = hist.model.history

# Plot training & validation loss value
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(h.history['loss'], label='loss')
ax.plot(h.history['val_loss'], label='val_loss')
plt.title('Learning curve')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
plt.legend(['Training', 'Validation'])
ax.set_yscale('log')
plt.savefig('./images/learningcurve.png', dpi=600, bbox_inches='tight')

"""
yaml_string = m.model.to_yaml()


with open('./models/keras-config.yml', 'w') as f:
    yaml.dump(yaml_string, f)

with open('./models/model-config.yml', 'w') as f:
    yaml.dump(config, f, indent=4)

from contextlib import redirect_stdout
with open('./models/summary.txt', "w") as f:
    with redirect_stdout(f):
        m.model.summary()

from functions.plot import plot_multif_prediction

from functions.utils_floodmodel import generate_prediction_array
y_pred_train = m.predict(X_train)
y_pred_train = generate_prediction_array(y_pred_train, y_train, forecast_range=14)

plt.plot(y_train, y_pred_train)


y_pred_valid = m.predict(X_valid)
y_pred_valid = generate_prediction_array(y_pred_valid, y_orig, forecast_range=14)

y_pred_test = m.predict(X_test)
y_pred_test = generate_prediction_array(y_pred_test, y_test, forecast_range=14)

from functions.plot import plot_multif_prediction
title='Setting: Time-Delay Neural Net: 64 hidden nodes, dropout 0.25'
plot_multif_prediction(y_pred_test, y_test, forecast_range=14, title=title)

#Update shapefile with Shapefile in order to include the predictions for each individual basin
