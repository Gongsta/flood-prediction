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
year_end = 2010
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
            '2010'
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
'''
glofas = xr.open_dataset('../data/dataset-cems-glofas-historical-fc9c62e9-1df3-4179-84dd-277f77e620fb/CEMS_ECMWF_dis24_20010101_glofas_v2.1.nc')

#Putting all of glofas files into a single file
for i in range(2,32):
    if i<10:
        i = '0' + str(i)

    glofas = glofas.merge(xr.open_dataset('../data/dataset-cems-glofas-historical-fc9c62e9-1df3-4179-84dd-277f77e620fb/CEMS_ECMWF_dis24_200101'+ str(i) + '_glofas_v2.1.nc'))

#Saving the 31 glofas to a single .nc file, stored in exterior folder not committed to github (the file is 670mb per month)
glofas.to_netcdf(path="../data/glofas2001.nc")

#Putting all of era5 into a single file
era5 = xr.open_dataset('../data/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_2005_01.nc')

for n in range(5,11):
    if n<10:
        n = '0'+ str(n)

    for i in range(2,13):
        if i<10:
            i = '0' + str(i)

         era5= era5.merge(xr.open_dataset('../data/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_20' + str(n) +'_' + str(i) + '.nc'))


era5.to_netcdf(path="../data/2005-2010-era5.nc")

glofas = xr.open_dataset("../data/glofas2001.nc")
'''

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





from functions.utils_floodmodel import get_mask_of_basin, add_shifted_variables, reshape_scalar_predictand

'''
#Get mask of basin Return a mask where all points outside the selected basin are False.
danube_catchment = get_mask_of_basin(glofas['dis24'].isel(time=0))
dis = glofas['dis'].where(danube_catchment)
'''

era5test = era5.isel(longitude=[-10,-11,-13], latitude=[0])
glofas = glofas.isel(longitude=[-20], latitude=[0])
#Taking the average latitude and longitude
era5 = era5.mean(['longitude','latitude'])
glofas = glofas.mean(['lon','lat'])

#Visualizing the features
#Converting to a dataarray
era5visualization = era5.to_array(dim='features').T
glofasvisualization = glofas.to_array(dim='features').T

import matplotlib.pyplot as plt
for f in era5visualization.features:
    plt.figure(figsize=(15,5))
    era5visualization.sel(features=f).plot(ax=plt.gca())
    #plt.savefig('era5visualization.png', dpi=600, bbox_inches='tight')


for f in glofasvisualization.features:
    plt.figure(figsize=(15,5))
    glofasvisualization.sel(features=f).plot(ax=plt.gca())

    #plt.savefig('glofasvisualization.png', dpi=600, bbox_inches='tight')

#sample_data = xr.merge([glofas, era5])



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

yPertinent = y
XPertinent = X.isel()
from functions.utils_floodmodel import reshape_scalar_predictand
X, y = reshape_scalar_predictand(X, y)
X.features

#Splitting the dataset into training, test, and validation

period_train = dict(time=slice(None, '2005'))
period_valid = dict(time=slice('2006', '2009'))
period_test = dict(time=slice('2009', '2010'))

X_train, y_train = X.loc[period_train], y.loc[period_train]
X_valid, y_valid = X.loc[period_valid], y.loc[period_valid]
X_test, y_test = X.loc[period_test], y.loc[period_test]


#Visualizing the distribution of discharge
import seaborn as sns
sns.distplot(y)
plt.ylabel('density')
plt.xlim([-100, 400])
plt.title('distribution of discharge')
plt.plot()
#lt.savefig('distribution_dis.png', dpi=600, bbox_inches='tight')


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
                  batch_size=90,
                  loss='mse')



m = DenseNN(**config)


config = dict(hidden_nodes=(64,),
                  dropout=0.25,
                  epochs=300,
                  batch_size=90,
                  loss='mse')

hist = m.fit(X_train, y_train, X_valid, y_valid)

#Summary of Model
m.model.summary()

m.save('./')
#save model


"""
#plot Graph of Network
from keras.utils import plot_model
plot_model(m.model, to_file='model.png', show_shapes=True)

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

"""

from functions.plot import plot_multif_prediction

title='Setting: Time-Delay Neural Net: 64 hidden nodes, dropout 0.25'
plot_multif_prediction(y_pred_test, y_orig, forecast_range=14, title=title);

