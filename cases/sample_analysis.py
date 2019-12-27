import sys
sys.path.append('/Users/stevengong/Desktop/flood-prediction')
from functions.floodmodel_utils import get_basin_mask, shift_and_aggregate, generate_prediction_array, reshape_scalar_predictand
import xarray as xr

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

#try removing this and see if it yields the same results. No removing it will cause the plot at ./images/test/failed_model.png
y = y.diff('time', 1)

Xda, yda = reshape_scalar_predictand(X, y)

period_train = dict(time=slice(None, '2005'))
period_valid = dict(time=slice('2006', '2011'))
period_test = dict(time=slice('2012', '2016'))

X_train, y_train = Xda.loc[period_train], yda.loc[period_train]
X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]
X_test, y_test = Xda.loc[period_test], yda.loc[period_test]



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import keras
from keras.layers.core import Dropout
from keras.constraints import MinMaxNorm, nonneg

from functions.floodmodel_utils import add_time

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
              batch_size=90,
              loss='mse')

m = DenseNN(**config)
hist = m.fit(X_train, y_train, X_valid, y_valid)


import matplotlib.pyplot as plt
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


#serialize model to yaml file
model_yaml = m.model.to_yaml()
with open('./models/test/testmodel.yml', 'w') as yaml_file:
    yaml_file.write(model_yaml)

#Serializing weights to h5 file
m.model.save_weights('./models/test/weights.h5')


#LATER ON... LOADING THE WEIGHTS
yaml_model = open('./models/test/testmodel.yml', 'r').read()
from keras.models import model_from_yaml
loaded_model = model_from_yaml(yaml_model)
loaded_model.load_weights('./models/test/weights.h5')

m.model = loaded_model

y_pred_train = m.predict(X_train)
#You may need to rewrite this function, you want the prediction to be based on the previous day's discharge
y_pred_train = generate_prediction_array(y_pred_train, y_orig, forecast_range=14)


y_pred_valid = m.predict(X_valid)
y_pred_valid = generate_prediction_array(y_pred_valid, y_orig, forecast_range=30)

y_pred_test = m.predict(X_test)
y_pred_test = generate_prediction_array(y_pred_test, y_orig, forecast_range=30)

period_case = dict(time=slice('2015-10-12', '2015-10-13'))
X_case, y_case = Xda.loc[period_case], yda.loc[period_case]

y_pred_case = m.predict(X_case)
y_pred_case = generate_prediction_array(y_pred_case, y_orig, forecast_range=2)


from functions.plot import plot_multif_prediction
title='Setting: Time-Delay Neural Net: 64 hidden nodes, dropout 0.25'
plot_multif_prediction(y_pred_test, y_orig, forecast_range=30, title=title)
plt.savefig('./images/test/multif_prediction.png', dpi=600)




def multi_forecast_case_study_tdnn(pipe_case):
    """
    Convenience function for predicting discharge via the pre-trained input pipe.
    Loads glofas forecast_rerun data from a in-function set path, used to evaluate
    the model predictions.
    Outputs are 3 xr.DataArrays: One for the model forecast, one for the forecast reruns,
                                 one for the truth/reanalysis.

    Parameters
    ----------
    pipe_case : trainer ML pipe ready for prediction

    Returns
    -------
    xr.DataArray (3 times)
    """

    features_2013 = xr.open_dataset('../data/features_xy.nc')
    y_orig = features_2013['dis']
    y = y_orig.copy()
    X = features_2013.drop(['dis', 'dis_diff'])

    #Try removing this code and see if it still works?
    y = y.diff('time', 1)

    Xda, yda = reshape_scalar_predictand(X, y)


    multif_list = []
    multifrerun_list = []
    for forecast in range(1, 5):
        if forecast == 1:
            date_init = '2013-05-18'
            date_end = '2013-06-17'
            fr_dir = '2013051800'
        elif forecast == 2:
            date_init = '2013-05-22'
            date_end = '2013-06-21'
            fr_dir = '2013052200'
        elif forecast == 3:
            date_init = '2013-05-25'
            date_end = '2013-06-24'
            fr_dir = '2013052500'
        elif forecast == 4:
            date_init = '2013-05-29'
            date_end = '2013-06-28'
            fr_dir = '2013052900'

        X_case = Xda.sel(time=slice(date_init, date_end)).copy()

        # prediction start from every nth day
        # if in doubt, leave n = 1 !!!
        n = 1
        X_pred = X_case[::n].copy()
        y_pred = pipe_case.predict(X_pred)

        multif_case = generate_prediction_array(y_pred, y_orig, forecast_range=30)
        multif_case.num_of_forecast.values = [forecast]
        multif_list.append(multif_case)

        # add glofas forecast rerun data
        # glofas forecast rerun data
        # frerun = xr.open_dataset('/Users/stevengong/Desktop/data/glofas_freruns_case_study.nc')
        # fr = frerun['dis'].sel(lon=slice(13.9, 14.), lat=slice(48.4, 48.3))
        # fr = fr.drop(labels=['lat', 'lon']).squeeze()
        # multifrerun_list.append(fr)

    # merge forecasts into one big array
    date_init = '2013-05-18'
    date_end = '2013-06-28'
    y_case_fin = yda.sel(time=slice(date_init, date_end)).copy()
    X_case_multi_core = Xda.sel(time=slice(date_init, date_end)
                              ).isel(features=1).copy().drop('features')*np.nan

    X_list = []
    for fc in multif_list:
        X_iter = X_case_multi_core.copy()
        X_iter.loc[{'time': fc.time.values.ravel()}] = fc.values[0]
        X_list.append(X_iter)
    X_multif_fin = xr.concat(X_list, dim='num_of_forecast')
    X_multif_fin.name = 'prediction'

    X_list = []
    # for frr in multifrerun_list:
    #     X_iter = X_case_multi_core.copy()
    #     ens_list = []
    #     for fr_num in frr.ensemble:
    #         fr_iter = frr.sel(ensemble=fr_num)
    #         X_ens_iter = X_iter.copy()
    #         X_ens_iter.loc[{'time': frr.time.values}] = fr_iter.values
    #         ens_list.append(X_ens_iter)
    #     ens_da = xr.concat(ens_list, dim='ensemble')
    #     X_list.append(ens_da)
    # X_multifr_fin = xr.concat(X_list, dim='num_of_forecast')
    # X_multifr_fin.name = 'forecast rerun'
    return X_multif_fin, y_case_fin



#from functions.utils_floodmodel import multi_forecast_case_study_tdnn

X_multif_fin, y_case_fin = multi_forecast_case_study_tdnn(m)


fig, ax = plt.subplots(figsize=(15, 5))
color_scheme = ['g', 'cyan', 'magenta', 'k']

y_orig.sel({'time': X_multif_fin.time.values.ravel()}
            ).to_pandas().plot(ax=ax, label='GloFAS Reanalysis')

#y_case_fin.to_pandas().plot(ax=ax, label='reanalysis', lw=4)
run = 0
for i in X_multif_fin.num_of_forecast:
    X_multif_fin.sel(num_of_forecast=i).to_pandas().T.plot(ax=ax,
                                                           label='forecast',
                                                           linewidth=2,
                                                           color='firebrick')
    # X_multifr_fin.sel(num_of_forecast=i).to_pandas().T.plot(ax=ax,
    #                                                         label='frerun',
    #                                                         linewidth=0.9,
    #                                                         linestyle='--',
    #                                                         color=color_scheme[run])
    run += 1
ax.set_ylabel('river discharge [m$^3$/s]')

from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='firebrick', lw=2),
                Line2D([0], [0], color='g', linestyle='--'),
                Line2D([0], [0], color='cyan', linestyle='--'),
                Line2D([0], [0], color='magenta', linestyle='--'),
                Line2D([0], [0], color='k', linestyle='--')]

legendlabels = ['reanalysis', 'neural net', 'EFAS 05-18', 'EFAS 05-22', 'EFAS 05-25', 'EFAS 05-29']
ax.legend(custom_lines, legendlabels, fontsize=11)
plt.title('Setting: Time-Delay Neural Net: 64 hidden nodes, dropout 0.25')
plt.savefig('./images/forecast.png', dpi=600)

