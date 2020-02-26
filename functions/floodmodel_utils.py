"""Contains utilities related to the flood model (data processing,saving prediction, generating prediction, etc.) """

import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas
from rasterio import features
from affine import Affine

np.seterr(divide='ignore', invalid='ignore')


def get_river_mask(dis):
    """Returns a DataArray where all values that are of a discharge of 10 or under is false (hence all areas that are a river returns true).
    In other words, a mask of the river in the area.

        Parameters:
        -----------
        dis : xr.DataArray
            A GloFAS DataArray with a discharge dimension contains the 'latitude' and 'longitude' coordinates, 'time' coordinates are optional. I would suggest
            to remove the 'time' coordinates when plotting the river by taking the mean time.

        Returns
        -------
        newDis : xr.DataArray
            a new DataArray where all points that have a discharge of 10 or under is returned as false.

        See Also
        --------
        get_basin_mask : Function which returns the mask of a basin in the xr.DataArray form.

        Examples
        --------
        # Obtaining the Elbe river area
        >> glofas = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')
        >> elbe_river_mask = get_mask_river(glofas['dis24'].mean('time'))
        #Note that if glofas is downloaded as containing all the global coordinates, the function will return all rivers in the world
        >> glofas = glofas.where(elbe_river_mask, drop=True)

        """
    newDis = dis > 10

    return (newDis)


def get_basin_mask(da, basin_name):

    """Returns a mask where all points outside the selected basin are False.

    Parameters:
    -----------
    da : xr.DataArray
        contains the coordinates
    kw_basins : str
        Name of the basin in the basins dataset shapefile

    Returns
    -------
    da : xr.DataArray
        the transformed dataArray with all points outside of the basin identified as False.

    Examples
    --------
    # Obtaining the mask of the Elbe basin
    >> glofas = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')
    >> elbe_basin_mask = get_basin_mask(glofas['dis24'].isel(time=0), 'Elbe')
    >> elbe_basin_mask

    <xarray.DataArray 'basins' (latitude: 1500, longitude: 3600)>
array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]])
Coordinates:
  * longitude  (longitude) float64 -179.9 -179.8 -179.8 ... 179.8 179.9 180.0
  * latitude   (latitude) float64 89.95 89.85 89.75 ... -59.75 -59.85 -59.95
    time       datetime64[ns] 1999-01-01


    #Applying the mask of the basin to the GloFAS dataset by dropping all datasets outside of the basin
    >> glofas = glofas.where(elbe_basin_mask, drop=True)

    #Applying the mask of the basin to the Era5 dataset by dropping all datasets outside of the basin
    >> era5 = era5.interp(latitude=glofas.latitude, longitude=glofas.longitude).where(elbe_basin_mask, drop=True)



    """
    def transform_from_latlon(lat, lon):
        """
        Performing affine transformation, for more information look here: https://pypi.org/project/affine/
        :param lat: xr.DataArray of latitude points
        :param lon: xr.DataArray of longitude points
        :return:
        """
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        trans = Affine.translation(lon[0], lat[0])
        scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
        return trans * scale

    def rasterize(shapes, coords, fill=np.nan, **kwargs):
        """Rasterize a list of (geometry, fill_value) tuples onto the given
        xray coordinates. This only works for 1d latitude and longitude
        arrays.
        """
        transform = transform_from_latlon(coords['latitude'], coords['longitude'])
        out_shape = (len(coords['latitude']), len(coords['longitude']))
        #Performing rasterization using the Rasterio library
        raster = features.rasterize(shapes, out_shape=out_shape,
                                    fill=fill, transform=transform,
                                    dtype=float, **kwargs)
        return xr.DataArray(raster, coords=coords, dims=('latitude', 'longitude'))


    shp2 = "../../data/basins/major_basins/Major_Basins_of_the_World.shp"
    basins = geopandas.read_file(shp2)
    single_basin = basins.query("NAME == '"+ basin_name +"'").reset_index(drop=True)
    shapes = [(shape, n) for n, shape in enumerate(single_basin.geometry)]

    da['basins'] = rasterize(shapes, da.coords)
    da = da.basins == 0
    return da.drop('basins')  # the basins coordinate is not used anymore from here on


