
import xarray as xr
#The open_mfdataset function automatically combines the many .nc files, the * represents the value that varies
era5 = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_*_*.nc', combine='by_coords')

glofas = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')


#To read a single shape by calling its index use the shape() method. The index is the shape's count from 0.
# So to read the 8th shape record you would use its index which is 7.
import shapefile


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
#Read the following documentation to learn more:  https://pypi.org/project/pyshp/

sf = shapefile.Reader("../basins/major_basins/Major_Basins_of_the_World.shp")

shapes = sf.shapes()

records = sf.records()

#Insert the name of the basin
basin = ""

danube_catchment = get_mask_of_basin(glofas['dis24'].isel(time=0), kw_basins='Elbe')
X = era5_features.where(danube_catchment).mean(['latitude', 'longitude'])



#function that returns the index of the basin based on its name
def get_basin_index(basin, records):

    for i in range(len(records)):
        if basin == records[i][3]:

            return i

index = get_basin_index(basin, records)

points = shapes[index].points

bbox = shapes[index].bbox

"""
#i cant make this work yet, where I would select the polygon instead of a general box
def return_area_basin(points, era5, glofas):

    glofas.where(all(glofas.lat < points[i][0]  for i in points) and all(glofas.lon > points[i][1]  for i in points), drop=True)


    return era5, glofas


era5, glofas = return_area_basin(points, era5, glofas)
"""
def createPointList(latMin, lonMin, latMax, lonMax, latList, lonList):

    #Lat is a list of all the available latitudes
    #Lon is a list of all the available longitudes

    lat = []
    lon = []

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]


    for i in latList:
        if i <= latMax and i >= latMin:
            lat.append(i)

    for i in lonList:
        if i <= lonMax and i >= lonMin:
            lon.append(i)


    if not lat:
        averageLat = (latMin + latMax)/2
        lat.append(find_nearest(latList, averageLat))


    if not lon:
        averageLon = (lonMin + lonMax)/2
        lon.append(find_nearest(lonList, averageLon))



    return lat, lon


lat, lon = createPointList(bbox[1], bbox[0], bbox[3], bbox[2], era5.latitude.values,era5.longitude.values)
lat, lon = createPointList(bbox[1], bbox[0], bbox[3], bbox[2], glofas.lat.values,glofas.lon.values)


era5 = era5.sel(latitude=lat, longitude=lon)
glofas = glofas.sel(lat=lat, lon=lon)


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

