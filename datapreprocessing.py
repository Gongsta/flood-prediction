
import xarray as xr
#The open_mfdataset function automatically combines the many .nc files, the * represents the value that varies
era5 = xr.open_mfdataset('../data/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_*_*.nc', combine='by_coords')

glofas = xr.open_mfdataset('../data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')


#To read a single shape by calling its index use the shape() method. The index is the shape's count from 0.
# So to read the 8th shape record you would use its index which is 7.
import shapefile

sf = shapefile.Reader("./basins/major_basins/Major_Basins_of_the_World.shp")

shapes = sf.shapes()

records = sf.records()

basin = "Elbe"

#function that returns the index of the basin based on its name
def get_basin_index(basin, records):

    for i in range(len(records)):
        if basin == records[i][3]:

            return i

index = get_basin_index(basin, records)

def return_area_basin(points, era5, glofas):


    return era5, glofas


era5, glofas = return_area_basin(shapes[index].points, era5, glofas)

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


#Creating an array of all the basins in Saudi Arabia
basins = []
for n in range(len(shapes)):
    basins.append(shapes[n].bbox)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def createPointList(latMin, lonMin, latMax, lonMax, latList, lonList):

    #Lat is a list of all the available latitudes
    #Lon is a list of all the available longitudes

    lat = []
    lon = []
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


lat, lon = createPointList(basins[20][1],basins[20][0], basins[20][3], basins[20][2],era5.latitude.values,era5.longitude.values)
lat, lon = createPointList(basins[20][1],basins[20][0], basins[20][3], basins[20][2],glofas.lat.values,glofas.lon.values)


era5 = era5.sel(latitude=lat, longitude=lon)


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

