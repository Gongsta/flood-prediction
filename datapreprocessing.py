
import xarray as xr
#The open_mfdataset function automatically combines the many .nc files, the * represents the value that varies
era5 = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/reanalysis-era5-single-levels_convective_precipitation,land_sea_mask,large_scale_precipitation,runoff,slope_of_sub_gridscale_orography,soil_type,total_column_water_vapour,volumetric_soil_water_layer_1,volumetric_soil_water_layer_2_*_*.nc', combine='by_coords')

glofas = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')




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

