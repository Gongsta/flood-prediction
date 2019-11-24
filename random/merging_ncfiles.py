glofas = xr.open_dataset('../data/dataset-cems-glofas-historical-fc9c62e9-1df3-4179-84dd-277f77e620fb/CEMS_ECMWF_dis24_20010101_glofas_v2.1.nc')

#Putting all of glofas files into a single file
for i in range(2,32):
    if i<10:
        i = '0' + str(i)

    glofas = glofas.merge(xr.open_dataset('../data/dataset-cems-glofas-historical-fc9c62e9-1df3-4179-84dd-277f77e620fb/CEMS_ECMWF_dis24_200101'+ str(i) + '_glofas_v2.1.nc'))

#Saving the 31 glofas to a single .nc file, stored in exterior folder not committed to github (the file is 670mb per month)
glofas.to_netcdf(path="../data/glofas2001.nc")

#Putting all of era5 into a single file, not necessary. Use .open_mfdataset instead
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
