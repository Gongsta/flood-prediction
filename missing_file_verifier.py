#Made this file because had some problems with files of different days not being there, so ran this code to check which days were missing
glofas = xr.open_mfdataset('/Volumes/Seagate Backup Plus Drive/data/*/CEMS_ECMWF_dis24_*_glofas_v2.1.nc', combine='by_coords')

missingTimes = {'year': [], 'month':[], 'day': []}
for year in range(2001, 2006):
    print(year)

    for month in range(1,13):
        print(month)
        if month < 10:
            month = '0' + str(month)

        for day in range(1,32):
            if day<10:
                day = '0' + str(day)

                try:
                    glofas = xr.open_dataset('/Volumes/Seagate Backup Plus Drive/data/'+ str(year) + '/CEMS_ECMWF_dis24_' +str(month) + str(day) + '_glofas_v2.1.nc')

                except:
                    missingTimes['year'].append(str(year))
                    missingTimes['month'].append(str(month))
                    missingTimes['day'].append(str(day))

