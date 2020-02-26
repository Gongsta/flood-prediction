#This file was used to download the Ensemble Prediction Systems (EPS) provided by Glofas
#to compare the performance of the models

from ftplib import FTP
ftp = FTP('ftp.ecmwf.int')
ftp.login("safer", "neo2008")
ftp.cwd("for_StevenGong")
ftp.dir()

import os
import tarfile

#Download Europe Files on cloud

# local_filename = os.path.join("/tmp/flood_prediction/pipeline/", "EPS.tar")
# with open(local_filename, 'wb') as fp:
#      ftp.retrbinary('RETR glofas2.1_areagrid_for_StevenGongEurope20200207_201306.tar', fp.write)

for i in range(2016,2020):
     for j in range(1,13):
          if j < 10:
               j = '0' + str(j)

          else:
               j = str(j)

          local_filename = os.path.join("/Volumes/portableHardDisk/data/", ("EuropeEPS" + str(i) + j +".tar"))
          with open(local_filename, 'wb') as fp:
               ftp.retrbinary(('RETR glofas2.1_areagrid_for_StevenGongEurope20200207_' +str(i) + j +'.tar'), fp.write)

          # my_tar = tarfile.open(("europeEPS" + str(i) + str(j) +".tar"))

          # my_tar.extractall("/Volumes/portableHardDisk/data/EPS")

#
#
# import xarray as xr
#
#
# test_file = xr.open_dataset("/tmp/flood_prediction/pipeline/glofas2.1_2018_areagrid_for_StevenGong_in_Europe_2013062800.nc")