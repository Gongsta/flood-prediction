from ftplib import FTP
ftp = FTP('ftp.ecmwf.int')
ftp.login("safer", "neo2008")
ftp.cwd("for_StevenGong")
ftp.dir()

import os
local_filename = os.path.join("/tmp/flood-prediction/pipeline/", "test")
with open(local_filename, 'wb') as fp:
     ftp.retrbinary('RETR glofas2.1_2018_areagrid_for_StevenGong_in_Europe_1998010100.nc', fp.write)


import xarray as xr


test_file = xr.open_dataset("/tmp/flood-prediction/pipeline/file")