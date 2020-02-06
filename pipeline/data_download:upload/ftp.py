from ftplib import FTP
ftp = FTP('ftp.ecmwf.int')
ftp.login("safer", "neo2008")
ftp.cwd("for_StevenGong")
ftp.dir()

import os
local_filename = os.path.join("/Users/stevengong/Desktop/flood-prediction/pipeline/data_download:upload/", "file")
with open(local_filename, 'wb') as fp:
     ftp.retrbinary('RETR glofas2.1_reforecast_ref2018_glofas2.1_era5wb_released_dis_points_noobs_2000120300_newobs_stat6122_197901_201712_20190125.nc', fp.write)


import xarray as xr


test_file = xr.open_dataset("./pipeline/data_download:upload/file")