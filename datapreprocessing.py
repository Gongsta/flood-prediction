import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

import sys
sys.path.append('./')

from functions import get_mask_of_basin, add_shifted_variables, reshape_scalar_predictand