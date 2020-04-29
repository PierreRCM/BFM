import numpy as np
import xarray as xr
import pandas as pd
import os
import psutil
process = psutil.Process(os.getpid())
data = np.load("../data/D_WT_1000SS.p")
a = data.keys()
coords = ["cellnum", "strain", "stator", "FPS", "dbead_nm", "nm_per_pix"]
all_array = []

for exp in data.keys():

    df = pd.DataFrame(np.array([data[exp][x] for x in ["x", "y"]]).T)
    df.columns = ["x", "y"]
    datarray = df.to_xarray()
    datarray = datarray.to_array()
    datarray = datarray.expand_dims("experiment")
    datarray = datarray.assign_coords(strain=("experiment", [data[exp]["strain"]]))
    datarray = datarray.assign_coords(stator=("experiment", [data[exp]["stator"]]))
    datarray = datarray.assign_coords(FPS=("experiment", [data[exp]["FPS"]]))
    datarray = datarray.assign_coords(dbead_nm=("experiment", [data[exp]["dbead_nm"]]))
    datarray = datarray.assign_coords(nm_per_pix=("experiment", [data[exp]["nm_per_pix"]]))
    if type(data[exp]["cellnum"]) != list:
        datarray = datarray.assign_coords(cellnum=("experiment", [data[exp]["cellnum"]]))
    else:
        datarray = datarray.assign_coords(cellnum=("experiment", data[exp]["cellnum"]))

    datarray.to_netcdf("../data/exp_" + str(exp))








