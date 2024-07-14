import xarray as xr
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import GOES
import pyproj as pyproj
from pyresample import utils
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import resample_nearest
import concurrent.futures

import datetime

from concurrent.futures import ProcessPoolExecutor
import argparse

def goes_reproject(filename, domain, var):
    ds = GOES.open_dataset(filename)
    CMI, LonCen, LatCen = ds.image(var, lonlat='center', domain=domain)
    sat = ds.attribute('platform_ID')
    band = ds.variable('band_id').data[0]
    wl = ds.variable('band_wavelength').data[0]
    standard_name = CMI.standard_name
    units = CMI.units
    time_bounds = CMI.time_bounds
    LonCenCyl, LatCenCyl = GOES.create_gridmap(domain, PixResol=2.0)
    LonCorCyl, LatCorCyl = GOES.calculate_corners(LonCenCyl, LatCenCyl)
    Prj = pyproj.Proj('+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=6378.137 +b=6378.137 +units=km')
    AreaID = 'cyl'
    AreaName = 'cyl'
    ProjID = 'cyl'
    Proj4Args = '+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=6378.137 +b=6378.137 +units=km'
    
    ny, nx = LonCenCyl.data.shape
    SW = Prj(LonCenCyl.data.min(), LatCenCyl.data.min())
    NE = Prj(LonCenCyl.data.max(), LatCenCyl.data.max())
    area_extent = [SW[0], SW[1], NE[0], NE[1]]
    AreaDef = utils.get_area_def(AreaID, AreaName, ProjID, Proj4Args, nx, ny, area_extent)
    
    SwathDef = SwathDefinition(lons=LonCen.data, lats=LatCen.data)
    CMICyl = resample_nearest(SwathDef, CMI.data, AreaDef, radius_of_influence=6000,
                            fill_value=np.nan, epsilon=3, reduce_data=True)
    del CMI, LonCen, LatCen, SwathDef, LonCenCyl, LatCenCyl
    
    return LatCorCyl.data, LonCorCyl.data, CMICyl.data

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def process_file(filepath):
    t1 = datetime.datetime.now()
    print(f'Processing {filepath} ...')
    LatCorCyl, LonCorCyl, CMICyl = goes_reproject(filepath, domain, 'Rad')
    print(f'Processing {filepath} took {datetime.datetime.now() - t1} sec')
    return LatCorCyl, LonCorCyl, CMICyl

def process_row(year, day, hour, nc_df, domain):
    print('Starting... ',year, day, hour)
    ds_src = xr.open_dataset(nc_df.iloc[0]['filepath'])
    flag = 1

    for idx, row in nc_df.iterrows():
        filepath = row['filepath']
        band = row['band']

        if flag == 1:
            lat, lon, data = process_file(filepath)
            lat_1d = np.unique(lat)
            lat_1d = lat_1d[1:]
            lon_1d = np.unique(lon)
            lon_1d = lon_1d[1:]
            ds = xr.Dataset({
                band: (('lat', 'lon'), data)
            },
            coords={
                'lat': lat_1d,
                'lon': lon_1d
            })
            ds.attrs = ds_src.attrs.copy()
            del ds_src
            flag = 0
        else:
            lat, lon, data = process_file(filepath)
            ds[band] = (('lat', 'lon'), data)

    dirpath = os.path.join('goes2go', str(year))
    create_dir_if_not_exists(dirpath)

    filename = f"{day}_{hour}_g17.nc"
    ds.to_netcdf(os.path.join(dirpath, filename))
    del ds

def parallel_process(df, band_counts, domain, max_workers):
    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in band_counts.iterrows():
            year = row['year']
            day = row['day']
            hour = row['hour']

            print(year, day, hour)

            nc_df = df[(df['year'] == year) & (df['day'] == day) & (df['hour'] == hour)]

            print('event dataframe created')

            tasks.append(executor.submit(process_row, year, day, hour, nc_df, domain))

        # Ensure all tasks are completed
        for task in tasks:
            task.result()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--max_workers', type=int, default=4, help='maximum number of workers for parallel processing')
    
    args = parser.parse_args()
    max_workers = args.max_workers

    df = pd.read_csv('goes2go.csv')
    print('dataframe read!!')
    band_counts = df.groupby(['year', 'day', 'hour']).size().reset_index(name='band_count')
    band_counts = band_counts[band_counts['band_count']==16]
    print('band count dataframe read!!')

    domain = [-180, -90, 0, 80]
    parallel_process(df, band_counts, domain, max_workers)