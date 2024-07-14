import os
import argparse
import concurrent.futures
import cdsapi
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

def download_era5_data(year, month, days, output_dir, api_key):
    filename = f"{year}-{month:02d}.nc"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return filepath

    # Use CDS API to download data
    c = cdsapi.Client(url = 'https://cds.climate.copernicus.eu/api/v2', key = api_key)
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['u_component_of_wind', 'v_component_of_wind', 'specific_humidity'],
            'pressure_level': [
                '1000', '975', '950', '925', '900', '875', '850', '825',
                '800', '775', '750', '700', '650', '600', '550', '500',
                '450', '400', '350', '300',
            ],
            'year': str(year),
            'month': f"{month:02d}",
            'day': days,
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'format': 'netcdf',
        },
        filepath)

    return filepath

def process_row(row, output_dir, api_key):
    year, month = row['year'], row['month']
    string = row['days'].strip("[]")
    string_list = string.split()
    days = [int(num) for num in string_list]
    try:
        filepath = download_era5_data(year, month, days, output_dir, api_key)
        print(f"Successfully downloaded and processed data for {year}-{month:02d}-{day:02d}")
    except Exception as e:
        print(f"Failed to process data for {year}-{month:02d}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download ERA5 data and calculate IVT')
    parser.add_argument('metafilename', type=str, help='CSV file containing the data to process')
    parser.add_argument('max_workers', type=int, help='Maximum number of workers for parallel processing')
    parser.add_argument('api_key', type=str, help='API Key')

    args = parser.parse_args()

    # Load the CSV file
    df = pd.read_csv(os.path.join('cdsapi_requests',args.metafilename))

    output_dir = 'era5'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use concurrent.futures for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_row, row, output_dir, args.api_key) for _, row in df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            future.result()

if __name__ == "__main__":
    main()
