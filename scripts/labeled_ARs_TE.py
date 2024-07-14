import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

def list_nc_files_to_csv_and_npy(url, csv_file, npy_file, start_year=1996, end_year=2022, download_path='nc_files', max_workers=4):
    # Get the content of the URL
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all links ending with .nc
    nc_files = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.nc')]
    
    # Filter files based on year range and create a list of dictionaries
    file_list = []
    for nc_file in nc_files:
        # Extract the year from the file name
        year_match = re.search(r'(\d{4})', nc_file)
        if year_match:
            year = int(year_match.group(1))
            if start_year <= year <= end_year:
                file_url = urljoin(url, nc_file)
                file_list.append({'File Name': nc_file, 'File URL': file_url, 'Year': year})
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(file_list)
    
    # Save DataFrame to CSV file
    df.to_csv(csv_file, index=False)
    print(f"List of .nc files saved to {csv_file}")
    
    # Save list as .npy file
    np.save(npy_file, file_list)
    print(f"List of .nc files saved to {npy_file}")

    # Download files in parallel
    urls = [file['File URL'] for file in file_list]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(download_file, urls, [download_path] * len(urls))

def download_file(url, download_path):
    filename = os.path.join(download_path, os.path.basename(url))
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded: {filename}")

if __name__ == "__main__":
    # Define argument parser for max_workers only
    parser = argparse.ArgumentParser(description="Download .nc files and save the list to CSV and .npy.")
    parser.add_argument('--max_workers', type=int, required=True, help='Maximum number of parallel workers.')

    # Parse arguments
    args = parser.parse_args()

    # Hardcoded parameters
    url = "https://portal.nersc.gov/archive/home/a/arhoades/Shared/www/TE_ERA5_ARs/"
    csv_file = "labeled_ARs_TE.csv"
    npy_file = "labeled_ARs_TE.npy"
    start_year = 1996
    end_year = 2022
    download_path = "nc_files"

    # Ensure the download directory exists
    os.makedirs(download_path, exist_ok=True)

    # Call the function to list .nc files, download them, and save to CSV and .npy
    list_nc_files_to_csv_and_npy(url, csv_file, npy_file, start_year, end_year, download_path, args.max_workers)
