import pandas as pd
import requests
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from datetime import datetime
import os

def download_file(url, filename):
  print(f"Started downloading {filename}")
  try:
    response = requests.get(url)
    if response.status_code == 200:
      with open(filename, 'wb') as file:
        file.write(response.content)
      print(f"Downloaded: {filename}")
    else:
      print(f"Failed to download: {url}")
  except Exception as e:
    print(f"Exception downloading {url}: {e}")


def download_goes_historical(start_date, end_date, max_workers):
  # Placeholder for the actual downloading logic
  print(f"Downloading GOES data from {start_date} to {end_date} using {max_workers} workers...")
  
  # Get the URLS
  csv_filename = 'goes_west_url_filtered.csv'
  df = pd.read_csv(csv_filename)
  df['Date'] = pd.to_datetime(df['Date'])
  filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

  # Set up the directory here
  # Download directory
  download_directory = 'AR_events'

  if os.path.exists(download_directory):
    print("Directory already exists.")
  else:
    os.mkdir(download_directory)
    print(f"Directory {download_directory} created successfully.")

  # List to hold futures
  futures = []

  # Use ThreadPoolExecutor for concurrent downloading
  with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Adjust max_workers as needed
    for index, row in filtered_df.iterrows():
      url = row['URL']
      filename = os.path.join(download_directory,row['URL'].split('/')[-1])
      futures.append(executor.submit(download_file, url, filename))

    # Process results as they complete
    for future in as_completed(futures):
      print(future.result())

  print("Download complete.")


## Code for running the script from the terminal
def validate_date(date_text):
  try:
    return datetime.strptime(date_text, '%Y-%m-%d')
  except ValueError:
    raise ValueError(f"Incorrect date format for '{date_text}', should be YYYY-MM-DD")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Download GOES historical data.')
  parser.add_argument('start_date', type=str, help='Start date in YYYY-MM-DD format')
  parser.add_argument('end_date', type=str, help='End date in YYYY-MM-DD format')
  parser.add_argument('max_workers', type=int, help='Maximum number of workers')

  args = parser.parse_args()

  # Validate the dates
  try:
    start_date = validate_date(args.start_date)
    end_date = validate_date(args.end_date)
    if end_date < start_date:
      print("End date must be after start date.")
      exit(1)
  except ValueError as e:
    print(e)
    exit(1)
  
  # Validate max_workers
  if args.max_workers <= 0:
    print("Max workers must be a positive integer.")
    exit(1)

  # Call the function with validated inputs
  download_goes_historical(start_date, end_date, args.max_workers)

