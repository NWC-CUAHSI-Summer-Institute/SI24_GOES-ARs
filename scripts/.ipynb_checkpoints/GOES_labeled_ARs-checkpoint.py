import os
import pandas as pd
from datetime import datetime, timedelta
from goes2go import GOES
import time
from concurrent.futures import ThreadPoolExecutor

def goes_get(start_date, end_date, save_path, satellite_name, product_name, domain_name, bands_name):
    """
    Retrieves GOES satellite data for the specified parameters and saves it to the given path.

    Parameters:
    start_date (str): The start date for data retrieval in 'YYYY-MM-DD HH:MM:SS' format.
    end_date (str): The end date for data retrieval in 'YYYY-MM-DD HH:MM:SS' format.
    save_path (str): The directory path where the data will be saved.
    satellite_name (str): The name of the satellite.
    product_name (str): The product name to retrieve.
    domain_name (str): The domain name for the data.
    bands_name (list): The list of band names to retrieve.

    Returns:
    dict: Information about the retrieved data.
    """
    
    # Record the start time to measure the execution time of the script
    start_time = time.time()
    
    # Initialize a GOES object with the specified parameters
    G = GOES(satellite=satellite_name, product=product_name, domain=domain_name, bands=bands_name)
    
    # Ensure the directory exists before saving the data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Retrieve data within the specified date range and save it to the specified directory
    data_info = G.timerange(start=start_date, end=end_date, save_dir=save_path)
    
    # Calculate and print the total run time of the script in hours and minutes
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes = remainder // 60
    print(f"Run Time: {int(hours)} hours, {int(minutes)} minutes")
    
    # Return the data information retrieved by the GOES object
    return data_info

if __name__ == "__main__":
    # Read dates from CSV file
    dates_df = pd.read_csv('AR_dates_g17.csv')
    dates_list = pd.to_datetime(dates_df['Dates']).tolist()

    save_path_base = 'output_labeled_ARs'
    satellite_name = 'goes17'
    product_name = 'ABI-L1b-RadF'
    domain_name = 'C'
    bands_name = list(range(1, 17))  # Bands from 1 to 16
    
    # Define the maximum number of workers (threads) for parallel processing
    max_workers = 38  # Adjust this based on the available resources and parallel efficiency
    
    # Create a ThreadPoolExecutor with max_workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # Loop through dates and hours for batch processing
        for date in dates_list:
            for hour in ['00', '06', '12', '18']:  # Specific hours to download data
                time_str = f"{hour}:00:00"
                start_time = datetime.combine(date.date(), datetime.strptime(time_str, '%H:%M:%S').time())
                end_time = start_time + timedelta(minutes=10)

                # Construct save path for this specific date and hour
                band_save_path = os.path.join(f"{save_path_base}/")
                os.makedirs(band_save_path, exist_ok=True)  # Create the directory if it doesn't exist

                # Submit goes_get function to executor (thread pool)
                future = executor.submit(
                    goes_get,
                    start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    band_save_path,
                    satellite_name,
                    product_name,
                    domain_name,
                    bands_name
                )
                futures.append(future)
        
        # Wait for all futures (parallel tasks) to complete
        for future in futures:
            data_information = future.result()
            # Optionally use data_information here as needed
            # print(f"Data retrieval completed successfully.")
