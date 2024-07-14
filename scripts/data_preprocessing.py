import os
import numpy as np
import xarray as xr
from scipy import ndimage
from skimage.transform import resize
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def day_of_year_to_timestep(day, hour, year):
    # Placeholder for converting day of year and hour to a timestep
    return day, hour

def process_chunk(i, j, sat_data, combined_mask_upscaled, chunk_size, overlap, input_vars):
    start_row = i * (chunk_size[0] - overlap)
    start_col = j * (chunk_size[1] - overlap)
    end_row = start_row + chunk_size[0]
    end_col = start_col + chunk_size[1]

    num_channels = len(input_vars) + 2
    chunk_data = np.empty((*chunk_size, num_channels))
    chunk_mask = np.empty((*chunk_size, 1))

    for k, var in enumerate(input_vars):
        data = sat_data[var].values
        data_chunk = data[start_row:end_row, start_col:end_col]
        data_chunk = np.nan_to_num(data_chunk, nan=0.0)
        chunk_data[:, :, k] = data_chunk

    downscaled_full_image = resize(sat_data['M6C08'].values, chunk_size, mode='reflect', anti_aliasing=True)
    downscaled_full_image = np.nan_to_num(downscaled_full_image, nan=0.0)

    chunk_data[:, :, len(input_vars)] = downscaled_full_image
    chunk_data[:, :, len(input_vars) + 1] = 1

    mask_chunk = combined_mask_upscaled[start_row:end_row, start_col:end_col]
    mask_chunk = np.nan_to_num(mask_chunk, nan=0.0)
    chunk_mask[:, :, 0] = mask_chunk

    return (chunk_data, chunk_mask, i, j)

def preprocess_file(file, sat_data_dir, mask_data_dir, input_vars, input_shape, chunk_size, overlap, lat_range, lon_range, save_dir, max_workers):
    year, filename = file
    num_chunks_x = (input_shape[0] - overlap) // (chunk_size[0] - overlap)
    num_chunks_y = (input_shape[1] - overlap) // (chunk_size[1] - overlap)
    num_chunks = num_chunks_x * num_chunks_y
    num_channels = len(input_vars) + 2

    try:
        day = int(filename.split('_')[0])
        hour = int(filename.split('_')[1])
        ts, mo = day_of_year_to_timestep(day, hour, int(year))

        ld_files = [f for f in os.listdir(mask_data_dir) if year + str(mo).zfill(2) in f and 'nff' not in f]
        if len(ld_files) > 0:
            ld = xr.open_dataset(os.path.join(mask_data_dir, ld_files[0]))
            ld = ld.sel(latitude=slice(*lat_range), longitude=slice(*lon_range))
            data = ld['AR_binary_tag'][ts, :, :].values

            binary_mask = data.astype(float)
            sigma = 20
            smoothed_mask = ndimage.gaussian_filter(binary_mask, sigma=sigma)
            combined_mask = np.where(data == 1, 1, 2 * smoothed_mask)
            combined_mask = np.where(combined_mask >= 1, 1, combined_mask)

            combined_mask_upscaled = resize(combined_mask, input_shape, mode='reflect', anti_aliasing=True)

            sat = xr.open_dataset(os.path.join(sat_data_dir, year, filename))

            X = np.empty((num_chunks, *chunk_size, num_channels))
            y = np.empty((num_chunks, *chunk_size, 1))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(num_chunks_x):
                    for j in range(num_chunks_y):
                        futures.append(executor.submit(process_chunk, i, j, sat, combined_mask_upscaled, chunk_size, overlap, input_vars))
                
                for future in as_completed(futures):
                    try:
                        chunk_data, chunk_mask, i, j = future.result()
                        chunk_idx = i * num_chunks_y + j
                        X[chunk_idx] = chunk_data
                        y[chunk_idx] = chunk_mask
                    except Exception as e:
                        print(f"Error processing chunk {i}, {j} for file {filename}: {e}")

            save_path_X = os.path.join(save_dir, f"X_{year}_{filename}.npy")
            save_path_y = os.path.join(save_dir, f"y_{year}_{filename}.npy")
            np.save(save_path_X, X)
            np.save(save_path_y, y)
        else:
            print(f'No labeled file found for {filename}')
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

def preprocess_and_save_data(sat_data_dir, mask_data_dir, input_vars, input_shape, chunk_size, overlap, lat_range, lon_range, save_dir, max_workers):
    file_list = [(y, f) for y in os.listdir(sat_data_dir) for f in os.listdir(os.path.join(sat_data_dir, y)) if os.path.isdir(os.path.join(sat_data_dir, y))]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file in file_list:
            futures.append(executor.submit(preprocess_file, file, sat_data_dir, mask_data_dir, input_vars, input_shape, chunk_size, overlap, lat_range, lon_range, save_dir, max_workers))
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and save satellite data.")
    parser.add_argument("--sat_data_dir", type=str, required=True, help="Directory containing satellite data.")
    parser.add_argument("--mask_data_dir", type=str, required=True, help="Directory containing mask data.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save preprocessed data.")
    parser.add_argument("--max_workers", type=int, required=True, help="Maximum number of parallel workers.")
    args = parser.parse_args()

    input_vars = [f'M6C{str(i).zfill(2)}' for i in range(1, 17)]
    input_shape = (4441, 4996)
    chunk_size = (896, 896)
    overlap = 112
    lat_range = (90, 0)
    lon_range = (180, 270)

    preprocess_and_save_data(
        args.sat_data_dir,
        args.mask_data_dir,
        input_vars,
        input_shape,
        chunk_size,
        overlap,
        lat_range,
        lon_range,
        args.save_dir,
        min(args.max_workers, os.cpu_count() - 1)  # Use a safe number of workers
    )
