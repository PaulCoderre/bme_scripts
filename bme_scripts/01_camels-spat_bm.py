#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import os
import hydrobm
import pandas as pd
from hydrobm.calculate import calc_bm
import numpy as np
import sys

# ### Inputs

# In[2]:

# ==================
# This script calculates the benchmark timeseries for each HydroBM benchmark at each of the FUSE CAMELS-SPAT basins 
# It outputs a csv for the benchmark flows at each gauge to output_dir
# ==================

# Define FUSE sumulation base directory
base_dir ='/work/comphyd_lab/users/cyril.thebault/Postdoc_Ucal/02_DATA/FUSE_Farahani/CAMELS-spat/Lumped/em-earth/KGE/1/'

output_dir = '../camels-spat/final_bm_flows/'



# Define calibration and validation periods (start and end)
calibration_range = ('1982-10-01', '1989-09-30')
validation_range  = ('1989-10-01', '2009-09-30')


# Specify the benchmarks and metrics to calculate
benchmarks = [
        # Streamflow benchmarks
        "mean_flow",
        "median_flow",
        "annual_mean_flow",
        "annual_median_flow",
        "monthly_mean_flow",
        "monthly_median_flow",
        "daily_mean_flow",
        "daily_median_flow",
        "eckhardt_baseflow",

        # Long-term rainfall-runoff ratio benchmarks
        "rainfall_runoff_ratio_to_all",
        "rainfall_runoff_ratio_to_annual",
        "rainfall_runoff_ratio_to_monthly",
        "rainfall_runoff_ratio_to_daily",
        "rainfall_runoff_ratio_to_timestep",

         # Short-term rainfall-runoff ratio benchmarks
        "monthly_rainfall_runoff_ratio_to_monthly",
        "monthly_rainfall_runoff_ratio_to_daily",
        "monthly_rainfall_runoff_ratio_to_timestep",

        # Precipitation deviation benchmarks
        "annual_scaled_daily_mean_flow",
        "monthly_scaled_daily_mean_flow",

        # Schaefli & Gupta (2007) benchmarks
        "scaled_precipitation_benchmark",  # equivalent to "rainfall_runoff_ratio_to_daily"
        "adjusted_precipitation_benchmark",

        # Parsimonious models
        "adjusted_smoothed_precipitation_benchmark", # from Schaefli & Gupta (2007)
        "baseflow_with_event_peaks",
        "api_scaled_flow"
     ]

# Read system number
if len(sys.argv) > 1:
    run_number = int(sys.argv[1])
else:
    run_number = 1  # default if none provided

runs_per_script = 24

# ### Pre-Processing

# In[3]:


# Make output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


# In[4]:

# =====================================
# List all directories that start with CAN or USA
folders = [f for f in os.listdir(base_dir)
           if os.path.isdir(os.path.join(base_dir, f)) and (f.startswith('CAN') or f.startswith('USA'))]

# Sort folders for reproducibility
folders = sorted(folders)

# Get index numbers for runs
low_range = run_number * runs_per_script - runs_per_script
upper_range = run_number * runs_per_script

# Slice safely
folders_subset = folders[low_range:upper_range]

print(f"Run number: {run_number}")
print(f"Processing directories {low_range} to {upper_range - 1} (total {len(folders_subset)})")
print(folders_subset)

if not folders_subset:
    print(f"No directories found for run {run_number}. Exiting.")
    sys.exit(0)


# In[5]:


# Create paths to the 'input' folders
input_dirs = [os.path.join(base_dir, folder, 'input') for folder in folders_subset]


# In[6]:



# In[7]:


# Iterate through each path and open the .nc file containing 'input' in its name
for input_dir in input_dirs:
    for file in os.listdir(input_dir):
        # Find input NetCDF
        if file.endswith('.nc') and 'input' in file:

            
            # Save filename to variable
            station_id = file.split('_input')[0]

            # Create full filepath and read inputs
            file_path = os.path.join(input_dir, file)
            print(f"Opening {file_path}...")
            ds = xr.open_dataset(file_path)

            # =================================================
            # Pre-processing
            # Extract variables of interest (squeeze removes singleton lat/lon dims)
            bm_input = ds[['pr', 'q_obs', 'temp']].to_dataframe().reset_index()

            # Check for NaNs in key columns
            nan_mask = bm_input[['pr', 'q_obs', 'temp']].isna().any(axis=1)
            if nan_mask.any():
                print(f"Warning: NaNs detected in {station_id} input data!")
                print(bm_input.loc[nan_mask, ['time', 'pr', 'q_obs', 'temp']])
            
            # Drop lat/lon since they're constant
            bm_input = bm_input.drop(columns=['latitude', 'longitude'])
            
            # Set date/time as index
            bm_input = bm_input.set_index('time')


            # Vectorized mask creation
            bm_input['cal_mask'] = (bm_input.index >= pd.to_datetime(calibration_range[0])) & \
                                   (bm_input.index <= pd.to_datetime(calibration_range[1]))
            
            bm_input['val_mask'] = (bm_input.index >= pd.to_datetime(validation_range[0])) & \
                                   (bm_input.index <= pd.to_datetime(validation_range[1]))

            # Trim to only include calibration or validation periods
            bm_input = bm_input[bm_input['cal_mask'] | bm_input['val_mask']]

            # ===========================================
            # Calculate Benchmarks
            print('Calculating Benchmarks')
    
            # Calculate the benchmarks and scores
            benchmark_flows, scores = calc_bm(
                bm_input,
        
                # Time period selection
                bm_input['cal_mask'],
                val_mask=bm_input['val_mask'],
        
                # Variable names in 'data'
                precipitation="pr",
                streamflow="q_obs",
        
                # Benchmark choices
                benchmarks=benchmarks,
                metrics=['nse', 'kge'],
                optimization_method="brute_force",
        
                # Snow model inputs
                calc_snowmelt=True,
                temperature="temp",
                snowmelt_threshold=0.0,
                snowmelt_rate=3.0,
            )

            # Add lat and lon
            lat_value = ds['latitude'].values.item()  # safer than float()
            lon_value = ds['longitude'].values.item()
            
            benchmark_flows['latitude'] = lat_value
            benchmark_flows['longitude'] = lon_value


            # Reset indices for safe merging
            benchmark_flows = benchmark_flows.reset_index()
            bm_input_reset = bm_input.reset_index()
            
            # Select columns to merge: observed flow + masks
            bm_merge = bm_input_reset[['time', 'q_obs', 'cal_mask', 'val_mask']]
            
            # Merge on 'time'
            benchmark_flows = benchmark_flows.merge(bm_merge, on='time', how='left')

            # ===================================
            # Save benchmark_flows to CSV
            out_file = os.path.join(output_dir, f"{station_id}_BM.csv")
            benchmark_flows.to_csv(out_file)
            print(f"Saved benchmark flows to {out_file}")

            ds.close()

            





