#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import networkx as nx
import hydrobm
import xarray as xr
import matplotlib.pyplot as plt
from hydrobm.calculate import calc_bm
import matplotlib as mpl
import seaborn as sns


# ### Description
# _________________
# This script requires data for precipitation, temperature, observed streamflow, simulated streamflow, subbasin order (for calculating upstream precipitation), and subbasin IDs. This script generates the timeseries for each HydroBM benchmark by iterating through a list of IDs. It then computes the skill score between the model simulated timeseries and benchmark timeseries for each benchmark.  

# In[2]:


# Input file paths

pobs= '../SMM_Models/hype/model/v10_model/final_model/hds_v3/Pobs.txt' # precipitation for each subbasin

tobs= '../SMM_Models/hype/model/v10_model/final_model/hds_v3/Tobs.txt' # temperature for each subbasin

qobs= '../SMM_Models/hype/model/v10_model/final_model/hds_v3/Qobs.txt' # observed flow for each analyzed subbasin

#cout = '../SMM_Models/hype/model/v10_model/final_model/hds_v3/results/timeCOUT_DD.txt' # simulated flow at each subbasin

cout = 'inputs/raven_streamflow.csv'

geodata= '../SMM_Models/hype/model/v10_model/final_model/hds_v3/GeoData.txt' # subbasin downstream order

gauge_info= 'inputs/gauge_info.csv' # list of subbasins to iterate through

output_dir= './results/raven_nse_skill/' # output directory

# Define skill score calculation
skill_score_metric = 'nse' # method of skill score, options are rmse and nse

start_date = "1980-10-01"
end_date = "2015-09-30"


# Define calibration and validation periods
calibration_ranges = [('1981-10-01', '1984-09-30'), # Change to 1981 for Raven and 1980 for the rest
               ('1989-10-01', '1998-09-30'),
               ('2003-10-01', '2007-09-30'),
               ('2012-10-01', '2015-09-30')]

validation_ranges = [('1984-10-01', '1989-09-30'),
               ('1998-10-01', '2003-09-30'),
               ('2007-10-01', '2012-09-30')]


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

        # Schaefli & Gupta (2007) benchmarks
        "scaled_precipitation_benchmark",  # equivalent to "rainfall_runoff_ratio_to_daily"
        "adjusted_precipitation_benchmark",
        "adjusted_smoothed_precipitation_benchmark",
     ]

# Define periods for skill score calculation and corresponding masks
periods = {
    'calibration': 'cal_mask',
    'validation': 'val_mask',
    'all': None  # No mask for 'all'
}


# ### Pre Processing

# In[3]:


# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# =============================
# Process data

# Read data
pobs= pd.read_csv(pobs, index_col=0, sep='\t') 
tobs= pd.read_csv(tobs, index_col=0, sep='\t') 
qobs= pd.read_csv(qobs, index_col=0, sep='\t') 
#cout= pd.read_csv(cout, index_col=0, sep='\t', skiprows=1) 
cout= pd.read_csv(cout, index_col=0) 
geodata= pd.read_csv(geodata, index_col=0, sep='\t') 
gauge_info= pd.read_csv(gauge_info, index_col=1)

# convert index to datetime
pobs.index = pd.to_datetime(pobs.index)
tobs.index = pd.to_datetime(tobs.index)
qobs.index = pd.to_datetime(qobs.index)
cout.index = pd.to_datetime(cout.index)

# Set index to int
geodata.index = geodata.index.astype(int)
gauge_info.index = gauge_info.index.astype(int)

# Convert column headers to integers
pobs.columns = pobs.columns.astype(int)
tobs.columns = tobs.columns.astype(int)
qobs.columns = qobs.columns.astype(int)
cout.columns = cout.columns.astype(int)

# trim to match start and end dates
pobs = pobs.loc[start_date:end_date]
tobs = tobs.loc[start_date:end_date]
qobs = qobs.loc[start_date:end_date]
cout = cout.loc[start_date:end_date]

# Convert the calibration and validation ranges to Pandas Timestamps
calibration_ranges = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in calibration_ranges]
validation_ranges = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in validation_ranges]

# replace missing values with nan in streamflow
qobs.replace(-9999, np.nan, inplace=True)

# =======================
# Create upstream to downstream digraph
riv_graph = nx.DiGraph()

# Add edges from DataFrame
for idx, row in geodata.iterrows():
    if row['maindown'] != '0':  # Skip if maindown is '0'
        riv_graph.add_edge(idx, row['maindown'])

# =======================
# Convert precipitation to m3
        
# Set area column to numeric
geodata['area'] = pd.to_numeric(geodata['area'])

# Create dictionary with subbasin ID and area
area_dict = geodata['area'].to_dict()

# Convert pobs from mm to m
pobs= pobs / 1000 # mm to m

# Multiply each column in pobs by the corresponding area value in area_dict to get m3
for col in pobs.columns:
    if col in area_dict:
        pobs[col] *= area_dict[col]


# ========================================
# Test case
# gauge_info = gauge_info.loc[[58308]]

# Define KGE function
def compute_kge(simulated, observed):
    """
    Computes KGE (Kling-Gupta Efficiency) between observed and simulated values.
    """
    simulated = np.asarray(simulated, dtype=float)
    observed = np.asarray(observed, dtype=float)

    # Drop NaNs pairwise
    mask = ~np.isnan(simulated) & ~np.isnan(observed)
    simulated = simulated[mask]
    observed = observed[mask]

    if len(simulated) == 0 or len(observed) == 0:
        return np.nan

    mean_obs = np.mean(observed)
    mean_sim = np.mean(simulated)
    std_obs = np.std(observed)
    std_sim = np.std(simulated)

    if mean_obs == 0 or std_obs == 0:  # avoid div by zero in ratios or corr
        return np.nan

    # Correlation (only if both series have variability)
    if std_obs > 0 and std_sim > 0:
        r = np.corrcoef(observed, simulated)[0, 1]
    else:
        r = np.nan

    # print(r)

    beta = mean_sim / mean_obs
    gamma = std_sim / std_obs

    if np.isnan(r) or np.isnan(beta) or np.isnan(gamma):
        return np.nan

    kge = 1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)
    return kge

    

# Define funciton for calculating skill scores
def calculate_skill_score(observed: pd.Series, simulated: pd.Series, benchmark: pd.Series, method: str ) -> float:
    """
    Calculate skill score based on NSE (sum of squared errors) or RMSE.

    Parameters
    ----------
    observed : pd.Series
        Observed values.
    simulated : pd.Series
        Simulated values.
    benchmark : pd.Series
        Benchmark values to compare against.
    method : str, optional
        Skill score method: 'nse' (default) or 'rmse'.

    Returns
    -------
    float
        Skill score. Returns np.nan if benchmark error is zero.
    """
    if method.lower() == 'nse':
        # NSE-based: sum of squared errors
        se_sim = ((observed - simulated) ** 2).sum()
        se_bm = ((observed - benchmark) ** 2).sum()
        skill_score = 1 - se_sim / se_bm if se_bm != 0 else np.nan

    elif method.lower() == 'rmse':
        # RMSE-based: root mean squared error
        rmse_sim = np.sqrt(((observed - simulated) ** 2).mean())
        rmse_bm = np.sqrt(((observed - benchmark) ** 2).mean())
        skill_score = 1 - rmse_sim / rmse_bm if rmse_bm != 0 else np.nan

    elif method.lower() == 'kge':
        # KGE-based
        kge_sim = compute_kge(simulated, observed)
        kge_bm = compute_kge(benchmark, observed)

        # print(kge_sim)
        # print(kge_bm)
    
        # Return NaN if either KGE is undefined or benchmark is exactly 1
        if np.isnan(kge_sim) or np.isnan(kge_bm) or kge_bm == 1:
            return np.nan

        skill_score = (kge_sim - kge_bm) / (1 - kge_bm)

    else:
        raise ValueError("Invalid method. Choose 'nse', 'rmse' or 'kge'.")

    return skill_score


# ### Calculate Skill Score

# In[5]:


# ===============================================
# Global dictionary to store skill scores for all subbasins
all_subbasin_scores = {}

# Iterate over each subbasin in list
for subbasin in gauge_info.index:

    print(f'Analyzing Subbasin {subbasin}')

    # =================================
    # Create HydroBM input
    
    # Find upstream segments for the given subbasin
    upstream_segments = list(nx.ancestors(riv_graph, subbasin))
    
    # Add the target segment to the upstream segments
    upstream_segments.append(subbasin)


    # Sum upstream precipitation
    precipitation_sum = pd.DataFrame(
        pobs[upstream_segments].sum(axis=1),
        columns=['precipitation']
    )
    
    # Mean upstream temperature
    temperature_mean = pd.DataFrame(
        tobs[upstream_segments].mean(axis=1),
        columns=['temperature']
    )

    # Create hydrobm input dataframe
    bm_input = pd.DataFrame({
        'streamflow': qobs[subbasin] * 84600,  # Streamflow volume for the given subbasin
        'precipitation': precipitation_sum['precipitation'],  # Sum of upstream precipitation in m3
        'temperature': temperature_mean['temperature'] # mean upstream temperature of the subbasin
    })

    # Create the cal_mask column
    bm_input['cal_mask'] = bm_input.index.to_series().apply(
        lambda x: any(pd.to_datetime(start) <= x <= pd.to_datetime(end) for start, end in calibration_ranges)
    )

    # Create the val_mask column
    bm_input['val_mask'] = bm_input.index.to_series().apply(
        lambda x: any(pd.to_datetime(start) <= x <= pd.to_datetime(end) for start, end in validation_ranges)
    )

    print('Calculating Benchmarks')
    
    # Calculate the benchmarks and scores
    benchmark_flows, scores = calc_bm(
        bm_input,

        # Time period selection
        bm_input['cal_mask'],
        val_mask=bm_input['val_mask'],

        # Variable names in 'data'
        precipitation="precipitation",
        streamflow="streamflow",

        # Benchmark choices
        benchmarks=benchmarks,
        metrics=['nse', 'kge'],
        optimization_method="brute_force",

        # Snow model inputs
        calc_snowmelt=True,
        temperature="temperature",
        snowmelt_threshold=0.0,
        snowmelt_rate=3.0,
    )

    
    # ====================================
    # Prepare to calculate skill scores
    
    # Prepare observed and simulated flows as DataFrames
    obs_df = pd.DataFrame({'observed_flow': qobs[subbasin] * 86400})
    sim_df = pd.DataFrame({'simulated_flow': cout[subbasin] * 86400})

    # Prepare cal and val masks as DataFrames
    cal_mask_df = pd.DataFrame({'cal_mask': bm_input['cal_mask']})
    val_mask_df = pd.DataFrame({'val_mask': bm_input['val_mask']})
    
    # Merge onto benchmark_flows using index
    benchmark_flows = benchmark_flows.merge(obs_df, left_index=True, right_index=True, how='left')
    benchmark_flows = benchmark_flows.merge(sim_df, left_index=True, right_index=True, how='left')
    
    # Merge masks onto benchmark_flows using index
    benchmark_flows = benchmark_flows.merge(cal_mask_df, left_index=True, right_index=True, how='left')
    benchmark_flows = benchmark_flows.merge(val_mask_df, left_index=True, right_index=True, how='left')

    # ======================
    # Calculate skill scores

    # Get list of benchmark columns
    bm_columns = [col for col in benchmark_flows.columns if col.startswith('bm_')]

    # Dictionary to store results
    skill_scores = {period: {} for period in periods}

    # Iterate over periods
    for period_name, mask_col in periods.items():

        print(f'Calculating Skill Score for: {period_name}')

        # Trim to only required period
        if mask_col is not None:
            df_period = benchmark_flows[benchmark_flows[mask_col]]
        else:
            df_period = benchmark_flows.copy()
            
        
        for bm_col in bm_columns:

            # Calculate skill score based on metric
            skill_score = calculate_skill_score(
                observed=df_period['observed_flow'],
                simulated=df_period['simulated_flow'],
                benchmark=df_period[bm_col],
                method=skill_score_metric  # nse or rmse 
            )

            # Remove benchmarks that don't work on unseen data
            if bm_col in ['bm_annual_mean_flow', 'bm_annual_median_flow'] and period_name in ['calibration', 'validation', 'all']:
                skill_score = np.nan
        
            # Store
            skill_scores[period_name][bm_col] = skill_score

    # Store subbasin results in the global dictionary
    all_subbasin_scores[subbasin] = skill_scores


# In[6]:


# ===============================
# Save outputs

# Convert global dictionary to a multi-index DataFrame for easy access
skill_scores_df = pd.concat({
    subbasin: pd.DataFrame(sub_scores) 
    for subbasin, sub_scores in all_subbasin_scores.items()
}, names=['subbasin', 'benchmark'])

# Reset MultiIndex to get subbasin and benchmark as columns
skill_scores_long = skill_scores_df.reset_index()
skill_scores_long = skill_scores_long.rename(columns={'level_0': 'subbasin', 'level_1': 'benchmark'})

# Now, melt the periods into a single column
skill_scores_long = skill_scores_long.melt(
    id_vars=['subbasin', 'benchmark'],
    value_vars=['calibration', 'validation', 'all'],
    var_name='period',
    value_name='skill_score'
)

# Save to CSV
output_file = os.path.join(output_dir, 'skill_scores.csv')
skill_scores_long.to_csv(output_file, index=False)

print(f"Skill scores saved to {output_file}")


# In[ ]:




