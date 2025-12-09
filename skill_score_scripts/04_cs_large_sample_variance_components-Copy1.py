#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import fft
import glob
import os

# ====================
# This script reads teh benchmark flows csv to calculate the variance components of observed streamflow at each CAMELS-SPAT
# gauge. It outputs the decomposition along with the NSE of each variance component from each benchmark. It does this only
# For the validation period.
# ====================

# In[2]: 

# Directory containing benchmark flow csv files
csv_directory = '../camels-spat/02_results/final_bm_flows/'  

# Output directory
output_directory = '../camels-spat/02_results/skill_scores/'

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Output files
output_file = os.path.join(output_directory, 'benchmark_variance.csv')
benchmark_summary_file = os.path.join(output_directory, 'benchmark_summary_across_files.csv')



def decompose_streamflow(flow_data, value_col='cout'):
    """
    Decompose streamflow time series into seasonal, interannual, and irregular components
    following the methodology from Ruzzante et al. (2025).
    
    Parameters:
    -----------
    flow_data : pd.DataFrame
        DataFrame with DatetimeIndex and flow column
    value_col : str
        Name of the column containing flow values
    
    Returns:
    --------
    dict containing:
        - 'seasonal': pd.Series of seasonal component
        - 'interannual': pd.Series of interannual component  
        - 'irregular': pd.Series of irregular component
        - 'anomalies': pd.Series of deseasonalized data
        - 'variance_fractions': dict with variance fraction for each component
    """
    
    # Ensure datetime index
    if not isinstance(flow_data.index, pd.DatetimeIndex):
        flow_data.index = pd.to_datetime(flow_data.index)
    
    # Get the flow series and ensure numeric
    flow = pd.to_numeric(flow_data[value_col], errors='coerce')
    
    # Replace -9999 with NaN (common missing value indicator)
    flow = flow.replace(-9999, np.nan)
    
    # Drop NaN values but keep the index
    valid_idx = flow.notna()
    flow = flow[valid_idx]
    
    if len(flow) == 0:
        raise ValueError(f"No valid data found for column {value_col}")
    
    # 1. Calculate seasonal component (mean for each calendar day)
    # Create a dataframe for grouping
    flow_df = pd.DataFrame({'flow': flow})
    flow_df['month'] = flow_df.index.month
    flow_df['day'] = flow_df.index.day
    
    # Calculate seasonal means
    seasonal_means = flow_df.groupby(['month', 'day'])['flow'].transform('mean')
    seasonal = pd.Series(seasonal_means.values, index=flow.index, name='seasonal')
    
    # 2. Calculate anomalies by removing seasonal component
    anomalies = flow - seasonal
    
    # 3. Apply FFT to anomalies
    n = len(anomalies)
    fft_values = fft.rfft(anomalies.values)
    frequencies = fft.rfftfreq(n, d=1)  # d=1 for daily data
    
    # 4. Separate into interannual and irregular components
    # Cutoff frequency: 2 year^-1 = 2/365.25 day^-1
    cutoff_freq = 2 / 365.25
    
    # Create masks for frequency separation
    interannual_mask = np.abs(frequencies) < cutoff_freq
    irregular_mask = np.abs(frequencies) >= cutoff_freq
    
    # Separate FFT components
    fft_interannual = fft_values.copy()
    fft_interannual[irregular_mask] = 0
    
    fft_irregular = fft_values.copy()
    fft_irregular[interannual_mask] = 0
    
    # 5. Inverse FFT to get time domain components
    interannual = pd.Series(
        fft.irfft(fft_interannual, n=n),
        index=flow.index,
        name='interannual'
    )
    
    irregular = pd.Series(
        fft.irfft(fft_irregular, n=n),
        index=flow.index,
        name='irregular'
    )
    
    # 6. Calculate variance fractions
    var_total = flow.var()
    var_seasonal = seasonal.var()
    var_interannual = interannual.var()
    var_irregular = irregular.var()
    
    variance_fractions = {
        'seasonal': var_seasonal / var_total,
        'interannual': var_interannual / var_total,
        'irregular': var_irregular / var_total
    }
    
    # Verify orthogonality: variance fractions should sum to 1
    variance_sum = sum(variance_fractions.values())
    if not np.isclose(variance_sum, 1.0, rtol=1e-3):
        print(f"Warning: Variance fractions sum to {variance_sum:.6f}, not 1.0")
        print(f"  This indicates the decomposition may not be perfectly orthogonal.")
        print(f"  Difference from 1.0: {abs(1.0 - variance_sum):.6e}")
    
    return {
        'seasonal': seasonal,
        'interannual': interannual,
        'irregular': irregular,
        'anomalies': anomalies,
        'variance_fractions': variance_fractions,
        'variance_sum': variance_sum,
        'original': flow
    }


# In[3]:


# Calculate component NSEs (Equation 2 from the paper)
def calculate_component_nse(obs_component, sim_component):
    """Calculate NSE for a specific component"""
    numerator = np.sum((obs_component - sim_component)**2)
    denominator = np.sum((obs_component - obs_component.mean())**2)
    
    # Handle cases where denominator is zero (no variability in observations)
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    
    return 1 - (numerator / denominator)


def calculate_kge(obs, sim):
    """
    Calculate Kling-Gupta Efficiency (KGE) and its components using Gupta et al. 2009 formulation.
    
    KGE = 1 - sqrt((r-1)² + (β-1)² + (α-1)²)
    
    Where:
    - r = Pearson correlation coefficient
    - β = bias ratio = mean(sim) / mean(obs)
    - α = variability ratio = std(sim) / std(obs)  [Gupta et al. 2009]
    
    Note: This differs from Kling et al. 2012 which uses CV ratio: (std(sim)/mean(sim)) / (std(obs)/mean(obs))
    
    Parameters:
    -----------
    obs : pd.Series or np.array
        Observed values
    sim : pd.Series or np.array
        Simulated values
    
    Returns:
    --------
    dict containing:
        - 'kge': Overall KGE value
        - 'r': Correlation component
        - 'beta': Bias ratio component
        - 'alpha': Variability ratio component (std ratio, not CV ratio)
    """
    # Convert to numpy arrays and remove NaN values
    obs = np.array(obs)
    sim = np.array(sim)
    
    # Create mask for valid values
    valid = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[valid]
    sim = sim[valid]
    
    if len(obs) == 0:
        return {
            'kge': np.nan,
            'r': np.nan,
            'beta': np.nan,
            'alpha': np.nan
        }
    
    # Calculate components with error handling
    std_obs = np.std(obs, ddof=1)  # Use sample std (N-1)
    std_sim = np.std(sim, ddof=1)
    
    # Check for zero or near-zero standard deviation
    if std_obs < 1e-10 or std_sim < 1e-10:
        # If one or both have no variability, correlation is undefined
        r = np.nan
    else:
        # Calculate correlation safely
        r = np.corrcoef(obs, sim)[0, 1]
    
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    
    # Calculate beta (bias ratio)
    if abs(mean_obs) < 1e-10:
        beta = np.nan
    else:
        beta = mean_sim / mean_obs
    
    # Calculate alpha (variability ratio) - GUPTA ET AL. 2009 FORMULATION
    # This is the standard deviation ratio, NOT normalized by means
    if abs(std_obs) < 1e-10:
        alpha = np.nan
    else:
        alpha = std_sim / std_obs
    
    # Calculate KGE - handle NaN components
    if np.isnan(r) or np.isnan(beta) or np.isnan(alpha):
        kge = np.nan
    else:
        kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (alpha - 1)**2)
    
    return {
        'kge': kge,
        'r': r,
        'beta': beta,
        'alpha': alpha  # Changed from 'gamma' to 'alpha'
    }


def calculate_variance_component_nses_and_kge(obs_decomp, sim_decomp):
    """
    Calculate NSE values for each variance component and KGE for overall time series.
    
    Parameters:
    -----------
    obs_decomp : dict
        Decomposition results for observed data (from decompose_streamflow)
    sim_decomp : dict
        Decomposition results for simulated data (from decompose_streamflow)
    
    Returns:
    --------
    dict containing:
        - 'nse_overall': Overall NSE
        - 'nse_seasonal': NSE for seasonal component
        - 'nse_interannual': NSE for interannual component
        - 'nse_irregular': NSE for irregular component
        - 'nse_weighted_sum': Weighted sum of component NSEs (should equal overall NSE)
        - 'kge_overall': Overall KGE
        - 'kge_r': Overall correlation component
        - 'kge_beta': Overall bias ratio component
        - 'kge_alpha': Overall variability ratio component (std ratio)
        - 'is_highly_seasonal': Boolean, True if seasonal variance fraction > 0.5
        - 'seasonal_variance_fraction': Fraction of variance that is seasonal
    """
    
    # Align indices to handle potential length mismatches
    # Find common indices between observed and simulated
    common_idx = obs_decomp['original'].index.intersection(sim_decomp['original'].index)
    
    if len(common_idx) == 0:
        print("Warning: No common time steps between observed and simulated data!")
        return {
            'nse_overall': np.nan,
            'nse_seasonal': np.nan,
            'nse_interannual': np.nan,
            'nse_irregular': np.nan,
            'nse_weighted_sum': np.nan,
            'nse_difference': np.nan,
            'kge_overall': np.nan,
            'kge_r': np.nan,
            'kge_beta': np.nan,
            'kge_alpha': np.nan,  # Changed from gamma to alpha
            'is_highly_seasonal': False,
            'seasonal_variance_fraction': np.nan,
            'observed_variance_fractions': {'seasonal': np.nan, 'interannual': np.nan, 'irregular': np.nan}
        }
    
    # Align all components to common indices
    obs_original_aligned = obs_decomp['original'].loc[common_idx]
    sim_original_aligned = sim_decomp['original'].loc[common_idx]
    
    obs_seasonal_aligned = obs_decomp['seasonal'].loc[common_idx]
    sim_seasonal_aligned = sim_decomp['seasonal'].loc[common_idx]
    
    obs_interannual_aligned = obs_decomp['interannual'].loc[common_idx]
    sim_interannual_aligned = sim_decomp['interannual'].loc[common_idx]
    
    obs_irregular_aligned = obs_decomp['irregular'].loc[common_idx]
    sim_irregular_aligned = sim_decomp['irregular'].loc[common_idx]
    
    # Calculate NSE for each component (Equation 2)
    nse_seasonal = calculate_component_nse(obs_seasonal_aligned, sim_seasonal_aligned)
    nse_interannual = calculate_component_nse(obs_interannual_aligned, sim_interannual_aligned)
    nse_irregular = calculate_component_nse(obs_irregular_aligned, sim_irregular_aligned)
    nse_overall = calculate_component_nse(obs_original_aligned, sim_original_aligned)
    
    # Calculate KGE and its components for overall flow only
    kge_overall_results = calculate_kge(obs_original_aligned, sim_original_aligned)
    
    # Get observed variance fractions (weights for Equation 3)
    var_fracs = obs_decomp['variance_fractions']
    
    # Calculate weighted sum of component NSEs (Equation 3)
    nse_weighted_sum = (
        var_fracs['seasonal'] * nse_seasonal +
        var_fracs['interannual'] * nse_interannual +
        var_fracs['irregular'] * nse_irregular
    )
    
    # Check if weighted sum matches overall NSE
    nse_difference = abs(nse_overall - nse_weighted_sum)
    if nse_difference > 1e-3:
        print(f"Warning: NSE weighted sum ({nse_weighted_sum:.6f}) differs from overall NSE ({nse_overall:.6f})")
        print(f"  Difference: {nse_difference:.6e}")
    
    # Classify as highly seasonal or not (threshold = 0.5)
    is_highly_seasonal = var_fracs['seasonal'] > 0.5
    
    results = {
        # NSE results
        'nse_overall': nse_overall,
        'nse_seasonal': nse_seasonal,
        'nse_interannual': nse_interannual,
        'nse_irregular': nse_irregular,
        'nse_weighted_sum': nse_weighted_sum,
        'nse_difference': nse_difference,
        
        # Overall KGE results
        'kge_overall': kge_overall_results['kge'],
        'kge_r': kge_overall_results['r'],
        'kge_beta': kge_overall_results['beta'],
        'kge_alpha': kge_overall_results['alpha'],  # Changed from gamma to alpha
        
        # Regime classification
        'is_highly_seasonal': is_highly_seasonal,
        'seasonal_variance_fraction': var_fracs['seasonal'],
        'observed_variance_fractions': var_fracs
    }
    
    return results


# ### Inputs

# In[4]:



# In[5]:


# Find all CSV files
csv_files = glob.glob(os.path.join(csv_directory, '*.csv'))

print(f"Found {len(csv_files)} CSV files")
print("Files to process:")
# for f in csv_files:
#     print(f"  - {os.path.basename(f)}")


# In[6]:


# Store all results
all_results = []

# Process each CSV file
for csv_path in csv_files:
    csv_name = os.path.basename(csv_path)
    print("\n" + "="*80)
    print(f"PROCESSING: {csv_name}")
    print("="*80)
    
    try:
        # Load data
        flow = pd.read_csv(csv_path)
        flow['time'] = pd.to_datetime(flow['time'])
        flow = flow.set_index('time')
        
        # Check if q_obs exists
        if 'q_obs' not in flow.columns:
            print(f"Warning: 'q_obs' column not found in {csv_name}. Skipping.")
            continue
        
        # Filter to validation period only if val_mask exists
        if 'val_mask' in flow.columns:
            n_total = len(flow)
            flow = flow[flow['val_mask'] == True].copy()
            n_val = len(flow)
            print(f"Filtered to validation period: {n_val}/{n_total} timesteps ({n_val/n_total*100:.1f}%)")
            
            if n_val == 0:
                print(f"Warning: No validation data in {csv_name}. Skipping.")
                continue
        else:
            print(f"Warning: 'val_mask' column not found in {csv_name}. Using all data.")
        
        # Decompose observed flow
        print(f"\nDecomposing observed data...")
        qobs_decomp = decompose_streamflow(flow, 'q_obs')
        
        # Calculate statistics for observed flow
        obs_mean = qobs_decomp['original'].mean()
        obs_std = qobs_decomp['original'].std()
        
        print(f"  Observed flow statistics: mean={obs_mean:.4f}, std={obs_std:.4f}")
        
        # Get all benchmark model columns
        benchmark_cols = [col for col in flow.columns if col.startswith('bm_')]
        
        if len(benchmark_cols) == 0:
            print(f"Warning: No benchmark columns found in {csv_name}. Skipping.")
            continue
        
        print(f"Found {len(benchmark_cols)} benchmark models")
        
        # Dictionary to store all benchmark decompositions
        benchmark_decomps = {}
        
        # Loop through each benchmark model and decompose
        for bm_col in benchmark_cols:
            try:
                print(f"  Decomposing {bm_col}...")
                benchmark_decomps[bm_col] = decompose_streamflow(flow, bm_col)
                print(f"    Length: {len(benchmark_decomps[bm_col]['original'])} timesteps")
            except Exception as e:
                print(f"    ✗ Failed to decompose {bm_col}: {str(e)}")
                continue
        
        print(f"  Observed data length: {len(qobs_decomp['original'])} timesteps")
        
        # Calculate NSE and KGE results for each benchmark
        nse_kge_results_all = {}
        for bm_col in benchmark_cols:
            if bm_col not in benchmark_decomps:
                print(f"  Skipping {bm_col} (decomposition failed)")
                continue
            try:
                nse_kge_results_all[bm_col] = calculate_variance_component_nses_and_kge(
                    qobs_decomp, 
                    benchmark_decomps[bm_col]
                )
            except Exception as e:
                print(f"  ✗ Failed to calculate metrics for {bm_col}: {str(e)}")
                continue
        
        # Create summary for this CSV
        for bm_col, results in nse_kge_results_all.items():
            all_results.append({
                'csv_file': csv_name,
                'benchmark': bm_col,
                'nse_overall': results['nse_overall'],
                'nse_seasonal': results['nse_seasonal'],
                'nse_interannual': results['nse_interannual'],
                'nse_irregular': results['nse_irregular'],
                'kge_overall': results['kge_overall'],
                'kge_r': results['kge_r'],
                'kge_beta': results['kge_beta'],
                'kge_alpha': results['kge_alpha'],  # Changed from kge_gamma to kge_alpha
                'obs_seasonal_var': results['observed_variance_fractions']['seasonal'],
                'obs_interannual_var': results['observed_variance_fractions']['interannual'],
                'obs_irregular_var': results['observed_variance_fractions']['irregular'],
                'is_highly_seasonal': results['is_highly_seasonal'],
                'obs_flow_mean': obs_mean,
                'obs_flow_std': obs_std
            })
        
        print(f"\n✓ Successfully processed {csv_name}")
        
    except Exception as e:
        print(f"✗ Error processing {csv_name}: {str(e)}")
        continue

# Create final summary dataframe
results_df = pd.DataFrame(all_results)


# In[7]:


# Format numeric columns
numeric_cols = ['nse_overall', 'nse_seasonal', 'nse_interannual', 'nse_irregular',
                'kge_overall', 'kge_r', 'kge_beta', 'kge_alpha',  # Changed from kge_gamma
                'obs_seasonal_var', 'obs_interannual_var', 'obs_irregular_var',
                'obs_flow_mean', 'obs_flow_std']

for col in numeric_cols:
    results_df[col] = results_df[col].apply(lambda x: f"{x:.4f}")

print("\n" + "="*80)
print("COMPLETE RESULTS FOR ALL FILES")
print("="*80)
print(results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")

# Create summary statistics by benchmark type (across all files)
print("\n" + "="*80)
print("AVERAGE PERFORMANCE BY BENCHMARK TYPE (ACROSS ALL FILES)")
print("="*80)


# In[8]:


# Convert back to numeric for aggregation
results_df_numeric = results_df.copy()
for col in numeric_cols:
    results_df_numeric[col] = pd.to_numeric(results_df_numeric[col], errors='coerce')

benchmark_summary = results_df_numeric.groupby('benchmark')[numeric_cols].mean()
print(benchmark_summary.to_string())

# Save benchmark summary to output directory
benchmark_summary.to_csv(benchmark_summary_file)
print(f"\n✓ Benchmark summary saved to: {benchmark_summary_file}")