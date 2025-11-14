#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import fft
import glob
import os


# ### Pre Processing CAMELS-SPAT

# In[2]:


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
    
    # Drop NaN values
    flow = flow.dropna()
    
    # 1. Calculate seasonal component (mean for each calendar day)
    seasonal = flow.groupby([flow.index.month, flow.index.day]).transform('mean')
    
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
    return 1 - (numerator / denominator)


def calculate_variance_component_nses(obs_decomp, sim_decomp):
    """
    Calculate NSE values for each variance component and verify Equation 3.
    
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
        - 'is_highly_seasonal': Boolean, True if seasonal variance fraction > 0.5
        - 'seasonal_variance_fraction': Fraction of variance that is seasonal
    """
    
    # Calculate NSE for each component (Equation 2)
    nse_seasonal = calculate_component_nse(obs_decomp['seasonal'], sim_decomp['seasonal'])
    nse_interannual = calculate_component_nse(obs_decomp['interannual'], sim_decomp['interannual'])
    nse_irregular = calculate_component_nse(obs_decomp['irregular'], sim_decomp['irregular'])
    nse_overall = calculate_component_nse(obs_decomp['original'], sim_decomp['original'])
    
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
        'nse_overall': nse_overall,
        'nse_seasonal': nse_seasonal,
        'nse_interannual': nse_interannual,
        'nse_irregular': nse_irregular,
        'nse_weighted_sum': nse_weighted_sum,
        'nse_difference': nse_difference,
        'is_highly_seasonal': is_highly_seasonal,
        'seasonal_variance_fraction': var_fracs['seasonal'],
        'observed_variance_fractions': var_fracs
    }
    
    return results


# ### Inputs

# In[4]:


# Directory containing CSV files
csv_directory = '../nse_skill_score/camels-spat/bm_outputs_full/'  # Change this to your directory


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
        
        # Decompose observed flow
        print(f"\nDecomposing observed data...")
        qobs_decomp = decompose_streamflow(flow, 'q_obs')
        
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
            print(f"  Decomposing {bm_col}...")
            benchmark_decomps[bm_col] = decompose_streamflow(flow, bm_col)
        
        # Calculate NSE results for each benchmark
        nse_results_all = {}
        for bm_col in benchmark_cols:
            nse_results_all[bm_col] = calculate_variance_component_nses(
                qobs_decomp, 
                benchmark_decomps[bm_col]
            )
        
        # Create summary for this CSV
        for bm_col, results in nse_results_all.items():
            all_results.append({
                'csv_file': csv_name,
                'benchmark': bm_col,
                'nse_overall': results['nse_overall'],
                'nse_seasonal': results['nse_seasonal'],
                'nse_interannual': results['nse_interannual'],
                'nse_irregular': results['nse_irregular'],
                'obs_seasonal_var': results['observed_variance_fractions']['seasonal'],
                'obs_interannual_var': results['observed_variance_fractions']['interannual'],
                'obs_irregular_var': results['observed_variance_fractions']['irregular'],
                'is_highly_seasonal': results['is_highly_seasonal']
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
                'obs_seasonal_var', 'obs_interannual_var', 'obs_irregular_var']

for col in numeric_cols:
    results_df[col] = results_df[col].apply(lambda x: f"{x:.4f}")

print("\n" + "="*80)
print("COMPLETE RESULTS FOR ALL FILES")
print("="*80)
print(results_df.to_string(index=False))

# Save results to CSV
output_file = 'benchmark_nse_results_all_files.csv'
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
    results_df_numeric[col] = pd.to_numeric(results_df_numeric[col])

benchmark_summary = results_df_numeric.groupby('benchmark')[numeric_cols].mean()
print(benchmark_summary.to_string())

# Save benchmark summary
benchmark_summary.to_csv('benchmark_summary_across_files.csv')
print(f"\n✓ Benchmark summary saved to: benchmark_summary_across_files.csv")

