import numpy as np
import pandas as pd
import time
from scipy import stats 

def run_critical_value_simulation(sample_sizes, num_replications, quantiles):
    """
    Runs a Monte Carlo simulation to find critical values for KS, CvM, and AD
    tests for a log-normal distribution with estimated parameters.
    """
    rng = np.random.default_rng(seed=42)
    results = {}

    for n in sample_sizes:
        print(f"Running simulation for n = {n}...")
        
        ks_results = np.zeros(num_replications)
        cvm_results = np.zeros(num_replications)
        ad_results = np.zeros(num_replications)

        for i in range(num_replications):
            # Sample from lognormal(0,1) and estimate parameters
            sample_x = rng.lognormal(mean=0.0, sigma=1.0, size=n)
            sample_y = np.log(sample_x)
            y_mean, y_std = np.mean(sample_y), np.std(sample_y, ddof=1)
            
            # Calculate z values to be used in calculations below
            z = stats.norm.cdf(np.sort(sample_y), loc=y_mean, scale=y_std)
            j = np.arange(1, n + 1)

            # --- Calculate the Test Statistics ---

            # Kolmogorov-Smirnov (KS) - calculated against the estimated CDF
            d_plus = np.max((j / n) - z)
            d_minus = np.max(z - ((j - 1) / n))
            ks_results[i] = np.max([d_plus, d_minus])

            # Cramér-von Mises (CvM) - calculated using the formula specified in introduction
            cvm_results[i] = (1 / (12 * n)) + np.sum((z - (2 * j - 1) / (2 * n))**2)

            # Anderson-Darling (AD) - calculated using the formula specified in introduction
            # A small epsilon is added to prevent log(0) errors with finite-precision numbers.
            epsilon = 1e-15
            ad_results[i] = -n - (1 / n) * np.sum((2 * j - 1) * (np.log(z + epsilon) + np.log(1 - z[::-1] + epsilon)))

        # Determine quantiles from the empirical distributions
        results[n] = {
            'KS': np.quantile(ks_results, quantiles),
            'CvM': np.quantile(cvm_results, quantiles),
            'AD': np.quantile(ad_results, quantiles)
        }

    # Return table results
    df = pd.DataFrame.from_dict(results, orient='index')
    df_list = []
    for test in ['KS', 'CvM', 'AD']:
        temp_df = df[test].apply(pd.Series)
        temp_df.columns = quantiles
        temp_df['Test'] = test
        temp_df = temp_df.set_index('Test', append=True).reorder_levels([1,0])
        df_list.append(temp_df)
    final_df = pd.concat(df_list)
    final_df.index.names = ['Test', 'n']
    return final_df

def print_formatted_tables(critical_value_df):
    """Prints the final tables in a publication-ready format."""
    for test_name in ['KS', 'CvM', 'AD']:
        print(f"\n\n--- Table for {test_name} Test ---")
        print(critical_value_df.loc[test_name].to_string(float_format="%.4f"))


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    
    # --- Start the timer ---
    start_time = time.perf_counter()
   
    # --- Simulation Parameters ---
    sample_sizes_to_run = [10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000]
    replications = 10000
    target_quantiles = [0.75, 0.90, 0.95, 0.99]
    
    # --- Run Simulation and Print Results ---
    critical_value_table = run_critical_value_simulation(sample_sizes_to_run, replications, target_quantiles)
    print_formatted_tables(critical_value_table)
    
    # --- Stop the timer and calculate duration ---
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    print("-" * 50)
    print(f"Total script execution time: {elapsed_time:.2f} seconds")
    print("-" * 50)
    
    
