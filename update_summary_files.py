#!/usr/bin/env python3
"""
Update experiment_summary.txt files with precision and recall statistics.
Calculates mean ± 95% confidence intervals for precision and recall from all_seeds_metrics.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

def calculate_ci_95(values):
    """Calculate 95% confidence interval using t-distribution"""
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)
    # t-value for 95% CI with n-1 degrees of freedom
    t_val = stats.t.ppf(0.975, n - 1)
    margin = t_val * se
    return mean, margin

def update_experiment_summary(experiment_num):
    """Update experiment_summary.txt with precision and recall metrics"""
    
    exp_dir = f'results/experiment{experiment_num}'
    seeds_file = os.path.join(exp_dir, 'all_seeds_metrics.csv')
    summary_file = os.path.join(exp_dir, 'experiment_summary.txt')
    
    # Check if files exist
    if not os.path.exists(seeds_file):
        print(f"❌ {seeds_file} not found")
        return
    
    if not os.path.exists(summary_file):
        print(f"❌ {summary_file} not found")
        return
    
    # Load per-seed metrics
    df = pd.read_csv(seeds_file)
    
    # Read existing summary file
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    
    # Find where to insert new metrics (after F1 lines)
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        
        # Check if this is an F1 line (OOD F1 specifically, so we add after it)
        if line.strip().startswith('OOD F1:'):
            # Extract k value from the section header
            # Look backwards to find the k= line
            k_value = None
            for j in range(i-1, max(0, i-10), -1):
                if lines[j].strip().startswith('--- k='):
                    k_value = int(lines[j].strip().split('=')[1].split()[0])
                    break
            
            if k_value is not None:
                # Calculate precision and recall for this k value
                test_data = df[(df['k'] == k_value) & (df['split'] == 'test')]
                ood_data = df[(df['k'] == k_value) & (df['split'] == 'ood')]
                
                # Test precision
                test_prec_mean, test_prec_ci = calculate_ci_95(test_data['precision'].values)
                # Test recall
                test_rec_mean, test_rec_ci = calculate_ci_95(test_data['recall'].values)
                # OOD precision
                ood_prec_mean, ood_prec_ci = calculate_ci_95(ood_data['precision'].values)
                # OOD recall
                ood_rec_mean, ood_rec_ci = calculate_ci_95(ood_data['recall'].values)
                
                # Add precision and recall lines
                new_lines.append(f"  Test Precision: {test_prec_mean:.4f} ± {test_prec_ci:.4f}\n")
                new_lines.append(f"  Test Recall: {test_rec_mean:.4f} ± {test_rec_ci:.4f}\n")
                new_lines.append(f"  OOD Precision: {ood_prec_mean:.4f} ± {ood_prec_ci:.4f}\n")
                new_lines.append(f"  OOD Recall: {ood_rec_mean:.4f} ± {ood_rec_ci:.4f}\n")
        
        i += 1
    
    # Write updated summary file
    with open(summary_file, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✅ Updated {summary_file}")

def main():
    print("="*80)
    print("UPDATING EXPERIMENT SUMMARY FILES WITH PRECISION AND RECALL")
    print("="*80)
    print()
    
    for exp_num in [1, 2, 3, 4]:
        print(f"\nProcessing Experiment {exp_num}...")
        update_experiment_summary(exp_num)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
