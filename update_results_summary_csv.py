#!/usr/bin/env python3
"""
Update results_summary.csv files with precision and recall columns.
Calculates mean and std for precision and recall from all_seeds_metrics.csv
"""

import pandas as pd
import numpy as np
import os

def update_results_summary_csv(experiment_num):
    """Add precision and recall columns to results_summary.csv"""
    
    exp_dir = f'results/experiment{experiment_num}'
    seeds_file = os.path.join(exp_dir, 'all_seeds_metrics.csv')
    summary_file = os.path.join(exp_dir, 'results_summary.csv')
    
    # Check if files exist
    if not os.path.exists(seeds_file):
        print(f"❌ {seeds_file} not found")
        return
    
    if not os.path.exists(summary_file):
        print(f"❌ {summary_file} not found")
        return
    
    # Load per-seed metrics
    seeds_df = pd.read_csv(seeds_file)
    
    # Load existing summary
    summary_df = pd.read_csv(summary_file)
    
    # Calculate precision and recall statistics for each k value
    precision_recall_stats = []
    
    for k_val in summary_df['k'].values:
        # Get test data for this k
        test_data = seeds_df[(seeds_df['k'] == k_val) & (seeds_df['split'] == 'test')]
        # Get OOD data for this k
        ood_data = seeds_df[(seeds_df['k'] == k_val) & (seeds_df['split'] == 'ood')]
        
        stats_row = {
            'k': k_val,
            'test_precision_mean': test_data['precision'].mean(),
            'test_precision_std': test_data['precision'].std(ddof=1),
            'test_recall_mean': test_data['recall'].mean(),
            'test_recall_std': test_data['recall'].std(ddof=1),
            'ood_precision_mean': ood_data['precision'].mean(),
            'ood_precision_std': ood_data['precision'].std(ddof=1),
            'ood_recall_mean': ood_data['recall'].mean(),
            'ood_recall_std': ood_data['recall'].std(ddof=1),
        }
        precision_recall_stats.append(stats_row)
    
    # Create dataframe with new columns
    new_cols_df = pd.DataFrame(precision_recall_stats)
    
    # Merge with existing summary (on k column)
    updated_df = pd.merge(summary_df, new_cols_df, on='k')
    
    # Reorder columns to be more logical:
    # k, test_auroc_mean, test_auroc_std, test_f1_mean, test_f1_std, 
    # test_precision_mean, test_precision_std, test_recall_mean, test_recall_std,
    # ood_auroc_mean, ood_auroc_std, ood_f1_mean, ood_f1_std,
    # ood_precision_mean, ood_precision_std, ood_recall_mean, ood_recall_std
    
    column_order = [
        'k',
        'test_auroc_mean', 'test_auroc_std',
        'test_f1_mean', 'test_f1_std',
        'test_precision_mean', 'test_precision_std',
        'test_recall_mean', 'test_recall_std',
        'ood_auroc_mean', 'ood_auroc_std',
        'ood_f1_mean', 'ood_f1_std',
        'ood_precision_mean', 'ood_precision_std',
        'ood_recall_mean', 'ood_recall_std',
    ]
    
    updated_df = updated_df[column_order]
    
    # Save updated CSV
    updated_df.to_csv(summary_file, index=False)
    
    print(f"✅ Updated {summary_file}")
    print(f"   Added columns: test_precision_mean, test_precision_std, test_recall_mean, test_recall_std")
    print(f"                  ood_precision_mean, ood_precision_std, ood_recall_mean, ood_recall_std")

def main():
    print("="*80)
    print("UPDATING RESULTS_SUMMARY.CSV FILES WITH PRECISION AND RECALL")
    print("="*80)
    print()
    
    for exp_num in [1, 2, 3, 4]:
        print(f"\nProcessing Experiment {exp_num}...")
        update_results_summary_csv(exp_num)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nYou can now use these columns in your analysis:")
    print("  - test_precision_mean, test_precision_std")
    print("  - test_recall_mean, test_recall_std")
    print("  - ood_precision_mean, ood_precision_std")
    print("  - ood_recall_mean, ood_recall_std")

if __name__ == "__main__":
    main()
