#!/usr/bin/env python3
"""
Demonstrate statistical tests using the all_seeds_metrics.csv files.
This script shows how to perform t-tests and other statistical comparisons.
"""

import pandas as pd
import numpy as np
from scipy import stats


def load_experiment_data(experiment_num):
    """Load all_seeds_metrics.csv for a given experiment"""
    filepath = f'results/experiment{experiment_num}/all_seeds_metrics.csv'
    df = pd.read_csv(filepath)
    print(f"✓ Loaded Experiment {experiment_num}: {len(df)} rows")
    return df


def perform_ttest_example():
    """
    Example: Perform t-tests comparing OOD AUROC between experiments.
    This demonstrates how to use the extracted per-seed data for statistical analysis.
    """
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS EXAMPLE: T-TESTS ON OOD AUROC")
    print("="*80)
    
    # Load all experiments
    exp1_df = load_experiment_data(1)
    exp2_df = load_experiment_data(2)
    exp3_df = load_experiment_data(3)
    exp4_df = load_experiment_data(4)
    
    # Example 1: Compare Experiment 1 vs Experiment 2 at k=1, OOD split
    print("\n" + "-"*80)
    print("Example 1: Exp 1 vs Exp 2 (OOD AUROC at k=1)")
    print("-"*80)
    
    exp1_k1_ood = exp1_df[(exp1_df['k'] == 1) & (exp1_df['split'] == 'ood')]['auroc'].values
    exp2_k1_ood = exp2_df[(exp2_df['k'] == 1) & (exp2_df['split'] == 'ood')]['auroc'].values
    
    print(f"Exp 1 (aiXcoder): n={len(exp1_k1_ood)}, mean={exp1_k1_ood.mean():.4f}, std={exp1_k1_ood.std():.4f}")
    print(f"Exp 2 (DeepSeek): n={len(exp2_k1_ood)}, mean={exp2_k1_ood.mean():.4f}, std={exp2_k1_ood.std():.4f}")
    
    t_stat, p_value = stats.ttest_ind(exp1_k1_ood, exp2_k1_ood)
    print(f"\nTwo-sample t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")
    
    # Example 2: Compare all Juliet C experiments (1-3) vs Devign (4) at k=1
    print("\n" + "-"*80)
    print("Example 2: Juliet C (Exp 1-3) vs Devign (Exp 4) - OOD AUROC at k=1")
    print("-"*80)
    
    exp3_k1_ood = exp3_df[(exp3_df['k'] == 1) & (exp3_df['split'] == 'ood')]['auroc'].values
    exp4_k1_ood = exp4_df[(exp4_df['k'] == 1) & (exp4_df['split'] == 'ood')]['auroc'].values
    
    # Pool all Juliet C experiments
    juliet_ood = np.concatenate([exp1_k1_ood, exp2_k1_ood, exp3_k1_ood])
    
    print(f"Juliet C (pooled): n={len(juliet_ood)}, mean={juliet_ood.mean():.4f}, std={juliet_ood.std():.4f}")
    print(f"Devign (Exp 4):    n={len(exp4_k1_ood)}, mean={exp4_k1_ood.mean():.4f}, std={exp4_k1_ood.std():.4f}")
    
    t_stat, p_value = stats.ttest_ind(juliet_ood, exp4_k1_ood)
    print(f"\nTwo-sample t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")
    print(f"  Significant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")
    
    # Example 3: Test effect of deduplication (k=1 vs k=12) within Experiment 1
    print("\n" + "-"*80)
    print("Example 3: Effect of Deduplication in Exp 1 (k=1 vs k=12, OOD AUROC)")
    print("-"*80)
    
    exp1_k12_ood = exp1_df[(exp1_df['k'] == 12) & (exp1_df['split'] == 'ood')]['auroc'].values
    
    print(f"k=1 (100% data):  n={len(exp1_k1_ood)}, mean={exp1_k1_ood.mean():.4f}, std={exp1_k1_ood.std():.4f}")
    print(f"k=12 (1.9% data): n={len(exp1_k12_ood)}, mean={exp1_k12_ood.mean():.4f}, std={exp1_k12_ood.std():.4f}")
    
    # Paired t-test (same seeds)
    t_stat, p_value = stats.ttest_rel(exp1_k1_ood, exp1_k12_ood)
    print(f"\nPaired t-test (same seeds across k values):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")
    
    # Example 4: Summary statistics for all experiments at k=1
    print("\n" + "-"*80)
    print("Example 4: Summary Statistics - All Experiments at k=1 (OOD AUROC)")
    print("-"*80)
    
    summary_data = {
        'Experiment': ['Exp 1 (aiXcoder, Juliet C)', 'Exp 2 (DeepSeek, Juliet C)', 
                       'Exp 3 (CodeLlama, Juliet C)', 'Exp 4 (DeepSeek, Devign)'],
        'Mean': [exp1_k1_ood.mean(), exp2_k1_ood.mean(), exp3_k1_ood.mean(), exp4_k1_ood.mean()],
        'Std': [exp1_k1_ood.std(), exp2_k1_ood.std(), exp3_k1_ood.std(), exp4_k1_ood.std()],
        'Min': [exp1_k1_ood.min(), exp2_k1_ood.min(), exp3_k1_ood.min(), exp4_k1_ood.min()],
        'Max': [exp1_k1_ood.max(), exp2_k1_ood.max(), exp3_k1_ood.max(), exp4_k1_ood.max()],
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # ANOVA to test if there are differences across all 4 experiments
    print("\n" + "-"*80)
    print("One-way ANOVA: Testing differences across all 4 experiments")
    print("-"*80)
    
    f_stat, p_value = stats.f_oneway(exp1_k1_ood, exp2_k1_ood, exp3_k1_ood, exp4_k1_ood)
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")
    print(f"  Significant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\n💡 You now have per-seed data for all experiments and can perform:")
    print("   • T-tests (paired or independent)")
    print("   • ANOVA / Kruskal-Wallis tests")
    print("   • Effect size calculations (Cohen's d, etc.)")
    print("   • Bootstrap confidence intervals")
    print("   • Any other statistical comparisons needed for your research!")


if __name__ == '__main__':
    perform_ttest_example()
