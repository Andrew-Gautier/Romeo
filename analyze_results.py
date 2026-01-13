"""
Analyze and visualize results from the deduplication experiment.
Creates plots comparing performance across different k values and seeds.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def load_all_results(experiment_dir):
    """Load results from all k values."""
    results = {}
    
    for k in range(1, 13):
        results_path = os.path.join(
            experiment_dir, 
            f'juliet_c_simhash_k={k}', 
            'results', 
            'experiment_results.json'
        )
        
        if os.path.exists(results_path):
            with open(results_path) as f:
                results[k] = json.load(f)
        else:
            print(f"Warning: Results not found for k={k}")
    
    return results


def create_summary_dataframe(results):
    """Create a pandas DataFrame summarizing results across k values."""
    rows = []
    
    for k, data in sorted(results.items()):
        agg = data.get('aggregate', {})
        row = {
            'k': k,
            'test_auroc_mean': agg.get('test_auroc_mean'),
            'test_auroc_std': agg.get('test_auroc_std'),
            'test_f1_mean': agg.get('test_f1_mean'),
            'test_f1_std': agg.get('test_f1_std'),
            'ood_auroc_mean': agg.get('ood_auroc_mean'),
            'ood_auroc_std': agg.get('ood_auroc_std'),
            'ood_f1_mean': agg.get('ood_f1_mean'),
            'ood_f1_std': agg.get('ood_f1_std'),
        }
        
        # Add per-seed results
        for seed, seed_data in data.get('seeds', {}).items():
            row[f'test_auroc_seed{seed}'] = seed_data['test']['auroc']
            row[f'test_f1_seed{seed}'] = seed_data['test']['f1']
            row[f'ood_auroc_seed{seed}'] = seed_data['ood']['auroc']
            row[f'ood_f1_seed{seed}'] = seed_data['ood']['f1']
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_performance_vs_k(df, output_dir):
    """Plot performance metrics vs k value."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    k_values = df['k'].values
    
    # Test AUROC
    ax = axes[0, 0]
    ax.errorbar(k_values, df['test_auroc_mean'], yerr=df['test_auroc_std'], 
                marker='o', capsize=5, label='Test AUROC')
    ax.set_xlabel('k (SimHash threshold)')
    ax.set_ylabel('AUROC')
    ax.set_title('Test Set AUROC vs k')
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Test F1
    ax = axes[0, 1]
    ax.errorbar(k_values, df['test_f1_mean'], yerr=df['test_f1_std'],
                marker='s', capsize=5, color='green', label='Test F1')
    ax.set_xlabel('k (SimHash threshold)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Test Set F1 vs k')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # OOD AUROC
    ax = axes[1, 0]
    ax.errorbar(k_values, df['ood_auroc_mean'], yerr=df['ood_auroc_std'],
                marker='o', capsize=5, color='orange', label='OOD AUROC')
    ax.set_xlabel('k (SimHash threshold)')
    ax.set_ylabel('AUROC')
    ax.set_title('OOD (Devign) AUROC vs k')
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # OOD F1
    ax = axes[1, 1]
    ax.errorbar(k_values, df['ood_f1_mean'], yerr=df['ood_f1_std'],
                marker='s', capsize=5, color='red', label='OOD F1')
    ax.set_xlabel('k (SimHash threshold)')
    ax.set_ylabel('F1 Score')
    ax.set_title('OOD (Devign) F1 vs k')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_vs_k.png'), dpi=150)
    plt.close()
    
    print(f"Saved: performance_vs_k.png")


def plot_test_vs_ood(df, output_dir):
    """Plot test performance vs OOD performance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    k_values = df['k'].values
    
    # AUROC comparison
    ax = axes[0]
    ax.plot(k_values, df['test_auroc_mean'], 'o-', label='Test (In-distribution)', color='blue')
    ax.fill_between(k_values, 
                    df['test_auroc_mean'] - df['test_auroc_std'],
                    df['test_auroc_mean'] + df['test_auroc_std'],
                    alpha=0.2, color='blue')
    ax.plot(k_values, df['ood_auroc_mean'], 's-', label='OOD (Devign)', color='orange')
    ax.fill_between(k_values,
                    df['ood_auroc_mean'] - df['ood_auroc_std'],
                    df['ood_auroc_mean'] + df['ood_auroc_std'],
                    alpha=0.2, color='orange')
    ax.set_xlabel('k (SimHash threshold)')
    ax.set_ylabel('AUROC')
    ax.set_title('Test vs OOD AUROC')
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # F1 comparison
    ax = axes[1]
    ax.plot(k_values, df['test_f1_mean'], 'o-', label='Test (In-distribution)', color='green')
    ax.fill_between(k_values,
                    df['test_f1_mean'] - df['test_f1_std'],
                    df['test_f1_mean'] + df['test_f1_std'],
                    alpha=0.2, color='green')
    ax.plot(k_values, df['ood_f1_mean'], 's-', label='OOD (Devign)', color='red')
    ax.fill_between(k_values,
                    df['ood_f1_mean'] - df['ood_f1_std'],
                    df['ood_f1_mean'] + df['ood_f1_std'],
                    alpha=0.2, color='red')
    ax.set_xlabel('k (SimHash threshold)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Test vs OOD F1')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_vs_ood.png'), dpi=150)
    plt.close()
    
    print(f"Saved: test_vs_ood.png")


def plot_generalization_gap(df, output_dir):
    """Plot the generalization gap (test - OOD performance)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values = df['k'].values
    auroc_gap = df['test_auroc_mean'] - df['ood_auroc_mean']
    f1_gap = df['test_f1_mean'] - df['ood_f1_mean']
    
    width = 0.35
    x = np.arange(len(k_values))
    
    bars1 = ax.bar(x - width/2, auroc_gap, width, label='AUROC Gap', color='steelblue')
    bars2 = ax.bar(x + width/2, f1_gap, width, label='F1 Gap', color='darkorange')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('k (SimHash threshold)')
    ax.set_ylabel('Performance Gap (Test - OOD)')
    ax.set_title('Generalization Gap: Test vs OOD Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generalization_gap.png'), dpi=150)
    plt.close()
    
    print(f"Saved: generalization_gap.png")


def plot_seed_variance(results, output_dir):
    """Plot variance across seeds for each k value."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (metric, title) in enumerate([
        ('test_auroc', 'Test AUROC'),
        ('test_f1', 'Test F1'),
        ('ood_auroc', 'OOD AUROC'),
        ('ood_f1', 'OOD F1'),
    ]):
        ax = axes[idx // 2, idx % 2]
        
        for k in sorted(results.keys()):
            data = results[k]
            seeds = list(data.get('seeds', {}).keys())
            
            if metric.startswith('test'):
                metric_key = metric.replace('test_', '')
                values = [data['seeds'][s]['test'][metric_key] for s in seeds]
            else:
                metric_key = metric.replace('ood_', '')
                values = [data['seeds'][s]['ood'][metric_key] for s in seeds]
            
            ax.scatter([k] * len(values), values, alpha=0.6, s=50)
        
        ax.set_xlabel('k (SimHash threshold)')
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Seed')
        ax.grid(True, alpha=0.3)
        
        if 'auroc' in metric:
            ax.set_ylim(0.5, 1.0)
        else:
            ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seed_variance.png'), dpi=150)
    plt.close()
    
    print(f"Saved: seed_variance.png")


def generate_latex_table(df, output_dir):
    """Generate LaTeX table of results."""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Performance across SimHash deduplication thresholds}
\label{tab:dedup_results}
\begin{tabular}{c|cc|cc}
\toprule
& \multicolumn{2}{c|}{Test (In-Distribution)} & \multicolumn{2}{c}{OOD (Devign)} \\
k & AUROC & F1 & AUROC & F1 \\
\midrule
"""
    
    for _, row in df.iterrows():
        latex += f"{int(row['k'])} & "
        latex += f"{row['test_auroc_mean']:.3f}±{row['test_auroc_std']:.3f} & "
        latex += f"{row['test_f1_mean']:.3f}±{row['test_f1_std']:.3f} & "
        latex += f"{row['ood_auroc_mean']:.3f}±{row['ood_auroc_std']:.3f} & "
        latex += f"{row['ood_f1_mean']:.3f}±{row['ood_f1_std']:.3f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    output_path = os.path.join(output_dir, 'results_table.tex')
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"Saved: results_table.tex")


def load_per_cwe_results(experiment_dir):
    """Load per-CWE CSV results for all k values."""
    cwe_data = {}
    
    for k in range(1, 13):
        csv_path = os.path.join(
            experiment_dir,
            f'juliet_c_simhash_k={k}',
            'results',
            'per_cwe_results.csv'
        )
        
        if os.path.exists(csv_path):
            cwe_data[k] = pd.read_csv(csv_path)
        else:
            print(f"Warning: Per-CWE results not found for k={k}")
    
    return cwe_data


def plot_cwe_performance_heatmap(cwe_data, output_dir):
    """Plot a heatmap of CWE performance across k values."""
    # Collect all CWEs
    all_cwes = set()
    for k_df in cwe_data.values():
        all_cwes.update(k_df['CWE'].values)
    all_cwes = sorted(all_cwes)
    
    # Build matrix
    k_values = sorted(cwe_data.keys())
    matrix = np.full((len(all_cwes), len(k_values)), np.nan)
    
    for j, k in enumerate(k_values):
        if k in cwe_data:
            for _, row in cwe_data[k].iterrows():
                i = all_cwes.index(row['CWE'])
                matrix[i, j] = float(row['AUROC_Mean'])
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, max(10, len(all_cwes) * 0.3)))
    
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)
    
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels(k_values)
    ax.set_yticks(range(len(all_cwes)))
    ax.set_yticklabels(all_cwes, fontsize=8)
    
    ax.set_xlabel('k (SimHash threshold)')
    ax.set_ylabel('CWE')
    ax.set_title('Per-CWE AUROC across k values')
    
    plt.colorbar(im, ax=ax, label='AUROC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cwe_heatmap.png'), dpi=150)
    plt.close()
    
    print(f"Saved: cwe_heatmap.png")


def plot_cwe_performance_distribution(cwe_data, output_dir):
    """Plot distribution of per-CWE AUROC for each k value."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    k_values = sorted(cwe_data.keys())
    positions = []
    data_to_plot = []
    
    for k in k_values:
        if k in cwe_data:
            aurocs = cwe_data[k]['AUROC_Mean'].astype(float).values
            data_to_plot.append(aurocs)
            positions.append(k)
    
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                   patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    
    ax.set_xlabel('k (SimHash threshold)')
    ax.set_ylabel('AUROC')
    ax.set_title('Distribution of Per-CWE AUROC across k values')
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cwe_distribution.png'), dpi=150)
    plt.close()
    
    print(f"Saved: cwe_distribution.png")


def analyze_cwe_stability(cwe_data, output_dir):
    """Analyze which CWEs are most/least stable across k values."""
    # Collect all CWEs
    all_cwes = set()
    for k_df in cwe_data.values():
        all_cwes.update(k_df['CWE'].values)
    
    # Build data for each CWE
    cwe_stats = {}
    for cwe in all_cwes:
        aurocs = []
        for k, k_df in cwe_data.items():
            row = k_df[k_df['CWE'] == cwe]
            if not row.empty:
                aurocs.append(float(row['AUROC_Mean'].values[0]))
        
        if len(aurocs) >= 3:  # Need at least 3 data points
            cwe_stats[cwe] = {
                'mean': np.mean(aurocs),
                'std': np.std(aurocs),
                'min': np.min(aurocs),
                'max': np.max(aurocs),
                'range': np.max(aurocs) - np.min(aurocs),
                'n_k_values': len(aurocs)
            }
    
    # Create DataFrame and save
    stats_df = pd.DataFrame(cwe_stats).T
    stats_df = stats_df.reset_index()
    stats_df.columns = ['CWE', 'Mean_AUROC', 'Std_AUROC', 'Min_AUROC', 'Max_AUROC', 'Range', 'N_K_Values']
    stats_df = stats_df.sort_values('Mean_AUROC', ascending=False)
    
    csv_path = os.path.join(output_dir, 'cwe_stability_analysis.csv')
    stats_df.to_csv(csv_path, index=False)
    
    print(f"Saved: cwe_stability_analysis.csv")
    
    # Print summary
    print("\nPer-CWE Analysis Summary:")
    print(f"  Total CWEs analyzed: {len(stats_df)}")
    print(f"\n  Top 5 CWEs by Mean AUROC:")
    for _, row in stats_df.head(5).iterrows():
        print(f"    {row['CWE']}: {row['Mean_AUROC']:.4f} ± {row['Std_AUROC']:.4f}")
    
    print(f"\n  Bottom 5 CWEs by Mean AUROC:")
    for _, row in stats_df.tail(5).iterrows():
        print(f"    {row['CWE']}: {row['Mean_AUROC']:.4f} ± {row['Std_AUROC']:.4f}")
    
    # Most stable (lowest std)
    stable_df = stats_df.sort_values('Std_AUROC')
    print(f"\n  Most Stable CWEs (lowest std across k):")
    for _, row in stable_df.head(5).iterrows():
        print(f"    {row['CWE']}: std={row['Std_AUROC']:.4f}, mean={row['Mean_AUROC']:.4f}")
    
    # Least stable (highest std)
    print(f"\n  Least Stable CWEs (highest std across k):")
    for _, row in stable_df.tail(5).iterrows():
        print(f"    {row['CWE']}: std={row['Std_AUROC']:.4f}, mean={row['Mean_AUROC']:.4f}")
    
    return stats_df


def main():
    parser = argparse.ArgumentParser(description='Analyze deduplication experiment results')
    parser.add_argument('--experiment-dir', type=str, required=True,
                        help='Path to experiment output directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for analysis (default: experiment_dir/analysis)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, 'analysis')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Loading Results...")
    print("="*60)
    
    results = load_all_results(args.experiment_dir)
    print(f"Loaded results for k values: {sorted(results.keys())}")
    
    print("\n" + "="*60)
    print("Creating Summary DataFrame...")
    print("="*60)
    
    df = create_summary_dataframe(results)
    df.to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    print("Saved: summary.csv")
    
    print("\n" + "="*60)
    print("Generating Plots...")
    print("="*60)
    
    plot_performance_vs_k(df, args.output_dir)
    plot_test_vs_ood(df, args.output_dir)
    plot_generalization_gap(df, args.output_dir)
    plot_seed_variance(results, args.output_dir)
    
    print("\n" + "="*60)
    print("Generating LaTeX Table...")
    print("="*60)
    
    generate_latex_table(df, args.output_dir)
    
    print("\n" + "="*60)
    print("Analyzing Per-CWE Results...")
    print("="*60)
    
    cwe_data = load_per_cwe_results(args.experiment_dir)
    if cwe_data:
        plot_cwe_performance_heatmap(cwe_data, args.output_dir)
        plot_cwe_performance_distribution(cwe_data, args.output_dir)
        analyze_cwe_stability(cwe_data, args.output_dir)
    else:
        print("No per-CWE data found, skipping CWE analysis")
    
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    print("\nTest Set Performance:")
    print(df[['k', 'test_auroc_mean', 'test_auroc_std', 'test_f1_mean', 'test_f1_std']].to_string(index=False))
    
    print("\nOOD Performance:")
    print(df[['k', 'ood_auroc_mean', 'ood_auroc_std', 'ood_f1_mean', 'ood_f1_std']].to_string(index=False))
    
    # Find best k values
    best_test_auroc_k = df.loc[df['test_auroc_mean'].idxmax(), 'k']
    best_ood_auroc_k = df.loc[df['ood_auroc_mean'].idxmax(), 'k']
    
    print(f"\nBest k for Test AUROC: k={int(best_test_auroc_k)}")
    print(f"Best k for OOD AUROC: k={int(best_ood_auroc_k)}")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
