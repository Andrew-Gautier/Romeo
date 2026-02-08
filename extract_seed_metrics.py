#!/usr/bin/env python3
"""
Extract per-seed metrics from experiment JSON files and create all_seeds_metrics.csv
Similar to the format used in experiment 4.
"""

import json
import csv
import os
from pathlib import Path


def extract_metrics_from_json(experiment_dir, output_file):
    """
    Extract per-seed metrics from k=*.json files in an experiment directory.
    
    Args:
        experiment_dir: Path to experiment results directory
        output_file: Path to output CSV file
    """
    experiment_path = Path(experiment_dir)
    
    # Collect all k=*.json files
    json_files = sorted(experiment_path.glob("k=*.json"), key=lambda x: int(x.stem.split('=')[1]))
    
    if not json_files:
        print(f"No k=*.json files found in {experiment_dir}")
        return
    
    # Prepare CSV data
    rows = []
    
    for json_file in json_files:
        k_value = int(json_file.stem.split('=')[1])
        print(f"Processing {json_file.name}...")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract metrics for each seed
        seeds_data = data.get('seeds', {})
        
        for seed, seed_data in seeds_data.items():
            # Get test metrics
            test_metrics = seed_data.get('test', {})
            if test_metrics:
                rows.append({
                    'k': k_value,
                    'seed': int(seed),
                    'split': 'test',
                    'auroc': test_metrics.get('auroc', 0.0),
                    'f1': test_metrics.get('f1', 0.0),
                    'accuracy': test_metrics.get('accuracy', 0.0),
                    'precision': test_metrics.get('precision', 0.0),
                    'recall': test_metrics.get('recall', 0.0),
                    'loss': test_metrics.get('loss', 0.0)
                })
            
            # Get OOD metrics
            ood_metrics = seed_data.get('ood', {})
            if ood_metrics:
                rows.append({
                    'k': k_value,
                    'seed': int(seed),
                    'split': 'ood',
                    'auroc': ood_metrics.get('auroc', 0.0),
                    'f1': ood_metrics.get('f1', 0.0),
                    'accuracy': ood_metrics.get('accuracy', 0.0),
                    'precision': ood_metrics.get('precision', 0.0),
                    'recall': ood_metrics.get('recall', 0.0),
                    'loss': ood_metrics.get('loss', 0.0)
                })
    
    # Sort by k, then seed, then split
    rows.sort(key=lambda x: (x['k'], x['seed'], x['split']))
    
    # Write to CSV
    if rows:
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['k', 'seed', 'split', 'auroc', 'f1', 'accuracy', 'precision', 'recall', 'loss']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✓ Created {output_file} with {len(rows)} rows")
        print(f"  K values: {sorted(set(r['k'] for r in rows))}")
        print(f"  Seeds: {sorted(set(r['seed'] for r in rows))}")
        print(f"  Splits: {sorted(set(r['split'] for r in rows))}")
    else:
        print(f"✗ No metrics found in {experiment_dir}")


def main():
    """Extract metrics for experiments 1, 2, and 3"""
    experiments = [
        ('results/experiment1', 'results/experiment1/all_seeds_metrics.csv'),
        ('results/experiment2', 'results/experiment2/all_seeds_metrics.csv'),
        ('results/experiment3', 'results/experiment3/all_seeds_metrics.csv'),
    ]
    
    print("="*80)
    print("EXTRACTING PER-SEED METRICS FROM EXPERIMENT JSON FILES")
    print("="*80)
    print()
    
    for exp_dir, output_file in experiments:
        if os.path.exists(exp_dir):
            print(f"\n{'='*80}")
            print(f"Processing {exp_dir}...")
            print(f"{'='*80}")
            extract_metrics_from_json(exp_dir, output_file)
        else:
            print(f"✗ Directory not found: {exp_dir}")
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
