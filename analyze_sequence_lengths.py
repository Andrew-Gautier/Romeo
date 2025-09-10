#!/usr/bin/env python
"""
Analyze sequence lengths in tensor files from Experiment_1 directory.
This script determines the maximum sequence length across all tensor files
in the Experiment_1 folder to help with model configuration.
"""

import os
import torch
import argparse
from tqdm import tqdm
import numpy as np

def analyze_tensor_file(file_path):
    """
    Analyze a single tensor file to determine sequence length information.
    
    Args:
        file_path (str): Path to the tensor file
        
    Returns:
        dict: Dictionary with sequence length statistics
    """
    try:
        # Load the tensor
        tensor = torch.load(file_path)
        
        if not isinstance(tensor, torch.Tensor):
            return {
                'file': file_path,
                'status': 'skipped',
                'reason': 'Not a tensor type'
            }
        
        # Get shape information
        shape = tensor.shape
        
        # For sequence tensors, typically (batch_size, seq_len, ...) or (batch_size, max_lines, tokens_per_line)
        if len(shape) < 2:
            return {
                'file': file_path,
                'status': 'skipped',
                'reason': f'Not a sequence tensor (shape: {shape})'
            }
        
        # For our specific case with shape (batch_size, max_lines, tokens_per_line)
        if len(shape) == 3:
            max_lines = shape[1]
            max_tokens = shape[2]
            
            # Count non-zero tokens to find actual sequence lengths
            # This assumes padding is done with zeros
            if tensor.dtype == torch.int64 or tensor.dtype == torch.int32 or tensor.dtype == torch.int:
                # Convert to numpy for easier analysis
                tensor_np = tensor.numpy()
                
                # Count non-zero tokens per line
                non_zero_counts = (tensor_np > 0).sum(axis=2)
                
                # Get max and average actual sequence length
                max_actual_tokens = non_zero_counts.max()
                avg_actual_tokens = non_zero_counts[non_zero_counts > 0].mean() if (non_zero_counts > 0).any() else 0
                
                # Count non-zero lines (lines with at least one non-zero token)
                non_zero_lines = (non_zero_counts > 0).sum(axis=1)
                max_actual_lines = non_zero_lines.max()
                avg_actual_lines = non_zero_lines.mean()
                
                return {
                    'file': file_path,
                    'status': 'analyzed',
                    'shape': shape,
                    'max_lines': max_lines,
                    'max_tokens_per_line': max_tokens,
                    'max_actual_tokens_per_line': int(max_actual_tokens),
                    'avg_actual_tokens_per_line': float(avg_actual_tokens),
                    'max_actual_lines': int(max_actual_lines),
                    'avg_actual_lines': float(avg_actual_lines),
                }
            else:
                # For non-integer tensors, just report the shape
                return {
                    'file': file_path,
                    'status': 'analyzed',
                    'shape': shape,
                    'max_lines': max_lines,
                    'max_tokens_per_line': max_tokens,
                }
        else:
            # For other tensor shapes, report basic information
            return {
                'file': file_path,
                'status': 'analyzed',
                'shape': shape,
            }
                
    except Exception as e:
        return {
            'file': file_path,
            'status': 'error',
            'error': str(e)
        }

def find_tensor_files(root_dir):
    """
    Find all tensor files (*.pt) in the given directory and its subdirectories.
    
    Args:
        root_dir (str): Root directory to search
        
    Returns:
        list: List of paths to tensor files
    """
    tensor_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pt') and 'sequences' in filename:
                tensor_files.append(os.path.join(dirpath, filename))
                
    return tensor_files

def analyze_experiment_tensors(experiment_dir='tensors/Experiment_1'):
    """
    Analyze all tensors in the experiment directory to find max sequence lengths.
    
    Args:
        experiment_dir (str): Path to the experiment directory
        
    Returns:
        dict: Dictionary with analysis results
    """
    # Find all tensor files
    tensor_files = find_tensor_files(experiment_dir)
    print(f"Found {len(tensor_files)} tensor files to analyze")
    
    # Analyze each tensor file
    results = []
    for file_path in tqdm(tensor_files):
        result = analyze_tensor_file(file_path)
        results.append(result)
    
    # Filter for successfully analyzed files
    analyzed = [r for r in results if r['status'] == 'analyzed']
    
    # Find max dimensions across all analyzed tensors
    max_stats = {}
    if analyzed:
        max_stats = {
            'max_lines_across_all': max([r.get('max_lines', 0) for r in analyzed]),
            'max_tokens_across_all': max([r.get('max_tokens_per_line', 0) for r in analyzed]),
        }
        
        # Only add these stats if there are valid values
        actual_tokens = [r.get('max_actual_tokens_per_line', 0) for r in analyzed if 'max_actual_tokens_per_line' in r]
        if actual_tokens:
            max_stats['max_actual_tokens_across_all'] = max(actual_tokens)
            
        actual_lines = [r.get('max_actual_lines', 0) for r in analyzed if 'max_actual_lines' in r]
        if actual_lines:
            max_stats['max_actual_lines_across_all'] = max(actual_lines)
    
    return {
        'total_files': len(tensor_files),
        'analyzed_files': len(analyzed),
        'skipped_files': len([r for r in results if r['status'] == 'skipped']),
        'error_files': len([r for r in results if r['status'] == 'error']),
        'max_stats': max_stats,
        'detailed_results': results
    }

def print_analysis_results(results):
    """
    Print a formatted summary of the analysis results.
    
    Args:
        results (dict): Analysis results from analyze_experiment_tensors
    """
    print("\n===== TENSOR SEQUENCE LENGTH ANALYSIS =====")
    print(f"Total tensor files: {results['total_files']}")
    print(f"Successfully analyzed: {results['analyzed_files']}")
    print(f"Skipped: {results['skipped_files']}")
    print(f"Errors: {results['error_files']}")
    
    if results['max_stats']:
        print("\n----- MAXIMUM DIMENSIONS -----")
        print(f"Max lines (code lines per function): {results['max_stats']['max_lines_across_all']}")
        print(f"Max tokens per line: {results['max_stats']['max_tokens_across_all']}")
        
        if 'max_actual_tokens_across_all' in results['max_stats']:
            print(f"Max actual tokens in any line (non-padding): {results['max_stats']['max_actual_tokens_across_all']}")
        
        if 'max_actual_lines_across_all' in results['max_stats']:
            print(f"Max actual lines in any function (non-padding): {results['max_stats']['max_actual_lines_across_all']}")
    
    # Print details for each language configuration
    print("\n----- CONFIGURATION BREAKDOWN -----")
    
    # Group by configuration
    configs = {}
    for result in results['detailed_results']:
        if result['status'] != 'analyzed' or 'shape' not in result:
            continue
            
        # Extract configuration from path
        path_parts = result['file'].split('/')
        if 'language_matrix_42' in path_parts:
            idx = path_parts.index('language_matrix_42')
            if idx + 1 < len(path_parts):
                config = path_parts[idx + 1]
                if config not in configs:
                    configs[config] = []
                configs[config].append(result)
    
    # Print stats for each configuration
    for config, results_list in configs.items():
        print(f"\nConfiguration: {config}")
        
        # Extract stats safely
        line_values = [r.get('max_lines', 0) for r in results_list if 'max_lines' in r]
        token_values = [r.get('max_tokens_per_line', 0) for r in results_list if 'max_tokens_per_line' in r]
        actual_token_values = [r.get('max_actual_tokens_per_line', 0) for r in results_list if 'max_actual_tokens_per_line' in r]
        actual_line_values = [r.get('max_actual_lines', 0) for r in results_list if 'max_actual_lines' in r]
        
        if line_values:
            print(f"  Max lines: {max(line_values)}")
        if token_values:
            print(f"  Max tokens per line: {max(token_values)}")
        if actual_token_values:
            print(f"  Max actual tokens per line: {max(actual_token_values)}")
        if actual_line_values:
            print(f"  Max actual lines: {max(actual_line_values)}")

def main():
    parser = argparse.ArgumentParser(description='Analyze tensor sequence lengths in Experiment_1')
    parser.add_argument('--dir', type=str, default='tensors/Experiment_1',
                        help='Directory containing the tensor files (default: tensors/Experiment_1)')
    
    args = parser.parse_args()
    
    print(f"Analyzing tensors in: {args.dir}")
    results = analyze_experiment_tensors(args.dir)
    print_analysis_results(results)
    
    # Optionally save detailed results to a file
    with open('tensor_analysis_results.txt', 'w') as f:
        f.write("===== TENSOR SEQUENCE LENGTH ANALYSIS =====\n")
        f.write(f"Total tensor files: {results['total_files']}\n")
        f.write(f"Successfully analyzed: {results['analyzed_files']}\n")
        f.write(f"Skipped: {results['skipped_files']}\n")
        f.write(f"Errors: {results['error_files']}\n")
        
        if results['max_stats']:
            f.write("\n----- MAXIMUM DIMENSIONS -----\n")
            f.write(f"Max lines: {results['max_stats']['max_lines_across_all']}\n")
            f.write(f"Max tokens per line: {results['max_stats']['max_tokens_across_all']}\n")
            
            if 'max_actual_tokens_across_all' in results['max_stats']:
                f.write(f"Max actual tokens per line: {results['max_stats']['max_actual_tokens_across_all']}\n")
            
            if 'max_actual_lines_across_all' in results['max_stats']:
                f.write(f"Max actual lines: {results['max_stats']['max_actual_lines_across_all']}\n")
        
        # Write detailed results for each file
        f.write("\n----- DETAILED RESULTS -----\n")
        for result in results['detailed_results']:
            if result['status'] == 'analyzed' and 'shape' in result:
                f.write(f"File: {result['file']}\n")
                f.write(f"  Shape: {result['shape']}\n")
                if 'max_lines' in result:
                    f.write(f"  Max lines: {result['max_lines']}\n")
                if 'max_tokens_per_line' in result:
                    f.write(f"  Max tokens per line: {result['max_tokens_per_line']}\n")
                if 'max_actual_tokens_per_line' in result:
                    f.write(f"  Max actual tokens per line: {result['max_actual_tokens_per_line']}\n")
                if 'avg_actual_tokens_per_line' in result:
                    f.write(f"  Avg actual tokens per line: {result['avg_actual_tokens_per_line']:.2f}\n")
                if 'max_actual_lines' in result:
                    f.write(f"  Max actual lines: {result['max_actual_lines']}\n")
                if 'avg_actual_lines' in result:
                    f.write(f"  Avg actual lines: {result['avg_actual_lines']:.2f}\n")
                f.write("\n")
    
    print(f"\nDetailed results saved to tensor_analysis_results.txt")

if __name__ == "__main__":
    main()
