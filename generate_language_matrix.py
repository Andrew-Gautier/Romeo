"""
Generate a testing matrix of tensors for cross-language vulnerability detection.
This script creates multiple tensor sets with different configurations:
1. C-only training, testing on C
2. Java-only training, testing on Java
3. C-only training, testing on Java (cross-language)
4. Java-only training, testing on C (cross-language)
5. Combined C+Java training, testing on both
"""

import os
import sys
import preprocessing
from transformers import AutoTokenizer

def generate_language_matrix(base_seed=42, sample_limit=100000, balanced=True):
    """
    Generate multiple tensor sets for cross-language testing.
    
    Args:
        base_seed (int): Base random seed to use (will be incremented for each config)
        sample_limit (int): Number of samples per class to use
        balanced (bool): Whether to balance vulnerability classes
    """
    print(f"=== Generating language testing matrix with {sample_limit} samples per class ===")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    
    # Configuration matrix
    configs = [
        {"name": "c_only", "desc": "C-only training and testing"},
        {"name": "java_only", "desc": "Java-only training and testing"},
        {"name": "c_to_java", "desc": "C training, Java testing (cross-language)"},
        {"name": "java_to_c", "desc": "Java training, C testing (cross-language)"},
        {"name": "combined", "desc": "Combined C+Java training and testing"}
    ]
    
    # Base output directory
    matrix_dir = f"tensors/Experiment_1/language_matrix_{base_seed}"
    os.makedirs(matrix_dir, exist_ok=True)
    
    # Generate each configuration
    for i, config in enumerate(configs):
        seed = base_seed + i
        config_dir = f"{matrix_dir}/{config['name']}_seed{seed}"
        
        print(f"\n\n{'='*80}")
        print(f"Generating configuration: {config['desc']} (seed: {seed})")
        print(f"{'='*80}")
        
        # Run preprocessing with the specific configuration
        stats = preprocessing.preprocess_data(
            c_db_path='c_10+.db',
            java_db_path='java_10+.db',
            tokenizer=tokenizer,
            limit_per_class=sample_limit,
            balance_classes=balanced,
            seed_id=seed,
            output_dir=config_dir
        )
        
        # Save a summary of this configuration
        summary_file = f"{config_dir}/Experiment_1.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Configuration: {config['desc']}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Sample limit per class: {sample_limit}\n")
            f.write(f"Balanced classes: {balanced}\n\n")
            
            f.write("=== Statistics ===\n")
            f.write(f"C samples: {stats['c_stats']['vulnerable']} vulnerable, {stats['c_stats']['non_vulnerable']} non-vulnerable\n")
            f.write(f"Java samples: {stats['java_stats']['vulnerable']} vulnerable, {stats['java_stats']['non_vulnerable']} non-vulnerable\n")
            
            f.write("\nSplit statistics:\n")
            for lang in ['c', 'java']:
                if lang in stats['language_splits']:
                    lang_stats = stats['language_splits'][lang]
                    f.write(f"\n{lang.upper()}:\n")
                    f.write(f"  Train: {lang_stats['train_positive']}/{lang_stats['train_total']} positive\n")
                    f.write(f"  Validation: {lang_stats['val_positive']}/{lang_stats['val_total']} positive\n")
                    f.write(f"  Test: {lang_stats['test_positive']}/{lang_stats['test_total']} positive\n")
        
        print(f"Configuration {config['name']} completed. Summary saved to {summary_file}")
    
    print("\nLanguage testing matrix generation complete!")
    print(f"All tensor sets saved in: {matrix_dir}")

if __name__ == "__main__":
    # Get command line arguments if provided
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100000

    generate_language_matrix(base_seed=seed, sample_limit=limit)
