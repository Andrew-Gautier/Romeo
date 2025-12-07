import os
import sys
import datetime
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from preprocessing import (
    load_data_from_db, 
    create_labels, 
    tokenize_and_pad,
    pad_sequence,
    MIN_TOKENS,
    MAX_TOKENS,
    LANGUAGES,
    detect_language_from_id
)
import sqlite3

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("aiXcoder/aixcoder-7b-base")

# Configuration
pretraining_sample_size = 10000
eval_sample_size = 10000
random_state = 42

# Datasets for pretraining
juliet_c_db = 'datasets/juliet_c.db'
juliet_java_db = 'datasets/juliet_java.db'
juliet_csharp_db = 'datasets/juliet_csharp.db'

# Datasets for evaluation
devign_c_db = 'datasets/devign.db'
python_db = 'datasets/bugsinpy.db'

def load_evaluation_data_from_db(db_path, tokenizer, limit_per_class=None, balance_classes=False, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS):
    """
    Load function data from evaluation databases where vuln field contains binary labels (0 or 1).
    
    Args:
        db_path (str): Path to the SQLite database
        tokenizer: Tokenizer used to compute token counts
        limit_per_class (int, optional): Maximum number of samples per class
        balance_classes (bool): Whether to balance vulnerable and non-vulnerable classes
        min_tokens (int): Minimum number of tokens per function
        max_tokens (int): Maximum number of tokens per function
        
    Returns:
        tuple: (all_functions, vulnerable_count, non_vulnerable_count)
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Fetch all candidates - vuln field contains '1' for vulnerable, '0' for non-vulnerable
        cursor.execute("SELECT grp, id, start, end, vuln, code FROM funcs WHERE vuln = '1'")
        vuln_candidates = cursor.fetchall()
        
        cursor.execute("SELECT grp, id, start, end, vuln, code FROM funcs WHERE vuln = '0'")
        non_vuln_candidates = cursor.fetchall()

        def filter_by_tokens(rows):
            out = []
            for row in rows:
                group, file_id, start, end, vuln, code = row
                tokens = tokenizer.encode(code, add_special_tokens=False)
                if min_tokens <= len(tokens) <= max_tokens:
                    out.append((group, file_id, start, end, vuln, code, tokens))
            return out

        vulnerable_funcs = filter_by_tokens(vuln_candidates)
        non_vulnerable_funcs = filter_by_tokens(non_vuln_candidates)
        
        # Apply balancing if needed
        if limit_per_class is not None:
            vuln_limit = min(len(vulnerable_funcs), limit_per_class)
            non_vuln_limit = min(len(non_vulnerable_funcs), limit_per_class)
            if balance_classes:
                min_size = min(vuln_limit, non_vuln_limit)
                vulnerable_funcs = vulnerable_funcs[:min_size]
                non_vulnerable_funcs = non_vulnerable_funcs[:min_size]
            else:
                vulnerable_funcs = vulnerable_funcs[:vuln_limit]
                non_vulnerable_funcs = non_vulnerable_funcs[:non_vuln_limit]

        conn.close()
        # Combine the data (include tokens)
        all_funcs = vulnerable_funcs + non_vulnerable_funcs
        return all_funcs, len(vulnerable_funcs), len(non_vulnerable_funcs)
    except Exception as e:
        print(f"Error in load_evaluation_data_from_db: {str(e)}")
        return [], 0, 0

def create_pretraining_tensors(db_paths, tokenizer, sample_size_per_class, output_dir, timestamp, balance_classes=True):
    """
    Create training and validation tensors from pretraining datasets with 70/30 split.
    
    Args:
        db_paths (dict): Dictionary mapping language names to database paths
        tokenizer: The tokenizer to use
        sample_size_per_class (int): Number of samples per class (vulnerable/non-vulnerable)
        output_dir (str): Base output directory
        timestamp (str): Timestamp for the run
        balance_classes (bool): Whether to balance classes
    
    Returns:
        dict: Statistics about the preprocessing operation
    """
    print("\n" + "="*80)
    print("PHASE 1: Creating Pretraining Tensors")
    print("="*80)
    
    all_data = []
    all_labels = []
    all_langs = []
    stats = {}
    
    # Load data from each database
    for lang, db_path in db_paths.items():
        print(f"\nLoading {lang} data from {db_path}...")
        
        if not os.path.exists(db_path):
            print(f"Warning: Database {db_path} not found. Skipping {lang}.")
            continue
            
        funcs, vuln_count, non_vuln_count = load_data_from_db(
            db_path, 
            tokenizer=tokenizer,
            limit_per_class=sample_size_per_class,
            balance_classes=balance_classes,
            min_tokens=MIN_TOKENS,
            max_tokens=MAX_TOKENS
        )
        
        if not funcs:
            print(f"No data loaded from {db_path}")
            continue
        
        # Create labels for this language
        data, labels, vuln_cnt, lang_ids = create_labels(funcs)
        
        all_data.extend(data)
        all_labels.extend(labels)
        all_langs.extend(lang_ids)
        
        stats[lang] = {
            'vulnerable': vuln_count,
            'non_vulnerable': non_vuln_count,
            'total': len(data)
        }
        
        print(f"{lang}: {vuln_count} vulnerable, {non_vuln_count} non-vulnerable")
    
    if not all_data:
        print("No data loaded. Exiting.")
        return None
    
    print(f"\nTotal samples loaded: {len(all_data)}")
    
    # Split into train (70%) and validation (30%)
    train_data, val_data, train_labels, val_labels, train_langs, val_langs = train_test_split(
        all_data, all_labels, all_langs,
        test_size=0.3,
        random_state=random_state,
        stratify=[int(l.item()) for l in all_labels]
    )
    
    print(f"Train: {len(train_data)}, Validation: {len(val_data)} samples")
    
    # Tokenize
    print("\nTokenizing data...")
    train_sequences = tokenize_and_pad(train_data, tokenizer, max_seq_length=MAX_TOKENS)
    val_sequences = tokenize_and_pad(val_data, tokenizer, max_seq_length=MAX_TOKENS)
    
    train_labels_tensor = torch.stack(train_labels)
    val_labels_tensor = torch.stack(val_labels)
    
    # Save tensors with directory structure: tensors/{timestamp}/pretraining/{language}/train
    base_dir = os.path.join(output_dir, timestamp, 'pretraining')
    
    # Save by language
    for lang in set(all_langs):
        # Train data
        train_lang_indices = [i for i, l in enumerate(train_langs) if l == lang]
        if train_lang_indices:
            lang_train_dir = os.path.join(base_dir, lang, 'train')
            os.makedirs(lang_train_dir, exist_ok=True)
            
            torch.save(train_sequences[train_lang_indices], f'{lang_train_dir}/sequences.pt')
            torch.save(train_labels_tensor[train_lang_indices], f'{lang_train_dir}/labels.pt')
            print(f"Saved {lang} training data: {len(train_lang_indices)} samples")
        
        # Validation data
        val_lang_indices = [i for i, l in enumerate(val_langs) if l == lang]
        if val_lang_indices:
            lang_val_dir = os.path.join(base_dir, lang, 'validation')
            os.makedirs(lang_val_dir, exist_ok=True)
            
            torch.save(val_sequences[val_lang_indices], f'{lang_val_dir}/sequences.pt')
            torch.save(val_labels_tensor[val_lang_indices], f'{lang_val_dir}/labels.pt')
            print(f"Saved {lang} validation data: {len(val_lang_indices)} samples")
    
    # Save combined data
    combined_dir = os.path.join(base_dir, 'combined')
    os.makedirs(os.path.join(combined_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(combined_dir, 'validation'), exist_ok=True)
    
    torch.save(train_sequences, f'{combined_dir}/train/sequences.pt')
    torch.save(train_labels_tensor, f'{combined_dir}/train/labels.pt')
    torch.save(train_langs, f'{combined_dir}/train/languages.pt')
    
    torch.save(val_sequences, f'{combined_dir}/validation/sequences.pt')
    torch.save(val_labels_tensor, f'{combined_dir}/validation/labels.pt')
    torch.save(val_langs, f'{combined_dir}/validation/languages.pt')
    
    print(f"\nSaved combined pretraining data to {combined_dir}")
    
    return {
        'stats': stats,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'timestamp': timestamp
    }

def create_evaluation_tensors(db_paths, tokenizer, sample_size_per_class, output_dir, timestamp, balance_classes=True):
    """
    Create train/val/test tensors from evaluation datasets with 60/20/20 split.
    
    Args:
        db_paths (dict): Dictionary mapping language names to database paths
        tokenizer: The tokenizer to use
        sample_size_per_class (int): Number of samples per class
        output_dir (str): Base output directory
        timestamp (str): Timestamp for the run
        balance_classes (bool): Whether to balance classes
    
    Returns:
        dict: Statistics about the preprocessing operation
    """
    print("\n" + "="*80)
    print("PHASE 2: Creating Evaluation Tensors")
    print("="*80)
    
    all_data = []
    all_labels = []
    all_langs = []
    stats = {}
    
    # Load data from each database
    for lang, db_path in db_paths.items():
        print(f"\nLoading {lang} data from {db_path}...")
        
        if not os.path.exists(db_path):
            print(f"Warning: Database {db_path} not found. Skipping {lang}.")
            continue
            
        # Use the evaluation-specific loader that handles binary labels
        funcs, vuln_count, non_vuln_count = load_evaluation_data_from_db(
            db_path,
            tokenizer=tokenizer,
            limit_per_class=sample_size_per_class,
            balance_classes=balance_classes,
            min_tokens=MIN_TOKENS,
            max_tokens=MAX_TOKENS
        )
        
        if not funcs:
            print(f"No data loaded from {db_path}")
            continue
        
        # Create labels
        data, labels, vuln_cnt, lang_ids = create_labels(funcs)
        
        all_data.extend(data)
        all_labels.extend(labels)
        all_langs.extend(lang_ids)
        
        stats[lang] = {
            'vulnerable': vuln_count,
            'non_vulnerable': non_vuln_count,
            'total': len(data)
        }
        
        print(f"{lang}: {vuln_count} vulnerable, {non_vuln_count} non-vulnerable")
    
    if not all_data:
        print("No data loaded. Exiting.")
        return None
    
    print(f"\nTotal samples loaded: {len(all_data)}")
    
    # Split into train (60%), validation (20%), test (20%)
    train_data, temp_data, train_labels, temp_labels, train_langs, temp_langs = train_test_split(
        all_data, all_labels, all_langs,
        test_size=0.4,
        random_state=random_state,
        stratify=[int(l.item()) for l in all_labels]
    )
    
    val_data, test_data, val_labels, test_labels, val_langs, test_langs = train_test_split(
        temp_data, temp_labels, temp_langs,
        test_size=0.5,
        random_state=random_state,
        stratify=[int(l.item()) for l in temp_labels]
    )
    
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)} samples")
    
    # Tokenize
    print("\nTokenizing data...")
    train_sequences = tokenize_and_pad(train_data, tokenizer, max_seq_length=MAX_TOKENS)
    val_sequences = tokenize_and_pad(val_data, tokenizer, max_seq_length=MAX_TOKENS)
    test_sequences = tokenize_and_pad(test_data, tokenizer, max_seq_length=MAX_TOKENS)
    
    train_labels_tensor = torch.stack(train_labels)
    val_labels_tensor = torch.stack(val_labels)
    test_labels_tensor = torch.stack(test_labels)
    
    # Save tensors with directory structure: tensors/{timestamp}/evaluation/{language}/train
    base_dir = os.path.join(output_dir, timestamp, 'evaluation')
    
    # Save by language
    for lang in set(all_langs):
        # Train data
        train_lang_indices = [i for i, l in enumerate(train_langs) if l == lang]
        if train_lang_indices:
            lang_train_dir = os.path.join(base_dir, lang, 'train')
            os.makedirs(lang_train_dir, exist_ok=True)
            
            torch.save(train_sequences[train_lang_indices], f'{lang_train_dir}/sequences.pt')
            torch.save(train_labels_tensor[train_lang_indices], f'{lang_train_dir}/labels.pt')
            print(f"Saved {lang} training data: {len(train_lang_indices)} samples")
        
        # Validation data
        val_lang_indices = [i for i, l in enumerate(val_langs) if l == lang]
        if val_lang_indices:
            lang_val_dir = os.path.join(base_dir, lang, 'validation')
            os.makedirs(lang_val_dir, exist_ok=True)
            
            torch.save(val_sequences[val_lang_indices], f'{lang_val_dir}/sequences.pt')
            torch.save(val_labels_tensor[val_lang_indices], f'{lang_val_dir}/labels.pt')
            print(f"Saved {lang} validation data: {len(val_lang_indices)} samples")
        
        # Test data
        test_lang_indices = [i for i, l in enumerate(test_langs) if l == lang]
        if test_lang_indices:
            lang_test_dir = os.path.join(base_dir, lang, 'test')
            os.makedirs(lang_test_dir, exist_ok=True)
            
            torch.save(test_sequences[test_lang_indices], f'{lang_test_dir}/sequences.pt')
            torch.save(test_labels_tensor[test_lang_indices], f'{lang_test_dir}/labels.pt')
            print(f"Saved {lang} test data: {len(test_lang_indices)} samples")
    
    # Save combined data
    combined_dir = os.path.join(base_dir, 'combined')
    os.makedirs(os.path.join(combined_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(combined_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(combined_dir, 'test'), exist_ok=True)
    
    torch.save(train_sequences, f'{combined_dir}/train/sequences.pt')
    torch.save(train_labels_tensor, f'{combined_dir}/train/labels.pt')
    torch.save(train_langs, f'{combined_dir}/train/languages.pt')
    
    torch.save(val_sequences, f'{combined_dir}/validation/sequences.pt')
    torch.save(val_labels_tensor, f'{combined_dir}/validation/labels.pt')
    torch.save(val_langs, f'{combined_dir}/validation/languages.pt')
    
    torch.save(test_sequences, f'{combined_dir}/test/sequences.pt')
    torch.save(test_labels_tensor, f'{combined_dir}/test/labels.pt')
    torch.save(test_langs, f'{combined_dir}/test/languages.pt')
    
    print(f"\nSaved combined evaluation data to {combined_dir}")
    
    return {
        'stats': stats,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'timestamp': timestamp
    }

def main():
    """Main execution function."""
    # Generate timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting pretraining generation pipeline at {timestamp}")
    
    output_dir = 'tensors'
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 1: Pretraining datasets
    pretraining_dbs = {
        'c': juliet_c_db,
        'java': juliet_java_db,
        'csharp': juliet_csharp_db
    }
    
    pretraining_stats = create_pretraining_tensors(
        db_paths=pretraining_dbs,
        tokenizer=tokenizer,
        sample_size_per_class=pretraining_sample_size,
        output_dir=output_dir,
        timestamp=timestamp,
        balance_classes=True
    )
    
    # Phase 2: Evaluation datasets
    evaluation_dbs = {
        'c': devign_c_db,
        'python': python_db
    }
    
    evaluation_stats = create_evaluation_tensors(
        db_paths=evaluation_dbs,
        tokenizer=tokenizer,
        sample_size_per_class=eval_sample_size,
        output_dir=output_dir,
        timestamp=timestamp,
        balance_classes=True
    )
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'random_state': random_state,
        'pretraining_sample_size': pretraining_sample_size,
        'eval_sample_size': eval_sample_size,
        'pretraining_stats': pretraining_stats,
        'evaluation_stats': evaluation_stats,
        'min_tokens': MIN_TOKENS,
        'max_tokens': MAX_TOKENS
    }
    
    metadata_path = os.path.join(output_dir, timestamp, 'pipeline_metadata.json')
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("Pipeline Complete!")
    print("="*80)
    print(f"Metadata saved to: {metadata_path}")
    print(f"Pretraining tensors saved to: {output_dir}/{timestamp}/pretraining/")
    print(f"Evaluation tensors saved to: {output_dir}/{timestamp}/evaluation/")

if __name__ == "__main__":
    main()
