import sqlite3
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import numpy as np

# Constants
MAX_FUNCTION_LINES = 60
MIN_FUNCTION_LINES = 10
MAX_SEQ_LENGTH = 64

def pad_sequences(sequences, maxlen, padding='post', value=0):
    """Pads sequences to the same length."""
    output = []
    for seq in sequences:
        if len(seq) > maxlen:
            # Truncate
            new_seq = seq[:maxlen]
        else:
            # Pad
            pad_length = maxlen - len(seq)
            if padding == 'post':
                new_seq = seq + [value] * pad_length
            else:  # 'pre'
                new_seq = [value] * pad_length + seq
        output.append(new_seq)
    return np.array(output)

# Function to display vulnerable lines in code for manual verification
def display_vulnerable_lines(db_path, num_samples=5, min_lines=MIN_FUNCTION_LINES, max_lines=MAX_FUNCTION_LINES):
    """
    Display vulnerable lines from the database for manual verification.
    
    Args:
        db_path (str): Path to the SQLite database
        num_samples (int): Number of samples to display
        min_lines (int): Minimum number of lines a function should have
        max_lines (int): Maximum number of lines a function should have
    """
    print(f"Displaying vulnerable lines from {db_path} for manual verification...")
    
    try:
        # Load some vulnerable functions
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT cve, file, start, end, vuln, code FROM funcs WHERE vuln IS NOT NULL AND vuln != '' "
            f"AND (end - start ) BETWEEN {min_lines} AND {max_lines} "
            "LIMIT ?", (num_samples,)
        )
        vuln_samples = cursor.fetchall()
        
        if not vuln_samples:
            print("No vulnerable samples found!")
            return
        
        # Display each vulnerable sample and highlight the vulnerable lines
        for i, (cve, file, start, end, vuln, code) in enumerate(vuln_samples):
            print(f"\n{'='*80}")
            print(f"Sample {i+1}")
            print(f"{'='*80}")
            print(f"File: {file}")
            print(f"Line range: {start}-{end}")
            print(f"Vulnerability info: {vuln}")
            
            # Parse vulnerability lines
            try:
                vuln_lines = [int(v.strip()) for v in vuln.split(',')]
                
                code_lines = code.split('\n')
                for j, line in enumerate(code_lines):
                    # Calculate absolute line number and check if it's vulnerable
                    print(f"    {j+1:2d}: {line}")
                
                # Show relative line numbers (0-based) for use in model
                print("\nVulnerable line indices for model:")
                rel_lines = [line - start for line in vuln_lines if 0 <= (line - start) < max_lines]
                print(rel_lines)
                
            except Exception as e:
                print(f"Error parsing vulnerability info: {str(e)}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error displaying vulnerable lines: {str(e)}")

# Database functions
def load_data_from_db(db_path, limit_per_class=None, balance_classes=False, min_lines=MIN_FUNCTION_LINES, max_lines=MAX_FUNCTION_LINES):
    """
    Load function data from a database with optional class balancing.
    
    Args:
        db_path (str): Path to the SQLite database
        limit_per_class (int, optional): Maximum number of samples per class
        balance_classes (bool): Whether to balance vulnerable and non-vulnerable classes
        min_lines (int): Minimum number of lines a function should have
        max_lines (int): Maximum number of lines a function should have
        
    Returns:
        tuple: (all_functions, vulnerable_count, non_vulnerable_count)
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get functions with vulnerabilities
        cursor.execute(
            "SELECT cve, file, start, end, vuln, code FROM funcs WHERE vuln IS NOT NULL AND vuln != '' "
            f"AND (end - start + 1) BETWEEN {min_lines} AND {max_lines}"
        )
        vulnerable_funcs = cursor.fetchall()
        
        # Get functions without vulnerabilities
        cursor.execute(
            "SELECT cve, file, start, end, vuln, code FROM funcs WHERE (vuln IS NULL OR vuln = '') "
            f"AND (end - start + 1) BETWEEN {min_lines} AND {max_lines}"
        )
        non_vulnerable_funcs = cursor.fetchall()
        
        # Apply balancing if needed
        if balance_classes and limit_per_class:
            # Limit both classes to the same size
            vuln_limit = min(len(vulnerable_funcs), limit_per_class)
            non_vuln_limit = min(len(non_vulnerable_funcs), limit_per_class)
            
            # If one class is smaller, limit both to the smaller size for balance
            if balance_classes:
                min_size = min(vuln_limit, non_vuln_limit)
                vulnerable_funcs = vulnerable_funcs[:min_size]
                non_vulnerable_funcs = non_vulnerable_funcs[:min_size]
            else:
                vulnerable_funcs = vulnerable_funcs[:vuln_limit]
                non_vulnerable_funcs = non_vulnerable_funcs[:non_vuln_limit]
        
        conn.close()
        # Combine the data
        all_funcs = vulnerable_funcs + non_vulnerable_funcs
        
        return all_funcs, len(vulnerable_funcs), len(non_vulnerable_funcs)
    except Exception as e:
        print(f"Error in load_data_from_db: {str(e)}")
        return [], 0, 0

# Process vulnerability information to create labels
def create_labels(functions, max_lines=MAX_FUNCTION_LINES):
    """
    Process vulnerability information to create labels for model training.
    
    Args:
        functions (list): List of function tuples from the database
        max_lines (int): Maximum number of lines to consider per function
        
    Returns:
        tuple: (data, labels, vulnerable_count)
    """
    data = []
    labels = []
    vuln_count = 0
    
    try:
        for cve, file, start, end, vuln, code in functions:
            data.append(code)
            # Create one-hot encoded vector for vulnerability location
            # If vuln is None or empty, all zeros (no vulnerability)
            label = torch.zeros(max_lines)
            
            if vuln:
                try:
                    # Parse the vulnerability line(s)
                    vuln_lines = [int(v.strip()) for v in vuln.split(',')]
                    for line_num in vuln_lines:
                        # Just use the integer from vuln directly
                        if 0 <= line_num < max_lines:
                            label[line_num] = 1
                    
                    # Count this as a vulnerable function if at least one label was set
                    if torch.sum(label) > 0:
                        vuln_count += 1
                except Exception as e:
                    # In case of parsing errors, treat as non-vulnerable
                    pass
            
            labels.append(label)
    except Exception as e:
        print(f"Error in create_labels: {str(e)}")
    
    return data, labels, vuln_count

def tokenize_and_pad(data, tokenizer, max_samples=None, max_seq_length=MAX_SEQ_LENGTH, max_function_lines=MAX_FUNCTION_LINES):
    """
    Tokenize and pad the input text data for model input.
    
    Args:
        data (list): List of code strings
        tokenizer: The tokenizer to use
        max_samples (int, optional): Maximum number of samples to process
        max_seq_length (int): Maximum sequence length for each line
        max_function_lines (int): Maximum number of lines per function
        
    Returns:
        torch.Tensor: Tensor of tokenized and padded sequences
    """
    # Limit samples if needed
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    sequences = []
    
    # Use regular tqdm instead of notebook version
    for idx, text in enumerate(data):
        if idx % 5 == 0:
            print(f"Processing sample {idx+1}/{len(data)}...")
        
        # Extract and limit lines
        lines = text.split('\n')[:max_function_lines]
        
        # Tokenize each line
        tokenized_lines = [tokenizer.encode(line) for line in lines]
        
        # Pad each line to max_seq_length
        padded_lines = pad_sequences(tokenized_lines, maxlen=max_seq_length, padding='post')
        
        # Pad to max_function_lines if needed
        if len(padded_lines) < max_function_lines:
            padding = np.zeros((max_function_lines - len(padded_lines), max_seq_length))
            padded_lines = np.vstack((padded_lines, padding))
        elif len(padded_lines) > max_function_lines:
            padded_lines = padded_lines[:max_function_lines]
        
        sequences.append(padded_lines)
    
    # Convert to tensor
    return torch.tensor(np.array(sequences))

def preprocess_data(c_db_path, java_db_path, tokenizer, limit_per_class=5000, balance_classes=True, random_state=42, 
                   max_function_lines=MAX_FUNCTION_LINES, max_seq_length=MAX_SEQ_LENGTH, output_dir='tensors'):
    """
    Complete preprocessing workflow: load data, create labels, split, tokenize and save tensors.
    
    Args:
        c_db_path (str): Path to C/C++ database
        java_db_path (str): Path to Java database
        tokenizer: The tokenizer to use
        limit_per_class (int): Maximum number of samples per class
        balance_classes (bool): Whether to balance vulnerable and non-vulnerable classes
        random_state (int): Random seed for reproducibility
        max_function_lines (int): Maximum number of lines per function
        max_seq_length (int): Maximum sequence length for each line
        output_dir (str): Directory to save output tensors
        
    Returns:
        dict: Statistics about the preprocessing operation
    """
    import os
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Load data from databases with class balancing
    print("Loading data from databases...")
    c_funcs, c_vuln_count, c_non_vuln_count = load_data_from_db(
        c_db_path, limit_per_class=limit_per_class, balance_classes=balance_classes, 
        max_lines=max_function_lines
    )
    
    java_funcs, java_vuln_count, java_non_vuln_count = load_data_from_db(
        java_db_path, limit_per_class=limit_per_class, balance_classes=balance_classes,
        max_lines=max_function_lines
    )
    
    print(f"C Code: {c_vuln_count} vulnerable, {c_non_vuln_count} non-vulnerable, ratio: {c_vuln_count/(c_non_vuln_count or 1):.2f}")
    print(f"Java Code: {java_vuln_count} vulnerable, {java_non_vuln_count} non-vulnerable, ratio: {java_vuln_count/(java_non_vuln_count or 1):.2f}")
    
    # Combine data and create labels
    all_funcs = c_funcs + java_funcs
    data, labels, vuln_count = create_labels(all_funcs, max_lines=max_function_lines)
    
    # Quick check of positive and negative samples in the dataset
    pos_samples = sum(1 for l in labels if torch.sum(l) > 0)
    neg_samples = len(labels) - pos_samples
    print(f"\nDataset class distribution:")
    print(f"Positive samples (with vulnerability): {pos_samples} ({pos_samples/len(labels)*100:.2f}%)")
    print(f"Negative samples (without vulnerability): {neg_samples} ({neg_samples/len(labels)*100:.2f}%)")
    print(f"Positive:Negative ratio = 1:{neg_samples/pos_samples:.2f}" if pos_samples > 0 else "No positive samples found")
    
    # Step 2: Split the data
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, test_size=0.3, random_state=random_state, 
        stratify=[1 if torch.sum(l) > 0 else 0 for l in labels]
    )
    
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.33, random_state=random_state
    )
    
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)} samples")
    
    # Check class distribution in splits
    train_pos = sum(1 for l in train_labels if torch.sum(l) > 0)
    val_pos = sum(1 for l in val_labels if torch.sum(l) > 0)
    test_pos = sum(1 for l in test_labels if torch.sum(l) > 0)
    
    print(f"\nClass distribution after splitting:")
    print(f"Train: {train_pos}/{len(train_labels)} positive ({train_pos/len(train_labels)*100:.2f}%)")
    print(f"Validation: {val_pos}/{len(val_labels)} positive ({val_pos/len(val_labels)*100:.2f}%)")
    print(f"Test: {test_pos}/{len(test_labels)} positive ({test_pos/len(test_labels)*100:.2f}%)")
    
    # Step 3: Tokenize and create tensors
    print("Tokenizing data...")
    train_sequences = tokenize_and_pad(train_data, tokenizer, max_seq_length=max_seq_length, max_function_lines=max_function_lines)
    val_sequences = tokenize_and_pad(val_data, tokenizer, max_seq_length=max_seq_length, max_function_lines=max_function_lines)
    test_sequences = tokenize_and_pad(test_data, tokenizer, max_seq_length=max_seq_length, max_function_lines=max_function_lines)
    
    # Convert labels to tensors
    train_labels_tensor = torch.stack(train_labels)
    val_labels_tensor = torch.stack(val_labels)
    test_labels_tensor = torch.stack(test_labels)
    
    # Step 4: Save tensors
    print("Saving tensors...")
    torch.save(train_sequences, f'{output_dir}/cwe_train_sequences.pt')
    torch.save(train_labels_tensor, f'{output_dir}/cwe_train_labels.pt')
    torch.save(val_sequences, f'{output_dir}/cwe_val_sequences.pt')
    torch.save(val_labels_tensor, f'{output_dir}/cwe_val_labels.pt')
    torch.save(test_sequences, f'{output_dir}/cwe_test_sequences.pt')
    torch.save(test_labels_tensor, f'{output_dir}/cwe_test_labels.pt')
    
    print("Dataset creation complete!")
    
    # Return statistics
    return {
        'c_stats': {'vulnerable': c_vuln_count, 'non_vulnerable': c_non_vuln_count},
        'java_stats': {'vulnerable': java_vuln_count, 'non_vulnerable': java_non_vuln_count},
        'overall': {
            'total_samples': len(labels),
            'positive_samples': pos_samples,
            'negative_samples': neg_samples,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data)
        }
    }

print("Preprocessing functions defined successfully!")


