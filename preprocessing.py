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
CVE_ID = dict(enumerate(range(100000)))  # Maps integer indices to CVE IDs
# Expanded language support (detected from id suffix)
LANGUAGES = ['c', 'cpp', 'java', 'python', 'csharp']
# Default token thresholds for filtering functions
MAX_TOKENS = 4096
MIN_TOKENS = 32

def pad_sequence(sequence, maxlen, padding='post', value=0):
    """Pad or truncate a single sequence to maxlen."""
    if len(sequence) > maxlen:
        return np.array(sequence[:maxlen])
    pad_length = maxlen - len(sequence)
    if padding == 'post':
        return np.array(sequence + [value] * pad_length)
    return np.array([value] * pad_length + sequence)

# Function to display vulnerable lines in code for manual verification
def display_vulnerable_lines(db_path, num_samples=5, tokenizer=None, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS):
    """
    Display vulnerable lines from the database for manual verification.
    
    Args:
        db_path (str): Path to the SQLite database
        num_samples (int): Number of samples to display
        tokenizer: Optional tokenizer to show token counts
        min_tokens (int): Minimum tokens per function
        max_tokens (int): Maximum tokens per function
    """
    print(f"Displaying vulnerable samples from {db_path} for manual verification...")
    
    try:
        # Load some vulnerable functions
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT grp, id, start, end, vuln, code FROM funcs WHERE vuln IS NOT NULL AND vuln != '' LIMIT ?",
            (num_samples,)
        )
        vuln_samples = cursor.fetchall()
        
        if not vuln_samples:
            print("No vulnerable samples found!")
            return
        
        # Display each vulnerable sample and highlight the vulnerable lines
        for i, (group, file_id, start, end, vuln, code) in enumerate(vuln_samples):
            print(f"\n{'='*80}")
            print(f"Sample {i+1}")
            print(f"{'='*80}")
            print(f"Group: {group}")
            print(f"File ID: {file_id}")
            print(f"Line range: {start}-{end}")
            print(f"Vulnerability info: {vuln}")
            
            # Parse vulnerability lines
            try:
                vuln_lines = [int(v.strip()) for v in vuln.split(',')]
                
                code_lines = code.split('\n')
                for j, line in enumerate(code_lines[:50]):
                    print(f"    {j+1:2d}: {line}")
                if tokenizer is not None:
                    tok_len = len(tokenizer.encode(code, add_special_tokens=False))
                    print(f"Token count: {tok_len} (filter range {min_tokens}-{max_tokens})")
                
            except Exception as e:
                print(f"Error parsing vulnerability info: {str(e)}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error displaying vulnerable lines: {str(e)}")

# Database functions
def detect_language_from_id(file_id: str) -> str:
    """Infer language from id suffix (file extension or tag)."""
    fid = str(file_id).lower()
    if fid.endswith('.c') or fid.endswith('.cpp') or fid.endswith('.cc') or fid.endswith('.cxx'):
        return 'c'
    if fid.endswith('.java'):
        return 'java'
    if fid.endswith('.py'):
        return 'python'
    if fid.endswith('.cs'):
        return 'csharp'
    for lang in LANGUAGES:
        if lang in fid:
            return lang
    return 'unknown'

def extract_cwe_from_group(group: str) -> str:
    """
    Extract CWE identifier from a group string.
    Juliet datasets typically have groups like 'CWE121_Stack_Based_Buffer_Overflow'.
    
    Args:
        group (str): The group string from the database
        
    Returns:
        str: The CWE identifier (e.g., 'CWE121') or 'unknown'
    """
    import re
    if group is None:
        return 'unknown'
    # Match CWE followed by digits
    match = re.match(r'(CWE\d+)', str(group))
    if match:
        return match.group(1)
    return 'unknown'


def balance_samples_by_cve(funcs, target_count, random_state=42):
    """
    Balance samples across CVEs by sampling evenly from each CVE category.
    
    Args:
        funcs (list): List of function tuples with (group, file_id, start, end, vuln, code, tokens)
        target_count (int): Target total number of samples
        random_state (int): Random seed for reproducibility
        
    Returns:
        list: Balanced list of functions
    """
    from collections import defaultdict
    import random
    
    random.seed(random_state)
    
    # Group functions by CWE
    cwe_groups = defaultdict(list)
    for func in funcs:
        group = func[0]  # grp is the first element
        cwe = extract_cwe_from_group(group)
        cwe_groups[cwe].append(func)
    
    num_cwes = len(cwe_groups)
    if num_cwes == 0:
        return []
    
    # Calculate how many samples to take from each CWE
    samples_per_cwe = target_count // num_cwes
    remainder = target_count % num_cwes
    
    balanced_funcs = []
    cwe_list = list(cwe_groups.keys())
    random.shuffle(cwe_list)  # Shuffle to randomly distribute remainder
    
    for i, cwe in enumerate(cwe_list):
        cwe_funcs = cwe_groups[cwe]
        # Add one extra sample to some CVEs to handle remainder
        extra = 1 if i < remainder else 0
        take_count = min(len(cwe_funcs), samples_per_cwe + extra)
        
        # Randomly sample from this CWE
        if take_count < len(cwe_funcs):
            sampled = random.sample(cwe_funcs, take_count)
        else:
            sampled = cwe_funcs
        balanced_funcs.extend(sampled)
    
    # Shuffle the final result
    random.shuffle(balanced_funcs)
    return balanced_funcs


def load_data_from_db(db_path, tokenizer, limit_per_class=None, balance_classes=False, balance_cve=False, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS, random_state=42):
    """
    Load function data from a database with optional class balancing.
    
    Args:
        db_path (str): Path to the SQLite database
        tokenizer: Tokenizer used to compute token counts
        limit_per_class (int, optional): Maximum number of samples per class
        balance_classes (bool): Whether to balance vulnerable and non-vulnerable classes
        balance_cve (bool): Whether to balance samples evenly across CVEs (for Juliet datasets)
        min_tokens (int): Minimum number of tokens per function
        max_tokens (int): Maximum number of tokens per function
        random_state (int): Random seed for CVE balancing
        
    Returns:
        tuple: (all_functions, vulnerable_count, non_vulnerable_count, cve_to_idx_mapping)
               cve_to_idx_mapping is a dict mapping CWE strings to integer indices
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Fetch ALL data first, then classify based on vuln field format
        cursor.execute("SELECT grp, id, start, end, vuln, code FROM funcs")
        all_candidates = cursor.fetchall()
        
        def is_vulnerable(vuln_value):
            """
            Determine if a sample is vulnerable based on the vuln field.
            Handles both formats:
            - Juliet: non-empty string (CWE lines) = vulnerable, empty = non-vulnerable
            - Devign/BugsInPy: '1' = vulnerable, '0' = non-vulnerable
            """
            if vuln_value is None:
                return False
            vuln_str = str(vuln_value).strip()
            if vuln_str == '':
                return False
            if vuln_str == '0':
                return False  # Explicitly non-vulnerable (Devign/BugsInPy format)
            return True  # '1' or CWE line numbers = vulnerable
        
        def filter_by_tokens(rows, check_vulnerable):
            out = []
            for row in rows:
                group, file_id, start, end, vuln, code = row
                if is_vulnerable(vuln) != check_vulnerable:
                    continue
                tokens = tokenizer.encode(code, add_special_tokens=False)
                if min_tokens <= len(tokens) <= max_tokens:
                    out.append((group, file_id, start, end, vuln, code, tokens))
            return out

        vulnerable_funcs = filter_by_tokens(all_candidates, check_vulnerable=True)
        non_vulnerable_funcs = filter_by_tokens(all_candidates, check_vulnerable=False)
        
        # Build CWE to index mapping from all data (before any filtering)
        all_cwes = set()
        for func in vulnerable_funcs + non_vulnerable_funcs:
            cwe = extract_cwe_from_group(func[0])
            all_cwes.add(cwe)
        cwe_to_idx = {cwe: idx for idx, cwe in enumerate(sorted(all_cwes))}
        
        # Apply CVE balancing if requested (for Juliet datasets)
        if balance_cve and limit_per_class is not None:
            print(f"Applying CVE-balanced sampling...")
            # Get unique CVEs and their counts before balancing
            from collections import Counter
            vuln_cwe_counts = Counter(extract_cwe_from_group(f[0]) for f in vulnerable_funcs)
            non_vuln_cwe_counts = Counter(extract_cwe_from_group(f[0]) for f in non_vulnerable_funcs)
            print(f"  Vulnerable samples span {len(vuln_cwe_counts)} CVEs")
            print(f"  Non-vulnerable samples span {len(non_vuln_cwe_counts)} CVEs")
            
            # Calculate target counts
            vuln_target = min(len(vulnerable_funcs), limit_per_class)
            non_vuln_target = min(len(non_vulnerable_funcs), limit_per_class)
            
            if balance_classes:
                min_size = min(vuln_target, non_vuln_target)
                vuln_target = min_size
                non_vuln_target = min_size
            
            # Balance by CVE
            vulnerable_funcs = balance_samples_by_cve(vulnerable_funcs, vuln_target, random_state)
            non_vulnerable_funcs = balance_samples_by_cve(non_vulnerable_funcs, non_vuln_target, random_state)
            
            # Report CVE distribution after balancing
            vuln_cwe_after = Counter(extract_cwe_from_group(f[0]) for f in vulnerable_funcs)
            non_vuln_cwe_after = Counter(extract_cwe_from_group(f[0]) for f in non_vulnerable_funcs)
            print(f"  After balancing: {len(vulnerable_funcs)} vulnerable, {len(non_vulnerable_funcs)} non-vulnerable")
            print(f"  CVEs represented: {len(vuln_cwe_after)} (vuln), {len(non_vuln_cwe_after)} (non-vuln)")
            
        # Apply standard balancing if CVE balancing is not used
        elif limit_per_class is not None:
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
        return all_funcs, len(vulnerable_funcs), len(non_vulnerable_funcs), cwe_to_idx
    except Exception as e:
        print(f"Error in load_data_from_db: {str(e)}")
        return [], 0, 0, {}

# Process vulnerability information to create labels
def create_labels(functions, cwe_to_idx=None):
    """
    Process vulnerability information to create labels for model training.
    
    Args:
        functions (list): List of function tuples (with tokens) from the database
        cwe_to_idx (dict, optional): Mapping from CWE strings to integer indices
        
    Returns:
        tuple: (data, labels, vulnerable_count, language_identifiers, cwe_indices)
               cwe_indices is a list of integer indices corresponding to the CWE of each sample
    """
    data = []
    labels = []
    language_identifiers = []
    cwe_indices = []
    vuln_count = 0
    
    # Default mapping if none provided
    if cwe_to_idx is None:
        cwe_to_idx = {}
    
    try:
        for group, file_id, start, end, vuln, code, tokens in functions:
            data.append(code)
            # Binary label: 1 if vulnerable, else 0
            # Handle both CWE labels (Juliet: non-empty = vulnerable) and binary strings (Devign/BugsInPy: '1' = vulnerable, '0' = secure)
            vuln_str = str(vuln).strip() if vuln is not None else ''
            if vuln_str in ['0', '']:
                is_vuln = 0  # Explicitly secure or empty
            elif vuln_str == '1':
                is_vuln = 1  # Explicitly vulnerable (binary label)
            else:
                is_vuln = 1  # CWE identifier or any other non-empty value = vulnerable
            
            labels.append(torch.tensor(is_vuln, dtype=torch.long))
            if is_vuln == 1:
                vuln_count += 1
            language_identifiers.append(detect_language_from_id(file_id))
            
            # Get CWE index for this sample
            cwe = extract_cwe_from_group(group)
            cwe_idx = cwe_to_idx.get(cwe, -1)  # -1 for unknown CVEs
            cwe_indices.append(cwe_idx)
    except Exception as e:
        print(f"Error in create_labels: {str(e)}")
    
    return data, labels, vuln_count, language_identifiers, cwe_indices

def tokenize_and_pad(data, tokenizer, max_samples=None, max_seq_length=MAX_TOKENS):
    """
    Tokenize and pad the input text data for model input.
    
    Args:
        data (list): List of code strings
        tokenizer: The tokenizer to use
        max_samples (int, optional): Maximum number of samples to process
        max_seq_length (int): Maximum sequence length per function (tokens)
        
    Returns:
        torch.Tensor: Tensor of tokenized and padded sequences
    """
    # Limit samples if needed
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    sequences = []

    for idx, text in enumerate(data):
        if idx % 1000 == 0:
            print(f"Processing sample {idx+1}/{len(data)}...")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        padded = pad_sequence(tokens, maxlen=max_seq_length, padding='post', value=0)
        sequences.append(padded)

    return torch.tensor(np.stack(sequences))

def preprocess_data(db_path, tokenizer, limit_per_class=50000, balance_classes=True, balance_cve=False, 
                   random_state=42, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS, max_seq_length=MAX_TOKENS,
                   output_dir='tensors', dataset_name=None, full_dataset_mode=False):
    """
    Complete preprocessing workflow for a single dataset: load data, create labels, split, tokenize and save tensors.
    
    Args:
        db_path (str): Path to the SQLite database
        tokenizer: The tokenizer to use
        limit_per_class (int): Maximum number of samples per class
        balance_classes (bool): Whether to balance vulnerable and non-vulnerable classes
        balance_cve (bool): Whether to balance samples evenly across CVEs (for Juliet datasets)
        random_state (int): Random seed for reproducibility
        min_tokens (int): Minimum number of tokens per function
        max_tokens (int): Maximum number of tokens per function
        max_seq_length (int): Maximum sequence length for tokenization
        output_dir (str): Directory to save output tensors
        dataset_name (str, optional): Name for the dataset (used in folder naming). 
                                      If None, extracted from db_path filename.
        full_dataset_mode (bool): If True, skip train/val/test split and save all data as a single dataset.
                                  Useful for creating OOD evaluation sets.
        
    Returns:
        dict: Statistics about the preprocessing operation
    """
    import os
    import datetime
    import json
    from collections import Counter
    
    # Extract dataset name from path if not provided
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(db_path))[0]
    
    # Generate folder name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{dataset_name}_{timestamp}_seed{random_state}"
    
    # Create the output directory
    output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load data
    print(f"Loading data from {db_path}...")
    funcs, vuln_count, non_vuln_count, cwe_to_idx = load_data_from_db(
        db_path, tokenizer=tokenizer, limit_per_class=limit_per_class, 
        balance_classes=balance_classes, balance_cve=balance_cve, 
        min_tokens=min_tokens, max_tokens=max_tokens, random_state=random_state
    )
    
    # Create reverse mapping (idx to CWE)
    idx_to_cwe = {idx: cwe for cwe, idx in cwe_to_idx.items()}
    
    print(f"Loaded: {vuln_count} vulnerable, {non_vuln_count} non-vulnerable")
    print(f"Total unique CWEs: {len(cwe_to_idx)}")
    
    # Create labels and CWE indices
    data, labels, _, _, cwe_indices = create_labels(funcs, cwe_to_idx)
    
    # Dataset statistics
    pos_samples = sum(1 for l in labels if l.item() > 0)
    neg_samples = len(labels) - pos_samples
    print(f"\nDataset: {pos_samples} positive ({pos_samples/len(labels)*100:.1f}%), "
          f"{neg_samples} negative ({neg_samples/len(labels)*100:.1f}%)")
    
    # Full dataset mode: skip train/val/test split
    if full_dataset_mode:
        print("\nFull dataset mode: skipping train/val/test split...")
        
        # Tokenize all data
        print("Tokenizing...")
        all_sequences = tokenize_and_pad(data, tokenizer, max_seq_length=max_seq_length)
        
        # Convert to tensors
        all_labels_tensor = torch.stack(labels)
        all_cwes_tensor = torch.tensor(cwe_indices, dtype=torch.long)
        
        # Still saving tensors with test_prefix so it works with the batch loading script. 
        print("Saving tensors...")
        torch.save(all_sequences, f'{output_dir}/test_sequences.pt')
        torch.save(all_labels_tensor, f'{output_dir}/test_labels.pt')
        torch.save(all_cwes_tensor, f'{output_dir}/test_cwe_indices.pt')

        # Save CWE mappings
        torch.save(cwe_to_idx, f'{output_dir}/cwe_to_idx.pt')
        torch.save(idx_to_cwe, f'{output_dir}/idx_to_cwe.pt')
        
        # CWE distribution stats
        cwe_dist = Counter(cwe_indices)
        
        # Save metadata
        metadata = {
            'dataset_name': dataset_name,
            'database': os.path.basename(db_path),
            'timestamp': datetime.datetime.now().isoformat(),
            'seed': random_state,
            'balance_classes': balance_classes,
            'balance_cve': balance_cve,
            'limit_per_class': limit_per_class,
            'min_tokens': min_tokens,
            'max_tokens': max_tokens,
            'max_seq_length': max_seq_length,
            'full_dataset_mode': True,
            'num_cwes': len(cwe_to_idx),
            'cwe_mapping': cwe_to_idx,
            'stats': {
                'total_samples': len(labels),
                'vulnerable_count': vuln_count,
                'non_vulnerable_count': non_vuln_count,
                'positive_count': pos_samples,
                'negative_count': neg_samples,
            }
        }
        
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nFull dataset saved to: {output_dir}")
        print(f"Total samples: {len(labels)}")
        print(f"CWEs represented: {len(cwe_dist)}")
        
        return {
            'output_dir': output_dir,
            'dataset_name': dataset_name,
            'total_samples': len(labels),
            'vulnerable': vuln_count,
            'non_vulnerable': non_vuln_count,
            'train_size': 0,
            'val_size': 0,
            'test_size': 0,
            'full_size': len(labels),
            'num_cwes': len(cwe_to_idx),
            'cwe_to_idx': cwe_to_idx,
            'idx_to_cwe': idx_to_cwe,
            'cwe_distribution': dict(cwe_dist),
            'seed': random_state,
            'balance_cve': balance_cve,
            'full_dataset_mode': True,
        }
    
    # Standard mode: Split the data (70% train, 20% val, 10% test)
    train_data, temp_data, train_labels, temp_labels, train_cwes, temp_cwes = train_test_split(
        data, labels, cwe_indices, test_size=0.3, random_state=random_state,
        stratify=[int(l.item()) for l in labels]
    )
    
    val_data, test_data, val_labels, test_labels, val_cwes, test_cwes = train_test_split(
        temp_data, temp_labels, temp_cwes, test_size=0.33, random_state=random_state
    )
    
    # Class distribution in splits
    train_pos = sum(int(l.item()) for l in train_labels)
    val_pos = sum(int(l.item()) for l in val_labels)
    test_pos = sum(int(l.item()) for l in test_labels)
    
    print(f"\nSplit sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print(f"Positive %: Train={train_pos/len(train_labels)*100:.1f}%, "
          f"Val={val_pos/len(val_labels)*100:.1f}%, Test={test_pos/len(test_labels)*100:.1f}%")
    
    # Step 3: Tokenize and create tensors
    print("\nTokenizing...")
    train_sequences = tokenize_and_pad(train_data, tokenizer, max_seq_length=max_seq_length)
    val_sequences = tokenize_and_pad(val_data, tokenizer, max_seq_length=max_seq_length)
    test_sequences = tokenize_and_pad(test_data, tokenizer, max_seq_length=max_seq_length)
    
    # Convert to tensors
    train_labels_tensor = torch.stack(train_labels)
    val_labels_tensor = torch.stack(val_labels)
    test_labels_tensor = torch.stack(test_labels)
    
    train_cwes_tensor = torch.tensor(train_cwes, dtype=torch.long)
    val_cwes_tensor = torch.tensor(val_cwes, dtype=torch.long)
    test_cwes_tensor = torch.tensor(test_cwes, dtype=torch.long)
    
    # Step 4: Save tensors
    print("Saving tensors...")
    
    torch.save(train_sequences, f'{output_dir}/train_sequences.pt')
    torch.save(train_labels_tensor, f'{output_dir}/train_labels.pt')
    torch.save(train_cwes_tensor, f'{output_dir}/train_cwe_indices.pt')

    torch.save(val_sequences, f'{output_dir}/val_sequences.pt')
    torch.save(val_labels_tensor, f'{output_dir}/val_labels.pt')
    torch.save(val_cwes_tensor, f'{output_dir}/val_cwe_indices.pt')

    torch.save(test_sequences, f'{output_dir}/test_sequences.pt')
    torch.save(test_labels_tensor, f'{output_dir}/test_labels.pt')
    torch.save(test_cwes_tensor, f'{output_dir}/test_cwe_indices.pt')
    
    # Save CWE mappings
    torch.save(cwe_to_idx, f'{output_dir}/cwe_to_idx.pt')
    torch.save(idx_to_cwe, f'{output_dir}/idx_to_cwe.pt')
    
    # CWE distribution stats
    train_cwe_dist = Counter(train_cwes)
    test_cwe_dist = Counter(test_cwes)
    
    # Save metadata
    metadata = {
        'dataset_name': dataset_name,
        'database': os.path.basename(db_path),
        'timestamp': datetime.datetime.now().isoformat(),
        'seed': random_state,
        'balance_classes': balance_classes,
        'balance_cve': balance_cve,
        'limit_per_class': limit_per_class,
        'min_tokens': min_tokens,
        'max_tokens': max_tokens,
        'max_seq_length': max_seq_length,
        'full_dataset_mode': False,
        'num_cwes': len(cwe_to_idx),
        'cwe_mapping': cwe_to_idx,
        'stats': {
            'total_samples': len(labels),
            'vulnerable_count': vuln_count,
            'non_vulnerable_count': non_vuln_count,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'train_positive': train_pos,
            'val_positive': val_pos,
            'test_positive': test_pos,
        }
    }
    
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset saved to: {output_dir}")
    print(f"CWE distribution: {len(train_cwe_dist)} CWEs in train, {len(test_cwe_dist)} CWEs in test")
    
    return {
        'output_dir': output_dir,
        'dataset_name': dataset_name,
        'total_samples': len(labels),
        'vulnerable': vuln_count,
        'non_vulnerable': non_vuln_count,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'num_cwes': len(cwe_to_idx),
        'cwe_to_idx': cwe_to_idx,
        'idx_to_cwe': idx_to_cwe,
        'train_cwe_distribution': dict(train_cwe_dist),
        'test_cwe_distribution': dict(test_cwe_dist),
        'seed': random_state,
        'balance_cve': balance_cve,
    }

print("Preprocessing functions defined successfully!")


