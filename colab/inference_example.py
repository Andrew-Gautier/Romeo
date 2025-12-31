"""
Quick reference for loading and using trained LSTM models.
Use this after training models with load_and_predict.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Model definition (must match training)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 batch_first, bidirectional, dropout, pretrained_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                          bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
    
    def forward(self, text):
        if text.dim() == 3:
            batch_size, seq_length, _ = text.size()
            text = text.view(batch_size, -1)
        
        embedded = self.dropout(self.embedding(text))
        lstm_output, (hidden, _) = self.rnn(embedded)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)
        output = self.fc(self.dropout(attended_output))
        output = torch.sigmoid(output)
        
        return output


def load_trained_model(model_path, device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to saved .pt file
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded LSTM model in eval mode
        checkpoint: Full checkpoint dict with metrics
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Initialize model (without pretrained weights for inference)
    model = LSTMClassifier(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        n_layers=config['n_layers'],
        batch_first=True,
        bidirectional=config['bidirectional'],
        dropout=config['dropout'],
        pretrained_weights=None
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model: {config['language']}")
    print(f"Best Val Loss: {checkpoint['best_val_loss']:.4f}")
    print(f"Final Val AUROC: {checkpoint['final_val_auroc']:.4f}")
    
    return model, checkpoint


def predict_sequences(model, sequences, device='cuda', batch_size=32, threshold=0.5):
    """
    Make predictions on sequences.
    
    Args:
        model: Trained LSTM model
        sequences: Tensor of shape [num_samples, seq_length]
        device: Device for inference
        batch_size: Batch size for inference
        threshold: Probability threshold for binary classification
    
    Returns:
        probabilities: Vulnerability probabilities [num_samples]
        predictions: Binary predictions [num_samples] (0=secure, 1=vulnerable)
    """
    model.eval()
    
    # Create dataloader
    dataset = TensorDataset(sequences.long())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    
    with torch.no_grad():
        for (batch_seq,) in dataloader:
            batch_seq = batch_seq.to(device)
            probs = model(batch_seq).squeeze(1)
            all_probs.append(probs.cpu())
    
    probabilities = torch.cat(all_probs)
    predictions = (probabilities > threshold).long()
    
    return probabilities, predictions


def find_optimal_threshold(model, sequences, labels, device='cuda', batch_size=32, 
                          metric='f1', min_recall=0.95):
    """
    Find optimal threshold that maximizes a metric while maintaining minimum recall.
    For security applications, prioritize recall (detecting vulnerabilities).
    
    Args:
        model: Trained LSTM model
        sequences: Validation sequences
        labels: True labels
        device: Device for evaluation
        batch_size: Batch size
        metric: Metric to optimize ('f1', 'f2', 'recall', 'precision')
        min_recall: Minimum recall constraint (default 0.95 for security)
    
    Returns:
        optimal_threshold: Best threshold value
        metrics_at_threshold: Dict with precision, recall, f1, f2 at optimal threshold
        threshold_analysis: Dict with metrics at different thresholds
    """
    from sklearn.metrics import precision_recall_curve, f1_score, fbeta_score
    import numpy as np
    
    # Get probabilities
    probabilities, _ = predict_sequences(model, sequences, device, batch_size, threshold=0.5)
    labels_np = labels.cpu().numpy()
    probs_np = probabilities.numpy()
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(labels_np, probs_np)
    
    # Calculate F1 and F2 scores for each threshold
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    # F2 score weights recall higher than precision (beta=2)
    f2_scores = 5 * (precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1] + 1e-10)
    
    # Filter thresholds that meet minimum recall requirement
    valid_indices = recalls[:-1] >= min_recall
    
    if not valid_indices.any():
        print(f"Warning: No threshold achieves minimum recall of {min_recall}")
        print(f"Maximum achievable recall: {recalls[:-1].max():.4f}")
        # Use threshold that maximizes recall
        optimal_idx = np.argmax(recalls[:-1])
    else:
        if metric == 'f1':
            optimal_idx = np.argmax(f1_scores * valid_indices)
        elif metric == 'f2':
            optimal_idx = np.argmax(f2_scores * valid_indices)
        elif metric == 'recall':
            optimal_idx = np.argmax(recalls[:-1] * valid_indices)
        elif metric == 'precision':
            optimal_idx = np.argmax(precisions[:-1] * valid_indices)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    optimal_threshold = thresholds[optimal_idx]
    
    metrics_at_threshold = {
        'threshold': optimal_threshold,
        'precision': precisions[optimal_idx],
        'recall': recalls[optimal_idx],
        'f1': f1_scores[optimal_idx],
        'f2': f2_scores[optimal_idx]
    }
    
    # Provide analysis of different threshold options
    threshold_analysis = {
        'conservative': {  # Very low threshold - catches almost everything
            'threshold': np.percentile(probs_np[labels_np == 1], 5),  # 5th percentile of positive examples
            'description': 'Extremely sensitive - minimizes false negatives'
        },
        'balanced_high_recall': {  # Balanced but biased toward recall
            'threshold': optimal_threshold,
            'description': f'Optimizes {metric} with min recall {min_recall}'
        },
        'default': {
            'threshold': 0.5,
            'description': 'Standard 50% threshold'
        }
    }
    
    # Calculate metrics for each suggested threshold
    for name, info in threshold_analysis.items():
        thresh = info['threshold']
        preds = (probs_np > thresh).astype(int)
        from sklearn.metrics import precision_score, recall_score
        info['precision'] = precision_score(labels_np, preds, zero_division=0)
        info['recall'] = recall_score(labels_np, preds, zero_division=0)
        info['f1'] = f1_score(labels_np, preds, zero_division=0)
        info['f2'] = fbeta_score(labels_np, preds, beta=2, zero_division=0)
    
    return optimal_threshold, metrics_at_threshold, threshold_analysis


def evaluate_model(model, sequences, labels, device='cuda', batch_size=32, threshold=0.5, 
                  return_probabilities=False):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained LSTM model
        sequences: Test sequences [num_samples, seq_length]
        labels: True labels [num_samples]
        device: Device for evaluation
        batch_size: Batch size
        threshold: Classification threshold
        return_probabilities: If True, return probabilities and predictions
    
    Returns:
        metrics: Dict with accuracy, precision, recall, f1, f2, false_negatives, false_positives
        (probabilities, predictions): Optional tuple if return_probabilities=True
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
    
    probabilities, predictions = predict_sequences(model, sequences, device, batch_size, threshold)
    
    labels_np = labels.cpu().numpy()
    predictions_np = predictions.numpy()
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels_np, predictions_np).ravel()
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(labels_np, predictions_np),
        'precision': precision_score(labels_np, predictions_np, zero_division=0),
        'recall': recall_score(labels_np, predictions_np, zero_division=0),
        'f1': f1_score(labels_np, predictions_np, zero_division=0),
        'f2': fbeta_score(labels_np, predictions_np, beta=2, zero_division=0),  # Favors recall
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_samples': len(labels_np),
        'positive_samples': int(labels_np.sum()),
        'negative_samples': int(len(labels_np) - labels_np.sum())
    }
    
    if return_probabilities:
        return metrics, probabilities, predictions
    
    return metrics


def evaluate_model_multirun(model, sequences, labels, device='cuda', batch_size=32, num_runs=5, 
                           seed_start=42, threshold=0.5, auto_threshold=False, min_recall=0.95):
    """
    Evaluate model multiple times with different random seeds and return statistics.
    
    Args:
        model: Trained LSTM model
        sequences: Test sequences [num_samples, seq_length]
        labels: True labels [num_samples]
        device: Device for evaluation
        batch_size: Batch size
        num_runs: Number of evaluation runs with different seeds
        seed_start: Starting seed value
        threshold: Fixed threshold to use (ignored if auto_threshold=True)
        auto_threshold: If True, find optimal threshold on first run
        min_recall: Minimum recall constraint when auto_threshold=True
    
    Returns:
        stats: Dict with mean, std, min, max for each metric
        all_results: List of metric dicts from each run
        threshold_used: The threshold value used
    """
    import numpy as np
    
    all_results = []
    threshold_used = threshold
    
    # If auto_threshold, find optimal on first evaluation
    if auto_threshold:
        print(f"Finding optimal threshold with min_recall={min_recall}...")
        optimal_thresh, optimal_metrics, threshold_analysis = find_optimal_threshold(
            model, sequences, labels, device, batch_size, metric='f2', min_recall=min_recall
        )
        threshold_used = optimal_thresh
        
        print(f"\nThreshold Analysis:")
        for name, info in threshold_analysis.items():
            print(f"  {name.upper()}: threshold={info['threshold']:.6f}")
            print(f"    Precision: {info['precision']:.4f}, Recall: {info['recall']:.4f}, F1: {info['f1']:.4f}, F2: {info['f2']:.4f}")
            print(f"    {info['description']}")
        
        print(f"\nUsing threshold: {threshold_used:.6f} (optimizes F2 with min_recall={min_recall})")
        if threshold_used < 0.01:
            print(f"  ℹ️  Low threshold indicates model outputs very low probabilities even for vulnerable code.")
            print(f"      This is common with sigmoid outputs when model is highly confident but probabilities")
            print(f"      are compressed near 0. The threshold adapts to the model's probability distribution.")
    
    for run_idx in range(num_runs):
        seed = seed_start + run_idx
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        metrics = evaluate_model(model, sequences, labels, device, batch_size, threshold_used)
        all_results.append(metrics)
    
    # Calculate statistics across runs
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'f2', 'false_negatives', 'false_positives']
    stats = {}
    
    for metric in metric_names:
        values = [result[metric] for result in all_results]
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'values': values
        }
    
    # Add threshold info
    stats['threshold'] = threshold_used
    stats['auto_threshold'] = auto_threshold
    
    return stats, all_results, threshold_used


def create_evaluation_matrix(models_dir, eval_base_dir, device='cuda', num_runs=5, seed_start=42,
                           use_auto_threshold=True, min_recall=0.90):
    """
    Create evaluation matrix: test all models on C and Python test sets with multiple runs.
    Supports automatic threshold tuning for security-focused applications.
    
    Args:
        models_dir: Directory containing trained models
        eval_base_dir: Base directory for evaluation data
        device: Device for evaluation
        num_runs: Number of evaluation runs per model-dataset pair
        seed_start: Starting seed for random evaluations
        use_auto_threshold: If True, automatically find optimal threshold per model
        min_recall: Minimum recall constraint for auto thresholding (security focus)
    
    Returns:
        results: Dict with evaluation metrics for each model-testset pair
    """
    import os
    import pandas as pd
    import numpy as np
    
    # Find all model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_lstm.pt')]
    
    # Test set configurations - try both 'test' and 'full' directories
    test_sets = {}
    for lang in ['c', 'python']:
        test_path = os.path.join(eval_base_dir, lang, 'test')
        if not os.path.exists(test_path):
            test_path = os.path.join(eval_base_dir, lang, 'full')
        test_sets[lang.upper()] = test_path
    
    results = {}
    matrix_data = []
    detailed_data = []
    threshold_data = []
    
    print("="*80)
    print("EVALUATION MATRIX: Models vs Test Sets")
    print(f"Running {num_runs} evaluations per model-dataset pair")
    if use_auto_threshold:
        print(f"Auto-threshold ENABLED: Optimizing for F2 with min_recall={min_recall}")
        print(f"This prioritizes vulnerability detection (minimizes false negatives)")
    else:
        print(f"Using fixed threshold: 0.5")
    print("="*80)
    
    for model_file in sorted(model_files):
        model_name = model_file.replace('_lstm.pt', '').upper()
        model_path = os.path.join(models_dir, model_file)
        
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")
        
        # Load model
        try:
            model, checkpoint = load_trained_model(model_path, device)
        except Exception as e:
            print(f"Error loading model {model_file}: {e}")
            continue
        
        results[model_name] = {}
        
        for test_name, test_path in test_sets.items():
            print(f"\n  Testing on {test_name} test set...")
            
            # Load test data
            seq_path = os.path.join(test_path, 'sequences.pt')
            label_path = os.path.join(test_path, 'labels.pt')
            
            if not os.path.exists(seq_path):
                print(f"    ⚠ Test data not found: {seq_path}")
                results[model_name][test_name] = None
                continue
            
            test_sequences = torch.load(seq_path)
            test_labels = torch.load(label_path)
            
            # FIXED: Actually count the class distribution from labels
            num_vulnerable = int(test_labels.sum().item())
            num_secure = int((test_labels == 0).sum().item())
            total_samples = len(test_labels)
            
            print(f"    Total Samples: {total_samples}")
            print(f"    Vulnerable: {num_vulnerable} ({num_vulnerable/total_samples*100:.1f}%)")
            print(f"    Secure: {num_secure} ({num_secure/total_samples*100:.1f}%)")
            
            # Sanity check for class imbalance
            if num_vulnerable == 0:
                print(f"    ⚠️  WARNING: No vulnerable samples in test set!")
            elif num_secure == 0:
                print(f"    ⚠️  WARNING: No secure samples in test set!")
            elif num_vulnerable == total_samples:
                print(f"    🚨 CRITICAL: Test set is 100% vulnerable - evaluation is meaningless!")
            elif num_secure == total_samples:
                print(f"    🚨 CRITICAL: Test set is 100% secure - evaluation is meaningless!")
            
            # Get probability distribution diagnostics
            print(f"    Analyzing model probability outputs...")
            probabilities, _ = predict_sequences(model, test_sequences, device, threshold=0.5)
            print(f"    Probability stats: min={probabilities.min():.6f}, max={probabilities.max():.6f}, mean={probabilities.mean():.6f}, median={probabilities.median():.6f}")
            print(f"    Probs < 0.01: {(probabilities < 0.01).sum()}/{len(probabilities)} ({(probabilities < 0.01).sum()/len(probabilities)*100:.1f}%)")
            print(f"    Probs > 0.99: {(probabilities > 0.99).sum()}/{len(probabilities)} ({(probabilities > 0.99).sum()/len(probabilities)*100:.1f}%)")
            print(f"    Probs 0.01-0.99: {((probabilities >= 0.01) & (probabilities <= 0.99)).sum()}/{len(probabilities)} ({((probabilities >= 0.01) & (probabilities <= 0.99)).sum()/len(probabilities)*100:.1f}%)")
            
            # Evaluate with multiple runs and optional auto-thresholding
            try:
                stats, all_results, threshold_used = evaluate_model_multirun(
                    model, test_sequences, test_labels, device, 
                    batch_size=32, num_runs=num_runs, seed_start=seed_start,
                    auto_threshold=use_auto_threshold, min_recall=min_recall
                )
                results[model_name][test_name] = stats
                
                print(f"    Threshold: {threshold_used:.6f}")
                print(f"    Recall:    {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f}")
                print(f"    Precision: {stats['precision']['mean']:.4f} ± {stats['precision']['std']:.4f}")
                print(f"    F2:        {stats['f2']['mean']:.4f} ± {stats['f2']['std']:.4f}")
                print(f"    False Negatives: {stats['false_negatives']['mean']:.1f}, False Positives: {stats['false_positives']['mean']:.1f}")
                
                # Store aggregated results for matrix
                matrix_data.append({
                    'Model': model_name,
                    'Test Set': test_name,
                    'Threshold': threshold_used,
                    'Auto_Threshold': use_auto_threshold,
                    'Total_Samples': total_samples,
                    'Vulnerable_Samples': num_vulnerable,
                    'Secure_Samples': num_secure,
                    'Prob_Min': probabilities.min().item(),
                    'Prob_Max': probabilities.max().item(),
                    'Prob_Mean': probabilities.mean().item(),
                    'Prob_Median': probabilities.median().item(),
                    'Accuracy_Mean': stats['accuracy']['mean'],
                    'Accuracy_Std': stats['accuracy']['std'],
                    'Accuracy_Range': stats['accuracy']['range'],
                    'Precision_Mean': stats['precision']['mean'],
                    'Precision_Std': stats['precision']['std'],
                    'Precision_Range': stats['precision']['range'],
                    'Recall_Mean': stats['recall']['mean'],
                    'Recall_Std': stats['recall']['std'],
                    'Recall_Range': stats['recall']['range'],
                    'F1_Mean': stats['f1']['mean'],
                    'F1_Std': stats['f1']['std'],
                    'F1_Range': stats['f1']['range'],
                    'F2_Mean': stats['f2']['mean'],
                    'F2_Std': stats['f2']['std'],
                    'F2_Range': stats['f2']['range'],
                    'FalseNegatives_Mean': stats['false_negatives']['mean'],
                    'FalseNegatives_Std': stats['false_negatives']['std'],
                    'FalsePositives_Mean': stats['false_positives']['mean'],
                    'FalsePositives_Std': stats['false_positives']['std']
                })
                
                # Store threshold info
                threshold_data.append({
                    'Model': model_name,
                    'Test Set': test_name,
                    'Threshold': threshold_used,
                    'Auto_Threshold': use_auto_threshold,
                    'Min_Recall_Constraint': min_recall if use_auto_threshold else None
                })
                
                # Store detailed per-run results
                for run_idx, run_results in enumerate(all_results):
                    detailed_data.append({
                        'Model': model_name,
                        'Test Set': test_name,
                        'Run': run_idx + 1,
                        'Seed': seed_start + run_idx,
                        'Threshold': threshold_used,
                        'Accuracy': run_results['accuracy'],
                        'Precision': run_results['precision'],
                        'Recall': run_results['recall'],
                        'F1': run_results['f1'],
                        'F2': run_results['f2'],
                        'False_Negatives': run_results['false_negatives'],
                        'False_Positives': run_results['false_positives'],
                        'True_Positives': run_results['true_positives'],
                        'True_Negatives': run_results['true_negatives']
                    })
                
            except Exception as e:
                print(f"    ✗ Error evaluating: {e}")
                import traceback
                traceback.print_exc()
                results[model_name][test_name] = None
        
        # Clear memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create summary tables
    print("\n" + "="*80)
    print("EVALUATION MATRIX SUMMARY")
    print("="*80)
    
    if matrix_data:
        df = pd.DataFrame(matrix_data)
        
        # Show dataset composition first
        print("\n--- Dataset Composition ---")
        dataset_info = df[['Test Set', 'Total_Samples', 'Vulnerable_Samples', 'Secure_Samples']].drop_duplicates()
        for _, row in dataset_info.iterrows():
            vuln_pct = row['Vulnerable_Samples'] / row['Total_Samples'] * 100
            print(f"{row['Test Set']:8} | Total: {int(row['Total_Samples']):6,} | "
                  f"Vulnerable: {int(row['Vulnerable_Samples']):6,} ({vuln_pct:.1f}%) | "
                  f"Secure: {int(row['Secure_Samples']):6,} ({100-vuln_pct:.1f}%)")
        
        # Show probability distribution analysis
        print("\n--- Probability Distribution Analysis ---")
        prob_info = df[['Model', 'Test Set', 'Prob_Min', 'Prob_Max', 'Prob_Mean', 'Prob_Median']].copy()
        print(prob_info.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
        
        # Show threshold information
        print("\n--- Thresholds Used ---")
        thresh_df = pd.DataFrame(threshold_data)
        # Format threshold column to show more decimal places
        thresh_display = thresh_df.copy()
        thresh_display['Threshold'] = thresh_display['Threshold'].apply(lambda x: f'{x:.6f}')
        print(thresh_display.to_string(index=False))
        
        # Add explanation if thresholds are very low
        avg_threshold = thresh_df['Threshold'].mean()
        if avg_threshold < 0.01:
            print(f"\nℹ️  Note: Very low thresholds (avg={avg_threshold:.6f}) indicate the model outputs")
            print(f"   probabilities compressed near 0 (common with sigmoid). The optimizer adapts")
            print(f"   to find the decision boundary that maximizes recall within the model's range.")
        
        # Pivot tables for mean values
        # Pivot tables for mean values
        rec_mean_pivot = df.pivot(index='Model', columns='Test Set', values='Recall_Mean')
        rec_std_pivot = df.pivot(index='Model', columns='Test Set', values='Recall_Std')
        f2_mean_pivot = df.pivot(index='Model', columns='Test Set', values='F2_Mean')
        f2_std_pivot = df.pivot(index='Model', columns='Test Set', values='F2_Std')
        f1_mean_pivot = df.pivot(index='Model', columns='Test Set', values='F1_Mean')
        f1_std_pivot = df.pivot(index='Model', columns='Test Set', values='F1_Std')
        prec_mean_pivot = df.pivot(index='Model', columns='Test Set', values='Precision_Mean')
        prec_std_pivot = df.pivot(index='Model', columns='Test Set', values='Precision_Std')
        fn_mean_pivot = df.pivot(index='Model', columns='Test Set', values='FalseNegatives_Mean')
        fp_mean_pivot = df.pivot(index='Model', columns='Test Set', values='FalsePositives_Mean')
        
        print("\n--- Recall (Mean ± Std) [SECURITY PRIORITY] ---")
        print(rec_mean_pivot.to_string(float_format=lambda x: f'{x:.4f}'))
        
        print("\n--- F2 Score (Mean ± Std) [Favors Recall 2x] ---")
        print(f2_mean_pivot.to_string(float_format=lambda x: f'{x:.4f}'))
        
        print("\n--- Precision (Mean ± Std) ---")
        print(prec_mean_pivot.to_string(float_format=lambda x: f'{x:.4f}'))
        
        print("\n--- False Negatives (Mean) [Minimize for Security] ---")
        print(fn_mean_pivot.to_string(float_format=lambda x: f'{x:.1f}'))
        
        print("\n--- False Positives (Mean) ---")
        print(fp_mean_pivot.to_string(float_format=lambda x: f'{x:.1f}'))
        
        # Save aggregated results to CSV
        csv_path = os.path.join(models_dir, 'evaluation_matrix_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Summary results saved to: {csv_path}")
        
        # Save threshold information
        thresh_csv_path = os.path.join(models_dir, 'evaluation_thresholds.csv')
        thresh_df.to_csv(thresh_csv_path, index=False)
        print(f"✓ Threshold information saved to: {thresh_csv_path}")
        
        # Save detailed per-run results
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_csv_path = os.path.join(models_dir, 'evaluation_matrix_detailed.csv')
            detailed_df.to_csv(detailed_csv_path, index=False)
            print(f"✓ Detailed per-run results saved to: {detailed_csv_path}")
        
        # Save individual metric pivot tables
        rec_mean_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_recall_mean.csv'))
        rec_std_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_recall_std.csv'))
        f2_mean_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_f2_mean.csv'))
        f2_std_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_f2_std.csv'))
        f1_mean_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_f1_mean.csv'))
        f1_std_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_f1_std.csv'))
        prec_mean_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_precision_mean.csv'))
        prec_std_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_precision_std.csv'))
        fn_mean_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_false_negatives.csv'))
        fp_mean_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_false_positives.csv'))
        print("✓ Metric-specific pivot tables saved")
    
    return results


# Example Usage
if __name__ == "__main__":
    import sys
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Configuration - adjust these paths for your environment
    models_dir = '/content/drive/MyDrive/romeo/models'
    eval_base_dir = '/content/drive/MyDrive/romeo/evaluation'
    
    # Check if running in Colab vs local
    if not os.path.exists(models_dir):
        # Try local paths
        models_dir = 'romeo/models'
        eval_base_dir = 'tensors/TIMESTAMP/evaluation'  # Replace TIMESTAMP
        
        if not os.path.exists(models_dir):
            print("Error: Models directory not found.")
            print("Please update the paths in the script:")
            print(f"  models_dir: {models_dir}")
            print(f"  eval_base_dir: {eval_base_dir}")
            sys.exit(1)
    
    # Run evaluation matrix with auto-thresholding for security
    # For security applications, use auto_threshold=True to minimize false negatives
    results = create_evaluation_matrix(
        models_dir, 
        eval_base_dir, 
        device,
        num_runs=5,
        use_auto_threshold=True,  # Enable dynamic thresholding
        min_recall=0.90  # Require at least 90% recall (adjustable: 0.85-0.95)
    )
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print("\nThresholding Strategy:")
    print("  - Auto-threshold optimizes F2 score (favors recall over precision)")
    print("  - Minimum recall constraint ensures vulnerability detection")
    print("  - Lower threshold = fewer false negatives (missed vulnerabilities)")
    print("  - To use fixed threshold (0.5), set use_auto_threshold=False")
    print("\nRecommendations for security applications:")
    print("  - min_recall=0.95: Very conservative, minimizes missed vulnerabilities")
    print("  - min_recall=0.90: Balanced for security (default)")
    print("  - min_recall=0.85: More precision, but may miss some vulnerabilities")
