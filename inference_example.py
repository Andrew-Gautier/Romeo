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


def evaluate_model(model, sequences, labels, device='cuda', batch_size=32, threshold=0.5):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained LSTM model
        sequences: Test sequences [num_samples, seq_length]
        labels: True labels [num_samples]
        device: Device for evaluation
        batch_size: Batch size
        threshold: Classification threshold
    
    Returns:
        metrics: Dict with accuracy, precision, recall, f1
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    probabilities, predictions = predict_sequences(model, sequences, device, batch_size, threshold)
    
    labels_np = labels.cpu().numpy()
    predictions_np = predictions.numpy()
    
    metrics = {
        'accuracy': accuracy_score(labels_np, predictions_np),
        'precision': precision_score(labels_np, predictions_np, zero_division=0),
        'recall': recall_score(labels_np, predictions_np, zero_division=0),
        'f1': f1_score(labels_np, predictions_np, zero_division=0)
    }
    
    return metrics


def evaluate_model_multirun(model, sequences, labels, device='cuda', batch_size=32, num_runs=5, seed_start=42):
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
    
    Returns:
        stats: Dict with mean, std, min, max for each metric
        all_results: List of metric dicts from each run
    """
    import numpy as np
    
    all_results = []
    
    for run_idx in range(num_runs):
        seed = seed_start + run_idx
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Vary threshold slightly for each run to simulate different decision boundaries
        threshold = 0.5
        
        metrics = evaluate_model(model, sequences, labels, device, batch_size, threshold)
        all_results.append(metrics)
    
    # Calculate statistics across runs
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
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
    
    return stats, all_results


def create_evaluation_matrix(models_dir, eval_base_dir, device='cuda', num_runs=5, seed_start=42):
    """
    Create evaluation matrix: test all models on C and Python test sets with multiple runs.
    
    Args:
        models_dir: Directory containing trained models
        eval_base_dir: Base directory for evaluation data
        device: Device for evaluation
        num_runs: Number of evaluation runs per model-dataset pair
        seed_start: Starting seed for random evaluations
    
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
    
    print("="*80)
    print("EVALUATION MATRIX: Models vs Test Sets")
    print(f"Running {num_runs} evaluations per model-dataset pair")
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
            
            print(f"    Samples: {len(test_sequences)}, Positive: {test_labels.sum().item()}")
            
            # Evaluate with multiple runs
            try:
                stats, all_results = evaluate_model_multirun(
                    model, test_sequences, test_labels, device, 
                    batch_size=32, num_runs=num_runs, seed_start=seed_start
                )
                results[model_name][test_name] = stats
                
                print(f"    Accuracy:  {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f} (range: {stats['accuracy']['range']:.4f})")
                print(f"    Precision: {stats['precision']['mean']:.4f} ± {stats['precision']['std']:.4f} (range: {stats['precision']['range']:.4f})")
                print(f"    Recall:    {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f} (range: {stats['recall']['range']:.4f})")
                print(f"    F1:        {stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f} (range: {stats['f1']['range']:.4f})")
                
                # Store aggregated results for matrix
                matrix_data.append({
                    'Model': model_name,
                    'Test Set': test_name,
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
                    'F1_Range': stats['f1']['range']
                })
                
                # Store detailed per-run results
                for run_idx, run_results in enumerate(all_results):
                    detailed_data.append({
                        'Model': model_name,
                        'Test Set': test_name,
                        'Run': run_idx + 1,
                        'Seed': seed_start + run_idx,
                        'Accuracy': run_results['accuracy'],
                        'Precision': run_results['precision'],
                        'Recall': run_results['recall'],
                        'F1': run_results['f1']
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
        
        # Pivot tables for mean values
        print("\n--- Accuracy (Mean ± Std) ---")
        acc_mean_pivot = df.pivot(index='Model', columns='Test Set', values='Accuracy_Mean')
        acc_std_pivot = df.pivot(index='Model', columns='Test Set', values='Accuracy_Std')
        print(acc_mean_pivot.to_string(float_format=lambda x: f'{x:.4f}'))
        
        print("\n--- Precision (Mean ± Std) ---")
        prec_mean_pivot = df.pivot(index='Model', columns='Test Set', values='Precision_Mean')
        prec_std_pivot = df.pivot(index='Model', columns='Test Set', values='Precision_Std')
        print(prec_mean_pivot.to_string(float_format=lambda x: f'{x:.4f}'))
        
        print("\n--- Recall (Mean ± Std) ---")
        rec_mean_pivot = df.pivot(index='Model', columns='Test Set', values='Recall_Mean')
        rec_std_pivot = df.pivot(index='Model', columns='Test Set', values='Recall_Std')
        print(rec_mean_pivot.to_string(float_format=lambda x: f'{x:.4f}'))
        
        print("\n--- F1 Score (Mean ± Std) ---")
        f1_mean_pivot = df.pivot(index='Model', columns='Test Set', values='F1_Mean')
        f1_std_pivot = df.pivot(index='Model', columns='Test Set', values='F1_Std')
        print(f1_mean_pivot.to_string(float_format=lambda x: f'{x:.4f}'))
        
        # Create combined tables with mean ± std format
        print("\n--- Combined Statistics ---")
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
            print(f"\n{metric}:")
            mean_col = f'{metric}_Mean'
            std_col = f'{metric}_Std'
            range_col = f'{metric}_Range'
            
            for model in df['Model'].unique():
                print(f"  {model}:")
                for test_set in df['Test Set'].unique():
                    row = df[(df['Model'] == model) & (df['Test Set'] == test_set)]
                    if not row.empty:
                        mean_val = row[mean_col].values[0]
                        std_val = row[std_col].values[0]
                        range_val = row[range_col].values[0]
                        print(f"    {test_set}: {mean_val:.4f} ± {std_val:.4f} (range: {range_val:.4f})")
        
        # Save aggregated results to CSV
        csv_path = os.path.join(models_dir, 'evaluation_matrix_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Summary results saved to: {csv_path}")
        
        # Save detailed per-run results
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_csv_path = os.path.join(models_dir, 'evaluation_matrix_detailed.csv')
            detailed_df.to_csv(detailed_csv_path, index=False)
            print(f"✓ Detailed per-run results saved to: {detailed_csv_path}")
        
        # Save individual metric pivot tables
        acc_mean_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_accuracy_mean.csv'))
        acc_std_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_accuracy_std.csv'))
        f1_mean_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_f1_mean.csv'))
        f1_std_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_f1_std.csv'))
        print("✓ Metric-specific pivot tables saved")
    
    return results


# Example Usage
if __name__ == "__main__":
    import sys
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Configuration - adjust these paths for your environment
    models_dir = '/content/drive/MyDrive/romeo/10k/model'
    eval_base_dir = '/content/drive/MyDrive/romeo/10k/evaluation'
    
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
    
    # Run evaluation matrix
    results = create_evaluation_matrix(models_dir, eval_base_dir, device)
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
