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


def evaluate_model(model, sequences, labels, device='cuda', batch_size=32):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained LSTM model
        sequences: Test sequences [num_samples, seq_length]
        labels: True labels [num_samples]
        device: Device for evaluation
        batch_size: Batch size
    
    Returns:
        metrics: Dict with accuracy, precision, recall, f1, auroc
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    probabilities, predictions = predict_sequences(model, sequences, device, batch_size)
    
    labels_np = labels.cpu().numpy()
    predictions_np = predictions.numpy()
    probs_np = probabilities.numpy()
    
    metrics = {
        'accuracy': accuracy_score(labels_np, predictions_np),
        'precision': precision_score(labels_np, predictions_np, zero_division=0),
        'recall': recall_score(labels_np, predictions_np, zero_division=0),
        'f1': f1_score(labels_np, predictions_np, zero_division=0),
        'auroc': roc_auc_score(labels_np, probs_np)
    }
    
    return metrics


def create_evaluation_matrix(models_dir, eval_base_dir, device='cuda'):
    """
    Create evaluation matrix: test all models on C and Python test sets.
    
    Args:
        models_dir: Directory containing trained models
        eval_base_dir: Base directory for evaluation data
        device: Device for evaluation
    
    Returns:
        results: Dict with evaluation metrics for each model-testset pair
    """
    import os
    import pandas as pd
    
    # Find all model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_lstm.pt')]
    
    # Test set configurations
    test_sets = {
        'C': os.path.join(eval_base_dir, 'c', 'test'),
        'Python': os.path.join(eval_base_dir, 'python', 'test')
    }
    
    results = {}
    matrix_data = []
    
    print("="*80)
    print("EVALUATION MATRIX: Models vs Test Sets")
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
            
            # Evaluate
            try:
                metrics = evaluate_model(model, test_sequences, test_labels, device)
                results[model_name][test_name] = metrics
                
                print(f"    Accuracy:  {metrics['accuracy']:.4f}")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1:        {metrics['f1']:.4f}")
                print(f"    AUROC:     {metrics['auroc']:.4f}")
                
                # Store for matrix
                matrix_data.append({
                    'Model': model_name,
                    'Test Set': test_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1': metrics['f1'],
                    'AUROC': metrics['auroc']
                })
                
            except Exception as e:
                print(f"    ✗ Error evaluating: {e}")
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
        
        # Pivot tables for each metric
        print("\n--- AUROC Scores ---")
        auroc_pivot = df.pivot(index='Model', columns='Test Set', values='AUROC')
        print(auroc_pivot.to_string(float_format=lambda x: f'{x:.4f}'))
        
        print("\n--- Accuracy Scores ---")
        acc_pivot = df.pivot(index='Model', columns='Test Set', values='Accuracy')
        print(acc_pivot.to_string(float_format=lambda x: f'{x:.4f}'))
        
        print("\n--- F1 Scores ---")
        f1_pivot = df.pivot(index='Model', columns='Test Set', values='F1')
        print(f1_pivot.to_string(float_format=lambda x: f'{x:.4f}'))
        
        # Save to CSV
        csv_path = os.path.join(models_dir, 'evaluation_matrix.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Full results saved to: {csv_path}")
        
        # Save pivot tables
        auroc_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_auroc.csv'))
        acc_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_accuracy.csv'))
        f1_pivot.to_csv(os.path.join(models_dir, 'evaluation_matrix_f1.csv'))
        print("✓ Pivot tables saved")
    
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
    
    # Run evaluation matrix
    results = create_evaluation_matrix(models_dir, eval_base_dir, device)
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
