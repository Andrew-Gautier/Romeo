"""
Quick reference for loading and using trained LSTM models.
Use this after training models with load_and_predict.py
"""

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


# Example Usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example 1: Load a trained model
    print("="*60)
    print("Example 1: Loading Trained Model")
    print("="*60)
    
    model_path = '/content/drive/MyDrive/romeo/models/c_lstm.pt'
    model, checkpoint = load_trained_model(model_path, device)
    
    # Example 2: Load test data and make predictions
    print("\n" + "="*60)
    print("Example 2: Making Predictions")
    print("="*60)
    
    # Load test sequences (adjust path)
    test_sequences = torch.load('/content/drive/MyDrive/romeo/pretraining/c/validation/sequences.pt')
    test_labels = torch.load('/content/drive/MyDrive/romeo/pretraining/c/validation/labels.pt')
    
    print(f"Test set size: {len(test_sequences)}")
    
    # Make predictions
    probabilities, predictions = predict_sequences(model, test_sequences, device)
    
    print(f"\nPrediction distribution:")
    print(f"  Secure (0): {(predictions == 0).sum().item()}")
    print(f"  Vulnerable (1): {(predictions == 1).sum().item()}")
    
    # Example 3: Evaluate model
    print("\n" + "="*60)
    print("Example 3: Model Evaluation")
    print("="*60)
    
    metrics = evaluate_model(model, test_sequences, test_labels, device)
    
    print("\nTest Set Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    
    # Example 4: Predict on single sample
    print("\n" + "="*60)
    print("Example 4: Single Sample Prediction")
    print("="*60)
    
    sample_sequence = test_sequences[0:1]  # First sample
    sample_label = test_labels[0].item()
    
    prob, pred = predict_sequences(model, sample_sequence, device)
    
    print(f"True Label: {sample_label} ({'Vulnerable' if sample_label == 1 else 'Secure'})")
    print(f"Predicted:  {pred.item()} ({'Vulnerable' if pred.item() == 1 else 'Secure'})")
    print(f"Confidence: {prob.item():.4f}")
    print(f"Correct: {'✓' if pred.item() == sample_label else '✗'}")
