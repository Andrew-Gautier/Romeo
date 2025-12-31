
"""
Google Colab script for training LSTM vulnerability detection models on pretraining data.
Trains separate models for C, Java, C#, and combined language datasets.

SETUP INSTRUCTIONS FOR GOOGLE COLAB:
1. Mount your Google Drive:
   from google.colab import drive
   drive.mount('/content/drive')

2. Upload your pretraining tensors to Google Drive in this structure:
   /content/drive/MyDrive/romeo/pretraining/
   ├── c/
   │   ├── train/
   │   │   ├── sequences.pt
   │   │   └── labels.pt
   │   └── validation/
   │       ├── sequences.pt
   │       └── labels.pt
   ├── java/
   │   ├── train/
   │   └── validation/
   ├── csharp/
   │   ├── train/
   │   └── validation/
   └── combined/
       ├── train/
       └── validation/

3. Install required packages:
   !pip install transformers torchmetrics

4. Run this script:
   !python load_and_predict.py

OUTPUT:
- Trained models will be saved to: /content/drive/MyDrive/romeo/models/
- Each model includes: state_dict, losses, AUROC scores, and config
- Training curves plots saved as PNG files
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchmetrics.classification import BinaryAUROC
import tqdm
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Constants
BATCH_SIZE = 32
VOCAB_SIZE = 49152
EMBEDDING_SIZE = 4096
LSTM_NODES = 256
OUTPUT_DIM = 1
LEARNING_RATE = 0.001
EPOCHS = 30
PATIENCE = 8  # Early stopping patience
GRADIENT_CLIP = 1.0  # Gradient clipping threshold

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load tokenizer and extract embeddings from HuggingFace model
print("Loading pretrained embeddings from aiXcoder...")
tokenizer = AutoTokenizer.from_pretrained("aiXcoder/aixcoder-7b-base")
hf_model = AutoModelForCausalLM.from_pretrained("aiXcoder/aixcoder-7b-base")

# Extract token embeddings
word_vectors = hf_model.model.embed_tokens.weight.data.clone()
print(f"Loaded embeddings shape: {word_vectors.shape}")

# Clean up HuggingFace model to save memory
del hf_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Paths to pretraining data (adjust these for your Colab setup)
c_pretraining_path = '/content/drive/MyDrive/romeo/10k/pretraining/c'
java_pretraining_path = '/content/drive/MyDrive/romeo/10k/pretraining/java'
csharp_pretraining_path = '/content/drive/MyDrive/romeo/10k/pretraining/csharp'
combined_path = '/content/drive/MyDrive/romeo/10k/pretraining//combined'
output_path = '/content/drive/MyDrive/romeo/10k/model'

# Create output directory
os.makedirs(output_path, exist_ok=True)


class LSTMClassifier(nn.Module):
    """LSTM-based binary classifier for vulnerability detection with attention."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 batch_first, bidirectional, dropout, pretrained_weights):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                          bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Load pretrained embeddings
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


def load_tensors(base_path, split='train'):
    """Load sequences and labels from a given path."""
    sequences_path = os.path.join(base_path, split, 'sequences.pt')
    labels_path = os.path.join(base_path, split, 'labels.pt')
    
    if not os.path.exists(sequences_path):
        raise FileNotFoundError(f"Sequences not found at {sequences_path}")
    
    sequences = torch.load(sequences_path)
    labels = torch.load(labels_path)
    
    print(f"  Loaded {split} - Sequences: {sequences.shape}, Labels: {labels.shape}")
    print(f"  Positive samples: {labels.sum().item()}/{len(labels)} ({labels.sum().item()/len(labels)*100:.2f}%)")
    
    return sequences, labels


def create_dataloaders(train_seq, train_labels, val_seq, val_labels, batch_size=32):
    """Create train and validation dataloaders."""
    train_dataset = TensorDataset(train_seq.long(), train_labels)
    val_dataset = TensorDataset(val_seq.long(), val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader


def train_epoch(model, iterator, optimizer, criterion, device, gradient_clip=None):
    """Train for one epoch with optional gradient clipping."""
    epoch_loss = 0
    model.train()
    
    for batch_sequences, batch_labels in tqdm.tqdm(iterator, desc='Training'):
        batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_sequences).squeeze(1)
        batch_labels = batch_labels.float()
        
        loss = criterion(predictions, batch_labels)
        loss.backward()
        
        # Apply gradient clipping if specified
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    """Evaluate model and compute loss and AUROC."""
    epoch_loss = 0
    model.eval()
    auroc = BinaryAUROC().to(device)
    
    with torch.no_grad():
        for batch_sequences, batch_labels in tqdm.tqdm(iterator, desc='Evaluating'):
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
            
            predictions = model(batch_sequences).squeeze(1)
            batch_labels = batch_labels.float()
            
            auroc.update(predictions, batch_labels.int())
            loss = criterion(predictions, batch_labels)
            epoch_loss += loss.item()
    
    auroc_score = auroc.compute()
    return epoch_loss / len(iterator), auroc_score.item()


def train_model(language_name, data_path, word_vectors, device, epochs=EPOCHS, patience=PATIENCE, gradient_clip=GRADIENT_CLIP):
    """Train a model for a specific language dataset with gradient clipping and configurable patience."""
    print("\n" + "="*80)
    print(f"Training model for: {language_name.upper()}")
    print("="*80)
    print(f"Configuration: Epochs={epochs}, Patience={patience}, Gradient Clip={gradient_clip}")
    
    # Load data
    print(f"Loading data from {data_path}...")
    train_seq, train_labels = load_tensors(data_path, 'train')
    val_seq, val_labels = load_tensors(data_path, 'validation')
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_seq, train_labels, val_seq, val_labels, BATCH_SIZE)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize model
    model = LSTMClassifier(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_SIZE,
        hidden_dim=LSTM_NODES,
        output_dim=OUTPUT_DIM,
        n_layers=2,
        batch_first=True,
        bidirectional=True,
        dropout=0.5,
        pretrained_weights=word_vectors
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    best_valid_loss = float('inf')
    epochs_since_improvement = 0
    train_losses = []
    valid_losses = []
    valid_aurocs = []
    
    print(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, gradient_clip)
        valid_loss, valid_auroc = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_aurocs.append(valid_auroc)
        
        epoch_time = time.time() - epoch_start
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | Val AUROC: {valid_auroc:.4f} | Time: {epoch_time:.1f}s')
        
        # Early stopping with configurable patience
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_since_improvement = 0
            print(f"  ✓ New best validation loss!")
        else:
            epochs_since_improvement += 1
            print(f"  No improvement for {epochs_since_improvement}/{patience} epochs")
        
        if epochs_since_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience} reached)")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {str(timedelta(seconds=int(total_time)))}")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    
    # Save model
    model_path = os.path.join(output_path, f'{language_name}_lstm.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_loss': best_valid_loss,
        'final_val_auroc': valid_aurocs[-1],
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'valid_aurocs': valid_aurocs,
        'training_time': total_time,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'vocab_size': VOCAB_SIZE,
            'embedding_dim': EMBEDDING_SIZE,
            'hidden_dim': LSTM_NODES,
            'output_dim': OUTPUT_DIM,
            'n_layers': 2,
            'bidirectional': True,
            'dropout': 0.5,
            'language': language_name,
            'patience': patience,
            'gradient_clip': gradient_clip,
            'learning_rate': LEARNING_RATE
        }
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Plot training curves
    plot_path = os.path.join(output_path, f'{language_name}_training_curves.png')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(valid_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{language_name.upper()} - Training & Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(valid_aurocs, label='Val AUROC', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUROC')
    ax2.set_title(f'{language_name.upper()} - Validation AUROC')
    ax2.set_ylim(0.5, 1.0)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {plot_path}")
    
    return model, train_losses, valid_losses, valid_aurocs


def main():
    """Main training loop for all language datasets."""
    print("="*80)
    print("LSTM Vulnerability Detection - Pretraining on Multiple Languages")
    print("="*80)
    
    # Define datasets to train on
    datasets = [
        ('c', c_pretraining_path),
        ('java', java_pretraining_path),
        ('csharp', csharp_pretraining_path),
        ('combined', combined_path)
    ]
    
    results = {}
    
    for lang_name, lang_path in datasets:
        if not os.path.exists(lang_path):
            print(f"\nSkipping {lang_name} - path not found: {lang_path}")
            continue
        
        try:
            model, train_losses, val_losses, val_aurocs = train_model(
                lang_name, lang_path, word_vectors, device, EPOCHS
            )
            
            results[lang_name] = {
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'final_val_auroc': val_aurocs[-1],
                'best_val_auroc': max(val_aurocs)
            }
            
            # Clear GPU memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"\nError training {lang_name} model: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    for lang, metrics in results.items():
        print(f"\n{lang.upper()}:")
        print(f"  Final Train Loss: {metrics['final_train_loss']:.4f}")
        print(f"  Final Val Loss: {metrics['final_val_loss']:.4f}")
        print(f"  Final Val AUROC: {metrics['final_val_auroc']:.4f}")
        print(f"  Best Val AUROC: {metrics['best_val_auroc']:.4f}")
    
    print("\n" + "="*80)
    print(f"All models saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()