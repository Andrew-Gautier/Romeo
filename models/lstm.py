import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import BinaryAUROC
from torch.optim import Adam
import tqdm
import os
import time
from datetime import datetime, timedelta

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
LSTM_NODES = 256
MAX_SEQ_LENGTH = 4096
VOCAB_SIZE = 49152
EMBEDDING_SIZE = 4096
OUTPUT_DIM = 1  # Binary classification: vulnerable (1) or secure (0)
PATIENCE = 5  # Early stopping patience
GRADIENT_CLIP = 1.0  # Gradient clipping threshold


# Make sure tensors are all on GPU
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)
device = torch.device('cuda')

train_sequences_tensor = torch.load("c/train_sequences.pt").long()
train_labels = torch.load("c/train_labels.pt") 
train_dataset = TensorDataset(train_sequences_tensor, train_labels)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last = True)

val_sequences_tensor = torch.load('c/val_sequences.pt').long()
val_labels = torch.load('c/val_labels.pt') 
val_dataset = TensorDataset(val_sequences_tensor, val_labels)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, drop_last = False)

test_sequences_tensor = torch.load("c/test_sequences.pt").long()
test_labels = torch.load("c/test_labels.pt") 
test_dataset = TensorDataset(test_sequences_tensor, test_labels)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, drop_last = False)

torch.manual_seed(691)

## Commented out code for HPC loading of weights
try:
    pretrained_weights = torch.load('aix3-7b-base (1).pt')
    print("Weights loaded successfully.")
except Exception as e:
    print(f"Failed to load weights: {e}")

## Parse word embeddings from the loaded weights 

word_vectors = pretrained_weights['tok_embeddings.weight']
print(word_vectors.shape)

def save_checkpoint(state, epoch, checkpoint_path="/c_only_check_point_path"):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    filename = os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch}.pth")
    torch.save(state, filename)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, batch_first, bidirectional, dropout, pretrained_weights):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=batch_first).to(device)
        self.fc = nn.Linear(hidden_dim * 2, output_dim).to(device)  # Adjusted for bidirectional
        self.dropout = nn.Dropout(dropout).to(device)
        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1).to(device)
        self.embedding.weight.data.copy_(pretrained_weights).to(device)

    def forward(self, text):
        # text = [batch size, sequence length]
        text = text.to(device)
        
        # Ensure text is 2D: [batch_size, seq_length]
        if text.dim() == 3:
            batch_size, seq_length, _ = text.size()
            text = text.view(batch_size, -1)
        
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, seq length, embedding dim]
        
        lstm_output, (hidden, _) = self.rnn(embedded)
        # lstm_output = [batch size, seq length, hidden dim * num directions]
        
        # Apply attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights = [batch size, seq length, 1]
        
        # Perform weighted sum of LSTM outputs using attention weights
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)
        # attended_output = [batch size, hidden dim * num directions]
        
        output = self.fc(self.dropout(attended_output))
        # output = [batch size, 1]
        
        # Apply sigmoid activation for binary classification
        output = torch.sigmoid(output)
        
        return output

model = LSTMClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_SIZE,  
    hidden_dim=LSTM_NODES,
    output_dim=OUTPUT_DIM,  # Binary classification
    n_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.5,
    pretrained_weights=word_vectors
)
print(model)

model.eval()

with torch.no_grad():
    # Get the first batch of data from the training loader
    batch_sequences, batch_labels = next(iter(train_loader))
        
    # Pass the batch of sequences through the model
    outputs = model(batch_sequences).to(device)
    
def train(model, iterator, optimizer, criterion, epoch, device, checkpoint_path="c_only_checkpoints", gradient_clip=GRADIENT_CLIP):
    epoch_loss = 0
    model.train()
    
    # Start epoch timer
    start_time = time.time()
    
    for batch_sequences, batch_labels in tqdm.tqdm(iterator, desc='Training'):
        # Move data to the device
        batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_sequences)
        
        # Reshape for binary classification
        predictions = predictions.squeeze(1)  # Remove the extra dimension: [batch_size, 1] -> [batch_size]
        batch_labels = batch_labels.float()  # Ensure labels are float for BCELoss

        loss = criterion(predictions, batch_labels)
        loss.backward()
        
        # Apply gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    # End epoch timer
    end_time = time.time()
    epoch_duration = end_time - start_time
    
    # Save checkpoint after the epoch
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': epoch_loss / len(iterator),
        'duration': epoch_duration,
    }, epoch, checkpoint_path=checkpoint_path)    
    
    return epoch_loss / len(iterator), epoch_duration

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    model.eval()
    auroc = BinaryAUROC().to(device)  # Initialize AUROC metric
    
    # Start evaluation timer
    start_time = time.time()

    with torch.no_grad():
        for batch_sequences, batch_labels in tqdm.tqdm(iterator, desc='Evaluation'):
            # Move data to the device
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
            
            predictions = model(batch_sequences)
            
            # Reshape for binary classification
            predictions = predictions.squeeze(1)  # [batch_size, 1] -> [batch_size]
            batch_labels = batch_labels.float()
            
            # Predictions are already probabilities (sigmoid applied in forward)
            # Update AUROC computation
            auroc.update(predictions, batch_labels.int())
            
            loss = criterion(predictions, batch_labels)
            epoch_loss += loss.item()
    
    # End evaluation timer
    end_time = time.time()
    eval_duration = end_time - start_time
    
    auroc_score = auroc.compute()  # Compute the final AUROC score
    auroc.reset()  # Reset AUROC metric for future use
    
    return epoch_loss / len(iterator), auroc_score.item(), eval_duration

# MAIN TRAINING LOOP


optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCELoss().to(device)

# Implement a basic early stopping counter
best_valid_loss = float('inf')
epochs_since_improvement = 0

# Store the loss values for plotting
train_losses = []
valid_losses = []
valid_aurocs = []

# Store timing information
train_times = []
eval_times = []
total_start_time = time.time()

print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Configuration: Epochs={EPOCHS}, Patience={PATIENCE}, Gradient Clip={GRADIENT_CLIP}")

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    
    train_loss, train_duration = train(model, train_loader, optimizer, criterion, epoch, device, gradient_clip=GRADIENT_CLIP)
    valid_loss, valid_auroc, eval_duration = evaluate(model, val_loader, criterion, device)
    
    epoch_total_duration = time.time() - epoch_start_time
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_aurocs.append(valid_auroc)
    train_times.append(train_duration)
    eval_times.append(eval_duration)
    
    # Format times as human-readable strings
    train_time_str = str(timedelta(seconds=int(train_duration)))
    eval_time_str = str(timedelta(seconds=int(eval_duration)))
    epoch_time_str = str(timedelta(seconds=int(epoch_total_duration)))
    total_time_str = str(timedelta(seconds=int(time.time() - total_start_time)))
    
    print(f'Epoch: {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.3f}, Val. Loss: {valid_loss:.3f}, Val. AUROC: {valid_auroc:.3f}')
    print(f'Time - Train: {train_time_str}, Eval: {eval_time_str}, Epoch: {epoch_time_str}, Total: {total_time_str}')
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        epochs_since_improvement = 0  # Reset counter
        print(f"✓ New best validation loss: {valid_loss:.3f}")
    else:
        epochs_since_improvement += 1  # Increment counter
        print(f"  No improvement for {epochs_since_improvement}/{PATIENCE} epochs")
    
    # Stop training if validation loss hasn't improved for PATIENCE consecutive epochs
    if epochs_since_improvement >= PATIENCE:
        print(f"Stopping early due to no improvement in validation loss for {PATIENCE} consecutive epochs.")
        break

total_training_time = time.time() - total_start_time
print(f"Total training time: {str(timedelta(seconds=int(total_training_time)))}")

# Calculate and print average times
avg_train_time = sum(train_times) / len(train_times)
avg_eval_time = sum(eval_times) / len(eval_times)
print(f"Average time per epoch - Training: {str(timedelta(seconds=int(avg_train_time)))}, Evaluation: {str(timedelta(seconds=int(avg_eval_time)))}")

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('C_only_loss.png')
plt.close()

# Optionally, plot the validation AUROC scores
plt.figure(figsize=(10, 5))
plt.plot(valid_aurocs, label='Validation AUROC')
plt.xlabel('Epochs')
plt.ylabel('AUROC')
plt.legend()
plt.ylim(0.5, 1.0)  # Set the y-axis to scale between 0.5 and 1
plt.xlim(0, 20)  # Set the x-axis to scale between 0 and 20
plt.savefig('C_only_auroc.png')
plt.close()

# Plot training and evaluation times
plt.figure(figsize=(10, 5))
plt.plot(train_times, label='Training Time (s)')
plt.plot(eval_times, label='Evaluation Time (s)')
plt.xlabel('Epochs')
plt.ylabel('Time (seconds)')
plt.legend()
plt.savefig('C_only_time.png')
plt.close()

# Evaluate the model on the test dataset
print("\n--- Test Set Evaluation ---")
test_start_time = time.time()
test_loss, test_auroc, test_duration = evaluate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.3f}, Test AUROC: {test_auroc:.3f}')
print(f'Test evaluation time: {str(timedelta(seconds=int(test_duration)))}')

# Save the final model
final_model_path = 'C_only.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'test_loss': test_loss,
    'test_auroc': test_auroc,
    'training_time': total_training_time,
    'test_time': test_duration,
    'train_times': train_times,
    'eval_times': eval_times,
    'timestamp': datetime.now().isoformat(),
    'config': {
        'vocab_size': VOCAB_SIZE,
        'embedding_dim': EMBEDDING_SIZE,
        'hidden_dim': LSTM_NODES,
        'output_dim': OUTPUT_DIM,
        'n_layers': 2,
        'bidirectional': True,
        'dropout': 0.5,
        'patience': PATIENCE,
        'gradient_clip': GRADIENT_CLIP,
        'learning_rate': LEARNING_RATE,
    }
}, final_model_path)
print(f'Model saved to {final_model_path}')