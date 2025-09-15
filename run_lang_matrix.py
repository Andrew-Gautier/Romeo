import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import BinaryAUROC
import tqdm
import os
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import sys

# Define constants
BATCH_SIZE = 20
LEARNING_RATE = 0.001
EPOCHS = 20
LSTM_NODES = 256
NUM_SENTENCES = 60
SENTENCE_LENGTH = 64
VOCAB_SIZE = 49152
EMBEDDING_SIZE = 4096

# Configuration for experiment
EXPERIMENT_CONFIGS = {
    'c_only': {
        'data_dir': 'tensors/Experiment_1/c_only_seed42/20250911_182225_seed42',
        'model_name': 'c_only_model',
        'description': 'Model trained on C language only'
    },
    'java_only': {
        'data_dir': 'tensors/Experiment_1/java_only_seed43/20250911_182247_seed43',
        'model_name': 'java_only_model',
        'description': 'Model trained on Java language only'
    },
    'multilingual': {
        'data_dir': 'tensors/Experiment_1/combined_seed46/20250911_182354_seed46',
        'model_name': 'multilingual_model',
        'description': 'Model trained on both C and Java languages'
    }
}

TEST_CONFIGS = {
    'c_test': {
        'test_dir': 'c',
        'description': 'Testing on C language'
    },
    'java_test': {
        'test_dir': 'java',
        'description': 'Testing on Java language'
    },
    'multi_test': {
        'test_dir': 'splits',
        'description': 'Testing on multilingual dataset'
    }
}

def setup_device():
    """Set up the device for training and evaluation"""
    cuda_available = torch.cuda.is_available()
    print("CUDA Available:", cuda_available)
    device = torch.device('cuda' if cuda_available else 'cpu')
    return device

def load_data(data_dir, language_dir, device):
    """Load data for training, validation, and testing"""
    print(f"Loading data from {data_dir}/{language_dir}")
    
    try:
        # Load training data
        train_sequences = torch.load(f"{data_dir}/{language_dir}/train_sequences.pt").long().to(device)
        train_labels = torch.load(f"{data_dir}/{language_dir}/train_labels.pt").to(device)
        
        # Load validation data
        val_sequences = torch.load(f"{data_dir}/{language_dir}/val_sequences.pt").long().to(device)
        val_labels = torch.load(f"{data_dir}/{language_dir}/val_labels.pt").to(device)
        
        # Load test data
        test_sequences = torch.load(f"{data_dir}/{language_dir}/test_sequences.pt").long().to(device)
        test_labels = torch.load(f"{data_dir}/{language_dir}/test_labels.pt").to(device)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(train_sequences, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        
        val_dataset = TensorDataset(val_sequences, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        test_dataset = TensorDataset(test_sequences, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset)
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_multilingual_test_data(data_dir, device):
    """Load multilingual test data from the splits directory"""
    print(f"Loading multilingual test data from {data_dir}/splits")
    
    try:
        # Load test data
        test_sequences = torch.load(f"{data_dir}/splits/cwe_test_sequences.pt").long().to(device)
        test_labels = torch.load(f"{data_dir}/splits/cwe_test_labels.pt").to(device)
        
        # Create dataset and dataloader
        test_dataset = TensorDataset(test_sequences, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        return {
            'test_loader': test_loader,
            'test_size': len(test_dataset)
        }
    except Exception as e:
        print(f"Error loading multilingual test data: {e}")
        return None

def load_pretrained_weights(weight_path, device):
    """Load pretrained weights for the embedding layer"""
    try:
        pretrained_weights = torch.load(weight_path, map_location=device)
        print("Weights loaded successfully.")
        word_vectors = pretrained_weights['tok_embeddings.weight']
        print(f"Word vectors shape: {word_vectors.shape}")
        return word_vectors
    except Exception as e:
        print(f"Failed to load weights: {e}")
        print("Using random initialization instead.")
        return torch.randn(VOCAB_SIZE, EMBEDDING_SIZE)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, batch_first, 
                 bidirectional, dropout, pretrained_weights=None, device='cuda'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                          bidirectional=bidirectional, dropout=dropout, 
                          batch_first=batch_first).to(device)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.attention = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1).to(device)
        
        # Initialize with pretrained weights if available
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights).to(device)

    def forward(self, text):
        # Move input to device
        text = text.to(self.device)
        
        # Handle different input shapes
        if text.dim() == 3:
            batch_size, sentence_length, num_sentences = text.size()
            text = text.view(batch_size, -1)
        else:
            batch_size = text.size(0)
        
        # Embedding layer
        embedded = self.dropout(self.embedding(text))
        
        # LSTM layer
        lstm_output, (hidden, _) = self.rnn(embedded)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)
        
        # Final linear layer with dropout
        output = self.fc(self.dropout(attended_output))
        
        # Apply sigmoid activation
        output = torch.sigmoid(output)
        
        return output

def save_checkpoint(state, epoch, checkpoint_path):
    """Save model checkpoint during training"""
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    filename = os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch}.pth")
    torch.save(state, filename)

def train(model, iterator, optimizer, criterion, epoch, device, checkpoint_path):
    """Train the model for one epoch"""
    epoch_loss = 0
    model.train()
    
    # Start epoch timer
    start_time = time.time()
    
    for batch_sequences, batch_labels in tqdm.tqdm(iterator, desc='Training'):
        # Move data to the device
        batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_sequences)
        
        predictions = predictions.view(-1, NUM_SENTENCES).float()
        batch_labels = batch_labels.view(-1, NUM_SENTENCES).float()

        loss = criterion(predictions, batch_labels)
        loss.backward()
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
    """Evaluate the model on validation or test data"""
    epoch_loss = 0
    model.eval()
    auroc = BinaryAUROC().to(device)
    
    # Start evaluation timer
    start_time = time.time()

    with torch.no_grad():
        for batch_sequences, batch_labels in tqdm.tqdm(iterator, desc='Evaluation'):
            # Move data to the device
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
            
            predictions = model(batch_sequences)
            predictions = predictions.view(-1, NUM_SENTENCES).float()
            batch_labels = batch_labels.view(-1, NUM_SENTENCES).float()
            
            # Update AUROC computation
            auroc.update(predictions, batch_labels.int())
            
            loss = criterion(predictions, batch_labels)
            epoch_loss += loss.item()
    
    # End evaluation timer
    end_time = time.time()
    eval_duration = end_time - start_time
    
    auroc_score = auroc.compute()
    auroc.reset()
    
    return epoch_loss / len(iterator), auroc_score.item(), eval_duration

def plot_training_metrics(train_losses, valid_losses, valid_aurocs, train_times, eval_times, model_name):
    """Create and save plots for training metrics"""
    output_dir = f"plots/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/{model_name}_loss.png")
    plt.close()
    
    # Plot validation AUROC scores
    plt.figure(figsize=(10, 5))
    plt.plot(valid_aurocs, label='Validation AUROC')
    plt.xlabel('Epochs')
    plt.ylabel('AUROC')
    plt.legend()
    plt.ylim(0.5, 1.0)
    plt.savefig(f"{output_dir}/{model_name}_auroc.png")
    plt.close()
    
    # Plot training and evaluation times
    plt.figure(figsize=(10, 5))
    plt.plot(train_times, label='Training Time (s)')
    plt.plot(eval_times, label='Evaluation Time (s)')
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.savefig(f"{output_dir}/{model_name}_time.png")
    plt.close()

def train_model(config_name, weights_path=None, device=None):
    """Train a model based on the given configuration"""
    if device is None:
        device = setup_device()
    
    config = EXPERIMENT_CONFIGS[config_name]
    data_dir = config['data_dir']
    model_name = config['model_name']
    
    # Determine language directory based on configuration
    if config_name == 'c_only':
        language_dir = 'c'
    elif config_name == 'java_only':
        language_dir = 'java'
    else:  # multilingual
        language_dir = 'splits'
    
    # Load data
    data = load_data(data_dir, language_dir, device)
    if data is None:
        print(f"Failed to load data for {config_name}. Skipping.")
        return None
    
    # Load pretrained weights if available
    word_vectors = None
    if weights_path:
        word_vectors = load_pretrained_weights(weights_path, device)
    
    # Create model
    model = LSTMClassifier(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_SIZE,
        hidden_dim=LSTM_NODES,
        output_dim=NUM_SENTENCES,
        n_layers=2,
        batch_first=True,
        bidirectional=True,
        dropout=0.5,
        pretrained_weights=word_vectors,
        device=device
    )
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss().to(device)
    checkpoint_path = f"checkpoints/{model_name}"
    
    # Metrics tracking
    best_valid_loss = float('inf')
    epochs_since_improvement = 0
    train_losses = []
    valid_losses = []
    valid_aurocs = []
    train_times = []
    eval_times = []
    total_start_time = time.time()
    
    print(f"Training {model_name} started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Training loop
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        train_loss, train_duration = train(model, data['train_loader'], optimizer, criterion, epoch, device, checkpoint_path)
        valid_loss, valid_auroc, eval_duration = evaluate(model, data['val_loader'], criterion, device)
        
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
        
        # Early stopping logic
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_since_improvement = 0
            print(f"New best validation loss: {valid_loss:.3f}")
            
            # Save best model
            best_model_path = f"models/{model_name}_best.pt"
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_since_improvement += 1
        
        # Stop training if validation loss hasn't improved for 3 consecutive epochs
        if epochs_since_improvement == 3:
            print("Stopping early due to no improvement in validation loss for 3 consecutive epochs.")
            break
    
    total_training_time = time.time() - total_start_time
    print(f"Total training time for {model_name}: {str(timedelta(seconds=int(total_training_time)))}")
    
    # Plot training metrics
    plot_training_metrics(train_losses, valid_losses, valid_aurocs, train_times, eval_times, model_name)
    
    # Save the final model
    final_model_path = f"models/{model_name}_final.pt"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': VOCAB_SIZE,
            'embedding_dim': EMBEDDING_SIZE,
            'hidden_dim': LSTM_NODES,
            'output_dim': NUM_SENTENCES,
            'n_layers': 2,
            'bidirectional': True,
            'dropout': 0.5,
        },
        'training': {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'valid_aurocs': valid_aurocs,
            'train_times': train_times,
            'eval_times': eval_times,
            'total_training_time': total_training_time,
            'timestamp': datetime.now().isoformat()
        }
    }, final_model_path)
    
    print(f"Model saved to {final_model_path}")
    
    return model

def evaluate_model_on_test_set(model_name, test_config, device=None):
    """Evaluate a trained model on a specific test set"""
    if device is None:
        device = setup_device()
    
    # Load the model
    model_path = f"models/{model_name}_best.pt"
    if not os.path.exists(model_path):
        model_path = f"models/{model_name}_final.pt"
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
    
    # Determine which config this model came from
    for config_name, config in EXPERIMENT_CONFIGS.items():
        if config['model_name'] == model_name:
            model_config = config
            break
    else:
        print(f"Could not find configuration for model {model_name}")
        return None
    
    # Load model and weights
    word_vectors = None  # We'll initialize without pretrained weights
    model = LSTMClassifier(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_SIZE,
        hidden_dim=LSTM_NODES,
        output_dim=NUM_SENTENCES,
        n_layers=2,
        batch_first=True,
        bidirectional=True,
        dropout=0.5,
        pretrained_weights=word_vectors,
        device=device
    )
    
    # Load model state
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model state: {e}")
        return None
    
    # Load test data
    test_dir = TEST_CONFIGS[test_config]['test_dir']
    
    # Handle multilingual test data differently
    if test_config == 'multi_test':
        for config_name, config in EXPERIMENT_CONFIGS.items():
            if config_name == 'multilingual':
                data_dir = config['data_dir']
                test_data = load_multilingual_test_data(data_dir, device)
                break
    else:
        # For C and Java tests, use the data from the respective directories
        test_language = test_config.split('_')[0]  # 'c' or 'java'
        for config_name, config in EXPERIMENT_CONFIGS.items():
            if test_language in config_name:  # Find matching config
                data_dir = config['data_dir']
                test_data = {}
                test_sequences = torch.load(f"{data_dir}/{test_dir}/test_sequences.pt").long().to(device)
                test_labels = torch.load(f"{data_dir}/{test_dir}/test_labels.pt").to(device)
                test_dataset = TensorDataset(test_sequences, test_labels)
                test_data['test_loader'] = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
                test_data['test_size'] = len(test_dataset)
                break
    
    if test_data is None:
        print(f"Failed to load test data for {test_config}")
        return None
    
    # Setup evaluation
    criterion = nn.BCELoss().to(device)
    
    # Evaluate
    print(f"\nEvaluating {model_name} on {test_config} ({TEST_CONFIGS[test_config]['description']})")
    test_start_time = time.time()
    test_loss, test_auroc, test_duration = evaluate(model, test_data['test_loader'], criterion, device)
    
    print(f'Test Loss: {test_loss:.3f}, Test AUROC: {test_auroc:.3f}')
    print(f'Test evaluation time: {str(timedelta(seconds=int(test_duration)))}')
    
    return {
        'model_name': model_name,
        'test_config': test_config,
        'test_loss': test_loss,
        'test_auroc': test_auroc,
        'test_duration': test_duration
    }

def create_performance_matrix(results):
    """Create a performance matrix from evaluation results"""
    # Define matrix structure
    models = ['c_only_model', 'java_only_model', 'multilingual_model']
    test_sets = ['c_test', 'java_test', 'multi_test']
    
    # Initialize empty dataframe
    index = ['C only', 'Java only', 'Multilingual']
    columns = ['Performance on C', 'Performance on Java', 'Performance on Multilingual']
    matrix = pd.DataFrame(index=index, columns=columns)
    
    # Fill the matrix with results
    for result in results:
        model_name = result['model_name']
        test_config = result['test_config']
        auroc = result['test_auroc']
        
        # Map model name to row
        if 'c_only' in model_name:
            row = 'C only'
        elif 'java_only' in model_name:
            row = 'Java only'
        elif 'multilingual' in model_name:
            row = 'Multilingual'
        else:
            continue
        
        # Map test config to column
        if test_config == 'c_test':
            col = 'Performance on C'
        elif test_config == 'java_test':
            col = 'Performance on Java'
        elif test_config == 'multi_test':
            col = 'Performance on Multilingual'
        else:
            continue
        
        # Add result to matrix
        matrix.loc[row, col] = f"{auroc:.3f}"
    
    return matrix

def run_experiment():
    """Run the full experiment - train models and evaluate them"""
    device = setup_device()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Path to pretrained weights (if available)
    weights_path = 'aix3-7b-base (1).pt'
    if not os.path.exists(weights_path):
        print(f"Warning: Pretrained weights not found at {weights_path}. Using random initialization.")
        weights_path = None
    
    # Train models
    print("======= TRAINING MODELS =======")
    models = {}
    for config_name in EXPERIMENT_CONFIGS:
        print(f"\n--- Training {config_name} model ---")
        models[config_name] = train_model(config_name, weights_path, device)
    
    # Evaluate models on all test sets
    print("\n======= EVALUATING MODELS =======")
    results = []
    for config_name, config in EXPERIMENT_CONFIGS.items():
        model_name = config['model_name']
        for test_config in TEST_CONFIGS:
            result = evaluate_model_on_test_set(model_name, test_config, device)
            if result:
                results.append(result)
    
    # Create and display performance matrix
    print("\n======= PERFORMANCE MATRIX =======")
    matrix = create_performance_matrix(results)
    print(matrix)
    
    # Save performance matrix to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    matrix.to_csv(f"results/performance_matrix_{timestamp}.csv")
    
    # Save detailed results
    result_df = pd.DataFrame(results)
    result_df.to_csv(f"results/detailed_results_{timestamp}.csv", index=False)
    
    print(f"\nResults saved to results/performance_matrix_{timestamp}.csv")
    
    return matrix, results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate vulnerability detection models")
    parser.add_argument("--train", choices=['c_only', 'java_only', 'multilingual', 'all'],
                       help="Train a specific model or all models")
    parser.add_argument("--evaluate", action="store_true", 
                       help="Evaluate trained models on all test sets")
    parser.add_argument("--weights", type=str, default='aix3-7b-base (1).pt',
                       help="Path to pretrained weights")
    
    args = parser.parse_args()
    
    if not args.train and not args.evaluate:
        # If no args provided, run the full experiment
        run_experiment()
    else:
        device = setup_device()
        
        # Train specified models
        if args.train:
            if args.train == 'all':
                for config_name in EXPERIMENT_CONFIGS:
                    print(f"\n--- Training {config_name} model ---")
                    train_model(config_name, args.weights, device)
            else:
                print(f"\n--- Training {args.train} model ---")
                train_model(args.train, args.weights, device)
        
        # Evaluate trained models
        if args.evaluate:
            results = []
            for config_name, config in EXPERIMENT_CONFIGS.items():
                model_name = config['model_name']
                for test_config in TEST_CONFIGS:
                    result = evaluate_model_on_test_set(model_name, test_config, device)
                    if result:
                        results.append(result)
            
            # Create and display performance matrix
            if results:
                print("\n======= PERFORMANCE MATRIX =======")
                matrix = create_performance_matrix(results)
                print(matrix)
                
                # Save performance matrix
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("results", exist_ok=True)
                matrix.to_csv(f"results/performance_matrix_{timestamp}.csv")
                print(f"\nResults saved to results/performance_matrix_{timestamp}.csv")