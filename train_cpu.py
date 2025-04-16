import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Import model and dataset
from models.network.temporal_attention import TemporalAttention
from models.data.telomere_dataset import TelomereDataset

def custom_collate(batch):
    """Custom collate function to handle potential None values or empty samples"""
    # Filter out None values
    batch = [item for item in batch if item is not None and 'features' in item and 'labels' in item]
    
    if not batch:
        return {'features': torch.tensor([]), 'labels': torch.tensor([])}
    
    # Extract features and labels
    features = torch.stack([item['features'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {'features': features, 'labels': labels}

def train_model(num_epochs=100, batch_size=32, learning_rate=0.001, weight_decay=0.01):
    """Train the Temporal Attention Network for telomere-based radiosensitivity prediction"""
    # Create directories for saving results
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Set device
    device = torch.device('cpu')
    print(f"Training on: {device}")
    
    # Initialize model
    model = TemporalAttention(input_features=11, embed_dim=64, num_heads=4, dropout_rate=0.3)
    model.to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize dataset
    dataset = TelomereDataset()
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Validate dataset by checking first sample
    try:
        sample = dataset[0]
        print(f"Sample feature shape: {sample['features'].shape}")
    except Exception as e:
        print(f"Warning: Dataset returned invalid sample")
        sample = {'features': torch.zeros(11), 'labels': torch.tensor([0.0])}
    
    # Split dataset
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # Create dataloaders with custom collate function and no workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=0  # Use 0 workers to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0
    )
    
    # Initialize optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    criterion = torch.nn.BCELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=False
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Training start time
    start_time = datetime.now()
    print(f"Training started at: {start_time}")
    
    # Training loop
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_batches = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch in progress_bar:
                features = batch['features']
                labels = batch['labels']
                
                # Skip empty batches
                if features.numel() == 0 or features.size(0) == 0:
                    print("Warning: Empty batch encountered, skipping")
                    continue
                
                # Move to device
                features = features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Calculate average loss (handle case where all batches are skipped)
            avg_train_loss = train_loss / max(1, train_batches)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_batches = 0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features']
                    labels = batch['labels']
                    
                    # Skip empty batches
                    if features.numel() == 0 or features.size(0) == 0:
                        print("Warning: Empty batch encountered, skipping")
                        continue
                    
                    # Move to device
                    features = features.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    predicted = (outputs > 0.5).float()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            # Calculate metrics (handle edge cases with empty arrays)
            avg_val_loss = val_loss / max(1, val_batches)
            accuracy = 100 * correct / max(1, total)
            
            # Calculate F1 score
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            if len(all_preds) > 0 and len(all_labels) > 0:
                true_positives = np.sum((all_preds == 1) & (all_labels == 1))
                false_positives = np.sum((all_preds == 1) & (all_labels == 0))
                false_negatives = np.sum((all_preds == 0) & (all_labels == 1))
                
                precision = true_positives / (true_positives + false_positives + 1e-10)
                recall = true_positives / (true_positives + false_negatives + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
            else:
                f1 = 0.0
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(accuracy)
            history['val_f1'].append(f1)
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%, F1: {f1:.4f}')
            
            # Early stopping and checkpoint saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, 'checkpoints/best_model.pt')
                print(f"✅ Model saved: best validation loss {avg_val_loss:.4f}")
            else:
                patience_counter += 1
                
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, f'checkpoints/model_epoch_{epoch+1}.pt')
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    except Exception as e:
        print(f"Training interrupted: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Training end time
    end_time = datetime.now()
    training_duration = end_time - start_time
    print(f"Training completed in: {training_duration}")
    
    # Generate plots and save model summary
    try:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_accuracy'], label='Accuracy')
        plt.plot(history['val_f1'], label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Performance Metrics')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/training_history.png')
        plt.close()
        print("✅ Training plots saved to plots/training_history.png")
    except Exception as e:
        print(f"Error creating plots: {str(e)}")
    
    return history

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train the model
    train_model(num_epochs=100, batch_size=32, learning_rate=0.001, weight_decay=0.01)
