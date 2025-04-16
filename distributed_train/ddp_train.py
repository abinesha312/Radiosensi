import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.network.temporal_attention import TemporalAttention
from models.data.telomere_dataset import TelomereDataset

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29400'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train(rank, world_size, args):
    """
    Run training process on a single GPU.
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
        args (argparse.Namespace): Command line arguments
    """
    setup(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Print basic info
    if rank == 0:
        print(f"Training with {world_size} GPU(s)")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        start_time = datetime.now()
        print(f"Training started at: {start_time}")
    
    # Create model and move to GPU
    model = TemporalAttention(
        input_features=11, 
        embed_dim=64, 
        num_heads=4,
        dropout_rate=0.3
    )
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Create dataset
    dataset = TelomereDataset(download_kaggle=False)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use fixed seeds for reproducible splits across processes
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=42
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Loss function
    criterion = torch.nn.BCELoss()
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=args.epochs // 3, 
        T_mult=2
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Set epoch for samplers
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Average loss across all processes
        train_loss = train_loss / len(train_loader)
        dist.all_reduce(torch.tensor(train_loss).to(device), op=dist.ReduceOp.SUM)
        train_loss = train_loss / world_size
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Sum values across processes
        val_loss = val_loss / len(val_loader)
        dist.all_reduce(torch.tensor(val_loss).to(device), op=dist.ReduceOp.SUM)
        val_loss = val_loss / world_size
        
        val_correct_tensor = torch.tensor(val_correct).to(device)
        val_total_tensor = torch.tensor(val_total).to(device)
        
        dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)
        
        val_accuracy = 100.0 * val_correct_tensor.item() / val_total_tensor.item()
        
        # Update learning rate
        scheduler.step()
        
        # Log only on rank 0
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.2f}%, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }
                
                os.makedirs('../checkpoints', exist_ok=True)
                torch.save(checkpoint, '../checkpoints/best_model_distributed.pt')
                print(f"✅ Model saved with val_loss: {val_loss:.4f}")
    
    if rank == 0:
        end_time = datetime.now()
        training_duration = end_time - start_time
        print(f"Training completed in: {training_duration}")
    
    # Clean up
    cleanup()

def main():
    """Parse command line arguments and start distributed training."""
    parser = argparse.ArgumentParser(description='Distributed training for telomere radiosensitivity prediction')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    
    args = parser.parse_args()
    
    # Get world size
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        print("No GPUs available. Running on CPU.")
        world_size = 1
    
    # Spawn processes
    mp.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
