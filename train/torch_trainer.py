import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from utils.vram_instrumentation import VRAMTracker, print_vram_summary, vram_checkpoint

logger = logging.getLogger(__name__)

def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def collate_fn(batch, tokenizer, max_seq_length):
    """Custom collate function for dynamic padding"""
    texts = [item['text'] for item in batch]
    
    # Tokenize the batch
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )
    
    # Create labels (same as input_ids for causal LM)
    tokenized['labels'] = tokenized['input_ids'].clone()
    
    return tokenized

def train_epoch(model, dataloader, optimizer, scheduler, device, rank, vram_tracker=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    if rank == 0:
        progress = tqdm(dataloader, desc="Training")
    else:
        progress = dataloader
    
    for step, batch in enumerate(progress):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        # Update progress bar and track VRAM
        if rank == 0:
            if isinstance(progress, tqdm):
                progress.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(step+1):.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}' if scheduler else f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # Track VRAM every 10 steps
            if vram_tracker and step % 10 == 0:
                vram_tracker.snapshot(f"Step {step}")
    
    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate(model, dataloader, device, rank):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        if rank == 0:
            progress = tqdm(dataloader, desc="Evaluating")
        else:
            progress = dataloader
            
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            
            if rank == 0 and isinstance(progress, tqdm):
                progress.set_postfix({'eval_loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss

def train_with_torch(model, tokenizer, dataset, config, do_instrument=True):
    """
    Trains the model using PyTorch distributed training.
    """
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Extract configuration parameters
    max_seq_length = config.get('max_seq_length', 2048)
    num_train_epochs = config.get('num_train_epochs', 1)
    learning_rate = config.get('learning_rate', 2e-5)
    per_device_batch_size = config.get('per_device_batch_size', 2)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
    max_steps = config.get('max_steps', None)
    output_dir = config.get('output_dir', "outputs")
    weight_decay = config.get('weight_decay', 0.01)
    warmup_steps = config.get('warmup_steps', 100)

    
    if rank == 0:
        logger.info("PyTorch Distributed Training Configuration:")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Rank: {rank}")
        logger.info(f"  Local rank: {local_rank}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Epochs: {num_train_epochs}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Batch size per device: {per_device_batch_size}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"  Max sequence length: {max_seq_length}")
    
    # Initialize VRAM tracking
    vram_tracker = None
    if do_instrument and rank == 0:
        vram_tracker = VRAMTracker()
        vram_checkpoint("Initial state")
    
    # Move model to device
    model = model.to(device)
    
    if do_instrument and rank == 0:
        vram_checkpoint("After model to device")
    
    # Wrap model with DDP for multi-GPU training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if rank == 0:
            logger.info("Model wrapped with DistributedDataParallel")
    
    if do_instrument and rank == 0:
        vram_checkpoint("After DDP wrapping")
    
    # Prepare dataset and dataloader
    # Split dataset for training/validation if needed
    if hasattr(dataset, 'train_test_split'):
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
    else:
        # Use 90% for training, 10% for validation
        dataset_size = len(dataset)
        train_size = int(0.9 * dataset_size)
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, dataset_size))
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    ) if world_size > 1 else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_seq_length),
        pin_memory=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_device_batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_seq_length),
        pin_memory=True,
        num_workers=2
    )
    
    if rank == 0:
        logger.info(f"Training samples: {len(train_dataset):,}")
        logger.info(f"Validation samples: {len(val_dataset):,}")
        logger.info(f"Training batches: {len(train_loader):,}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Setup scheduler
    total_steps = len(train_loader) * num_train_epochs
    if max_steps and max_steps < total_steps:
        total_steps = max_steps
        
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    if rank == 0:
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")
    
    if do_instrument and rank == 0:
        vram_checkpoint("Before training start")
    
    # Training loop
    global_step = 0
    for epoch in range(num_train_epochs):
        if rank == 0:
            logger.info(f"\nEpoch {epoch + 1}/{num_train_epochs}")
        
        # Set sampler epoch for proper shuffling
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        if do_instrument and rank == 0:
            vram_tracker.snapshot(f"Epoch {epoch + 1} start")
        
        # Train epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, rank, vram_tracker
        )
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device, rank)
        
        if rank == 0:
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if do_instrument:
                vram_tracker.snapshot(f"Epoch {epoch + 1} end")
        
        # Save checkpoint
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model state
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'global_step': global_step
            }
            
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Check if we've reached max_steps
        global_step += len(train_loader)
        if max_steps and global_step >= max_steps:
            if rank == 0:
                logger.info(f"Reached max_steps ({max_steps}), stopping training")
            break
    
    # Save final model
    if rank == 0:
        logger.info("Saving final model...")
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
        
        if do_instrument:
            vram_checkpoint("Training completed")
            logger.info("VRAM usage history:")
            vram_tracker.print_history()
            
            peak_usage = vram_tracker.get_peak_usage()
            if peak_usage:
                logger.info(f"Peak VRAM usage: {peak_usage['reserved_GB']:.2f}GB at {peak_usage['label']}")
    
    # Cleanup distributed training
    cleanup_distributed()
    
    return model