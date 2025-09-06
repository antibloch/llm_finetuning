from trl import SFTTrainer, SFTConfig
from utils.vram_instrumentation import VRAMMonitorCallback
import torch
import os


def train_with_trl(model, tokenizer, dataset, config, do_instrument=True):
    """
    Trains the model using TRL's SFTTrainer with multi-GPU support via DeepSpeed or Accelerate.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer/processing_class
        dataset: Training dataset
        config: Configuration dictionary
        do_instrument: Whether to use VRAM monitoring
    """

    if config is None:
        config = get_recommended_config(model_size="7b", gpu_memory_gb=48)
    
    # Extract configuration parameters
    max_seq_length = config.get('max_seq_length')
    num_train_epochs = config.get('num_train_epochs', 1)
    learning_rate = config.get('learning_rate', 2e-5)
    per_device_batch_size = config.get('per_device_batch_size', 2)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
    max_steps = config.get('max_steps', 60)
    output_dir = config.get('output_dir', "outputs")
    optim = config.get('optim', "adamw_8bit")
    
    # Multi-GPU backend configuration
    multi_gpu_backend = config.get('multi_gpu_backend', None)  # "deepspeed" or "accelerate"
    deepspeed_config_path = config.get('deepspeed_config_path', None)
    
    # Additional training optimizations
    gradient_checkpointing = config.get('gradient_checkpointing', False)
    fp16 = config.get('fp16', False)
    bf16 = config.get('bf16', False)
    dataloader_num_workers = config.get('dataloader_num_workers', 0)
    save_steps = config.get('save_steps', 500)
    logging_steps = config.get('logging_steps', 10)
    warmup_steps = config.get('warmup_steps', 0)
    warmup_ratio = config.get('warmup_ratio', 0.0)
    
    # Log multi-GPU setup information
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        effective_batch_size = per_device_batch_size * num_gpus * gradient_accumulation_steps
        print(f"Multi-GPU training detected:")
        print(f"  - Number of GPUs: {num_gpus}")
        print(f"  - Backend: {multi_gpu_backend or 'auto'}")
        print(f"  - Per device batch size: {per_device_batch_size}")
        print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  - Effective batch size: {effective_batch_size}")
        if deepspeed_config_path:
            print(f"  - DeepSpeed config: {deepspeed_config_path}")
    
    # Create SFT configuration - FIXED: Use eval_strategy instead of evaluation_strategy
    sft_config = SFTConfig(
        # Core training parameters
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        optim=optim,
        output_dir=output_dir,
        
        # Dataset and model parameters - FIXED: moved from SFTTrainer to SFTConfig
        dataset_text_field="text",
        max_seq_length=max_seq_length,  # FIXED: now passed here instead of max_length
        packing=config.get('packing', False),  # FIXED: moved to SFTConfig
        
        # Multi-GPU configuration
        deepspeed=deepspeed_config_path if multi_gpu_backend == "deepspeed" else None,
        
        # Performance optimizations
        gradient_checkpointing=gradient_checkpointing,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_drop_last=True,  # Ensures consistent batch sizes across GPUs
        remove_unused_columns=False,  # Prevents issues with custom datasets
        
        # Logging and saving
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,  # Keep only last 3 checkpoints
        report_to="none",
        
        # Learning rate scheduling
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=config.get('lr_scheduler_type', 'linear'),
        
        # Additional stability settings
        max_grad_norm=config.get('max_grad_norm', 1.0),
        weight_decay=config.get('weight_decay', 0.01),
        adam_beta1=config.get('adam_beta1', 0.9),
        adam_beta2=config.get('adam_beta2', 0.999),
        adam_epsilon=config.get('adam_epsilon', 1e-8),
        
        # FIXED: Changed evaluation_strategy to eval_strategy
        eval_strategy=config.get('eval_strategy', 'no'),  # FIXED: renamed parameter
        eval_steps=config.get('eval_steps', 500),
        per_device_eval_batch_size=config.get('per_device_eval_batch_size', per_device_batch_size),
        
        # Memory optimization
        ddp_find_unused_parameters=False,  # Improves DDP performance
        dataloader_pin_memory=True,  # Faster data transfer to GPU
    )
    
    # Initialize callbacks
    callbacks = []
    if do_instrument:
        callback = VRAMMonitorCallback(
            detailed_logging=True,
            track_per_step=True, 
            step_interval=50
        )
        callbacks.append(callback)
    
    # Create trainer - FIXED: Simplified parameters, moved most to SFTConfig
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # FIXED: Use processing_class instead of tokenizer
        train_dataset=dataset,
        args=sft_config,
        callbacks=callbacks if callbacks else None,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Training completed. Saving final model...")
    trainer.save_model()
    
    return trainer


def get_recommended_config(num_gpus=None, model_size="7b", gpu_memory_gb=24):
    """
    Get recommended configuration based on setup.
    
    Args:
        num_gpus: Number of GPUs (auto-detected if None)
        model_size: Model size ("1b", "3b", "7b", "13b", "30b", "70b")
        gpu_memory_gb: GPU memory in GB
    
    Returns:
        Dictionary with recommended configuration
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    # Base configuration
    config = {
        'max_seq_length': 2048,
        'num_train_epochs': 1,
        'learning_rate': 2e-5,
        'max_steps': -1,  # Use epochs instead
        'output_dir': "./outputs",
        'optim': "adamw_8bit",
        'gradient_checkpointing': True,
        'warmup_ratio': 0.03,
        'lr_scheduler_type': 'cosine',
        'save_steps': 500,
        'logging_steps': 10,
        'dataloader_num_workers': 4,
        'packing': False,  # Can be enabled for efficiency
        'eval_strategy': 'no',  # FIXED: Use eval_strategy instead of evaluation_strategy
    }
    
    # Model size specific settings
    size_configs = {
        "1b": {"per_device_batch_size": 8, "gradient_accumulation_steps": 1},
        "3b": {"per_device_batch_size": 4, "gradient_accumulation_steps": 2},
        "7b": {"per_device_batch_size": 2, "gradient_accumulation_steps": 4},
        "13b": {"per_device_batch_size": 1, "gradient_accumulation_steps": 8},
        "30b": {"per_device_batch_size": 1, "gradient_accumulation_steps": 16},
        "70b": {"per_device_batch_size": 1, "gradient_accumulation_steps": 32},
    }
    
    if model_size in size_configs:
        config.update(size_configs[model_size])
    
    # Multi-GPU backend recommendation
    if num_gpus > 1:
        if model_size in ["30b", "70b"] or gpu_memory_gb < 20:
            config['multi_gpu_backend'] = "deepspeed"
            config['deepspeed_config_path'] = "deepspeed_zero2.json"
        else:
            config['multi_gpu_backend'] = "accelerate"
    
    # Precision settings based on GPU
    if gpu_memory_gb >= 40:  # A100, H100
        config['bf16'] = True
    else:
        config['fp16'] = True
    
    # Adjust for memory constraints
    if gpu_memory_gb < 12:
        config['per_device_batch_size'] = max(1, config['per_device_batch_size'] // 2)
        config['gradient_accumulation_steps'] *= 2
        config['gradient_checkpointing'] = True
    
    return config
