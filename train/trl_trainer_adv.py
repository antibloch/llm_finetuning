from trl import SFTTrainer, SFTConfig
from utils.vram_instrumentation import VRAMMonitorCallback


def train_with_trl(model, tokenizer, dataset, config, do_instrument=True):
    """
    Trains the model using TRL's SFTTrainer.
    """

    max_seq_length = config.get('max_seq_length', 2048)
    learning_rate = config.get('learning_rate', 2e-5)
    per_device_batch_size = config.get('per_device_batch_size', 2)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
    output_dir = config.get('output_dir', "outputs")
    optim = config.get('optim', 'adamw_8bit')
    num_train_epochs = config.get('num_train_epochs', 1)
    weight_decay = config.get('weight_decay', 0.01)
    max_grad_norm = config.get('max_grad_norm', 1.0)
    lr_scheduler_type = config.get('lr_scheduler_type', 'cosine')
    warmup_ratio = config.get('warmup_ratio', 0.1)
    logging_steps = config.get('logging_steps', 10)
    save_steps = config.get('save_steps', 200)
    save_total_limit = config.get('save_total_limit', 2)
    dataloader_num_workers = config.get('dataloader_num_workers', 2)

    # Create SFT configuration
    sft_config = SFTConfig(
        output_dir=output_dir,
        
        num_train_epochs=num_train_epochs,
        
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_drop_last=True,
        
        optim=optim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        
        bf16=True, #  Reduces memory usage, good for large models
        tf32=True, # Free performance boost on compatible hardware
        # If get NaN losses, try disabling bf16 first
        
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        
        remove_unused_columns=False,
        
        eval_strategy="no",
        
        max_length=max_seq_length,
        packing=False,
    
        report_to="none",
        dataset_text_field="text",
    )

    callbacks = []
    if do_instrument:
        callback = VRAMMonitorCallback(
            detailed_logging=True,
            track_per_step=True, 
            step_interval=50
        )
        callbacks.append(callback)

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = sft_config,
        callbacks = callbacks,
    )

    # Start training
    trainer.train()


    return trainer