from trl import SFTTrainer, SFTConfig
from utils.vram_instrumentation import VRAMMonitorCallback


def train_with_trl(model, tokenizer, dataset, config, do_instrument=True):
    """
    Trains the model using TRL's SFTTrainer.
    """

    max_seq_length = config.get('max_seq_length')
    num_train_epochs = config.get('num_train_epochs', 1)
    learning_rate = config.get('learning_rate', 2e-5)
    per_device_batch_size = config.get('per_device_batch_size', 2)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
    max_steps = config.get('max_steps', 60)
    output_dir = config.get('output_dir', "outputs")
    optim = config.get('optim', "adamw_8bit")

    sft_config = SFTConfig(
            per_device_train_batch_size = per_device_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            learning_rate = learning_rate,
            num_train_epochs = num_train_epochs,
            max_steps = max_steps,
            optim = optim,
            output_dir = output_dir,
            report_to = "none"
        )
    
    if do_instrument:
        callback = VRAMMonitorCallback(
            detailed_logging=True,
            track_per_step=True, 
            step_interval=50
        )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        args = sft_config,
        callbacks = [callback] if do_instrument else None,
    )

    # Start training
    trainer.train()

    return trainer