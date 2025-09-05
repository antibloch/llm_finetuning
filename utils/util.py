from trl import SFTTrainer, SFTConfig


def train_with_trl(model, tokenizer, dataset, config):
    """
    Trains the model using TRL's SFTTrainer.
    """

    max_seq_length = config['max_seq_length']
    learning_rate = config['learning_rate'] if 'learning_rate' in config else 2e-5
    per_device_batch_size = config['per_device_batch_size'] if 'per_device_batch_size' in config else 2
    gradient_accumulation_steps = config['gradient_accumulation_steps'] if 'gradient_accumulation_steps' in config else 4
    max_steps = config['max_steps'] if 'max_steps' in config else 60
    output_dir = config['output_dir'] if 'output_dir' in config else "outputs"
    optim = config['optim'] if 'optim' in config else "adamw_8bit"

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        args = SFTConfig(
            per_device_train_batch_size = per_device_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            # A much lower learning rate is needed for full fine-tuning
            learning_rate = learning_rate,
            max_steps = max_steps,
            optim = optim,
            output_dir = output_dir,
            report_to = "none"
        ),
    )

    # Start training
    trainer.train()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

