from unsloth import FastLanguageModel
from utils.param_counter import count_parameters
from utils.vram_instrumentation import vram_checkpoint, VRAMTracker, print_vram_summary


def get_model_stuff(config, do_lora=True):
    # some hyperparameters
    MODEL_NAME = config['MODEL_NAME']
    max_seq_length = config['max_seq_length']
    dtype = config['dtype']
    load_in_4bit = config['load_in_4bit']
    load_in_8bit = config['load_in_8bit']

    tracker = VRAMTracker()
    tracker.snapshot("Before loading model and tokenizer")

    # Load the model and tokenizer using Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        load_in_8bit = load_in_8bit,
    )

    # Add LoRA adapters to make it trainable (but if it is full-finetuning, skip this step)
    if do_lora:
        lora_alpha = config['lora_alpha'] if 'lora_alpha' in config else 16
        lora_r = config['lora_r'] if 'lora_r' in config else 16
        lora_dropout = config['lora_dropout'] if 'lora_dropout' in config else 0    
        bias = config['bias'] if 'bias' in config else "none"
        use_gradient_checkpointing = config['use_gradient_checkpointing'] if 'use_gradient_checkpointing' in config else "unsloth"
        random_state = config['random_state'] if 'random_state' in config else 3407
        model = FastLanguageModel.get_peft_model(
            model,
            r = lora_r, # Rank: suggested 8, 16, 32, etc.
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            bias = bias,
            use_gradient_checkpointing = use_gradient_checkpointing,
            random_state = random_state,
        )


    # instrument params
    tracker.snapshot("After loading model and tokenizer")
    tracker.print_history()
    count_parameters(model)
    print_vram_summary()


    return model, tokenizer
