import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from utils.param_counter import *
from utils.vram_instrumentation import vram_checkpoint, VRAMTracker, print_vram_summary


def get_model_stuff(config, do_lora=True, do_instrument=True):
    MODEL_NAME = config['MODEL_NAME']
    if do_instrument:
        tracker = VRAMTracker()
        tracker.snapshot("Before loading model and tokenizer")


    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set padding side to left for decoder-only models
    tokenizer.padding_side = 'left'


    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",  # Distribute across all available GPUs
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))

    if do_lora:
        from peft import get_peft_model, LoraConfig, TaskType

        lora_alpha = config['lora_alpha'] if 'lora_alpha' in config else 16
        lora_r = config['lora_r'] if 'lora_r' in config else 16
        lora_dropout = config['lora_dropout'] if 'lora_dropout' in config else 0    
        bias = config['bias'] if 'bias' in config else "none"
        target_modules = config['target_modules'] if 'target_modules' in config else [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


    if do_instrument:
        tracker.snapshot("After loading model and tokenizer")
        tracker.print_history()
        print_trainable_parameters(model)
        print_vram_summary()


    return model, tokenizer
