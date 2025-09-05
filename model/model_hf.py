import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback
)

def get_model_stuff(config, do_lora=True):
    MODEL_NAME = config['MODEL_NAME']

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token


    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Distribute across all available GPUs
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))


    return model, tokenizer
