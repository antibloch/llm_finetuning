
## Anaconda Environment Setup

### Huggingface, Unsloth, TRL
```code
conda create --name llmer python=3.10 -y
conda activate llmer

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

pip install "unsloth[cu121-torch230]"

pip install scikit-learn scipy

pip install transformers datasets accelerate

pip install bitsandbytes peft

pip install sentencepiece protobuf xformers
pip install trl

pip install flash-attn

pip install PyYAML
```



### LitGPT
```code
conda create -n lip python=3.10 -y && conda activate lip
pip install 'litgpt[extra]

huggingface-cli login
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir
```


## Running Full Parameter Finetuning

### Pre evaluation of performance of base model
```code
python main_pre_eval.py
```

### Huggingface (for loading models and tokenizers) and TRL (for training)
```code
python main_hf_trl.py
# python main_hf_trladv.py
```

### Huggingface (for loading models and tokenizers) Unsloth (for model optimization with LoRA) and TRL (for training)
```code
python main_unsloth_trl.py
# python main_unsloth_trladv.py
```


### Huggingface (for loading models and tokenizers) Unsloth (for model optimization with LoRA) and Torch Distributed (for training)
```code
python main_unsloth_torch.py
# python main_unsloth_torchadv.py
```


### LitGPT
```code
python dataset/litgpt_csqa.py
litgpt finetune_full meta-llama/Meta-Llama-3-8B-Instruct --config config.yaml
```

# References
- [LitGPT](litgpt/tutorials/finetune_full.md at main · Lightning-AI/litgpt · GitHub)