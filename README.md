
## Anaconda Environment Setup

### Huggingface, Unsloth, TRL
```code
conda create --name llmer python=3.10 -y
conda activate llmer

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install "unsloth[cu121-torch230]"
pip install "transformers>=4.41.0" "datasets>=2.14.0" "accelerate>=0.29.0" "peft>=0.10.0" "trl>=0.8.6"
pip install bitsandbytes sentencepiece protobuf xformers
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