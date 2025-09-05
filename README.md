
## Anaconda Environment Setup

### Huggingface, Unsloth, TRL
```code
conda create --name llmer python=3.10 -y
conda activate llmer

conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install "unsloth[cu121-torch230]"

pip install transformers datasets accelerate bitsandbytes peft sentencepiece protobuf xformers flash-attn scikit-learn scipy PyYAML

pip install git+https://github.com/huggingface/trl.git

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
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=29500 main_unsloth_torch.py
# torchrun --nproc_per_node=4 --master_port=29500  main_unsloth_torchadv.py
```


### LitGPT
```code
python dataset/litgpt_csqa.py
litgpt finetune_full meta-llama/Meta-Llama-3-8B-Instruct --config config.yaml
```


## Running LoRA Finetuning

### Pre evaluation of performance of base model
```code
python main_pre_eval.py --lora
```

### Huggingface (for loading models and tokenizers) and TRL (for training)
```code
python main_hf_trl.py --lora
# python main_hf_trladv.py --lora
```

### Huggingface (for loading models and tokenizers) Unsloth (for model optimization with LoRA) and TRL (for training)
```code
python main_unsloth_trl.py --lora
# python main_unsloth_trladv.py --lora
```



# References
- [LitGPT](litgpt/tutorials/finetune_full.md)
- [Unsloth](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)
- [Huggingface TRL](https://huggingface.co/docs/trl/index)