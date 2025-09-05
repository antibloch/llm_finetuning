
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

### Accelerate setup
```code
In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------
Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: yes
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
Do you want to use DeepSpeed? [yes/NO]: NO
Do you want to use FullyShardedDataParallel? [yes/NO]: yes
----------------------------------------------------------------------------------------------------------
What should be your FSDP version? [2]:
2
Do you want to enable resharding after forward? [YES/no]: YES
Do you want to offload parameters and gradients to CPU? [yes/NO]: yes
----------------------------------------------------------------------------------------------------------
What should be your auto wrap policy?
TRANSFORMER_BASED_WRAP
Do you want to use the model's _no_split_modules to wrap. Only applicable for 🤗 Transformers [yes/NO]: yes
----------------------------------------------------------------------------------------------------------
What should be your FSDP's state dict type?
SHARDED_STATE_DICT
Do you want to enable CPU RAM efficient model loading? Only applicable for 🤗 Transformers models. [YES/no]: YES
Do you want to enable FSDP activation checkpointing? [yes/NO]: yes
Do you want to use the parallelism config? [yes/NO]: NO
How many GPU(s) should be used for distributed training? [1]:4
----------------------------------------------------------------------------------------------------------
Do you wish to use mixed precision?
bf16
accelerate configuration saved at /home/junaid/.cache/huggingface/accelerate/default_config.yaml
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

### Huggingface (for loading models and tokenizers) and TRL (for training)
```code
# first run and setup for multi-GPU training
accelerate config

# Basic multi-GPU with Accelerate
accelerate launch main_hf_trl_multi.py

# With DeepSpeed for large models
accelerate launch --config_file deepspeed_zero2.yaml main_hf_trl_multi.py

accelerate launch --multi_gpu --num_processes=2 main_hf_trl_multi.py

torchrun \ # python -m torch.distributed.run 
    --nproc_per_node 2 \
    --nnodes 2 \
    --rdzv_id 2299 \ # A unique job id 
    --rdzv_backend c10d \
    --rdzv_endpoint master_node_ip_address:29500 \
    main_hf_trl_multi.py


torchrun --nproc_per_node=2 --nnodes=2 --rdzv_endpoint=master:29500 main_hf_trl_multi.py
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