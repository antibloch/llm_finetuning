
# Huggingface Fine-tuning
```code
conda create --name llmer python=3.10 -y
conda activate llmer

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install "unsloth[cu121-torch230]"
pip install "transformers>=4.41.0" "datasets>=2.14.0" "accelerate>=0.29.0" "peft>=0.10.0" "trl>=0.8.6"
pip install bitsandbytes sentencepiece protobuf xformers
```




# LitGPT Fine-tuning Example
```code
conda create -n lip python=3.10 -y && conda activate lip
pip install 'litgpt[extra]

huggingface-cli login
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir

python dataset/litgpt_csqa.py

litgpt finetune_full meta-llama/Meta-Llama-3-8B-Instruct --config config.yaml
```




# References
- [LitGPT](litgpt/tutorials/finetune_full.md at main · Lightning-AI/litgpt · GitHub)