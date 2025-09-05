

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