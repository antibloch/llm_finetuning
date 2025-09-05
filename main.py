from model.model_unsloth import *
from dataset.csqa import load_dataset
from utils.param_counter import *
from train.trl_trainer import *
import yaml


def main():
    # Path to your config file
    config_path = "config.yaml"

    # Open the config file and load its contents
    with open(config_path, "r") as file:
        config = yaml.safe_load(file) # Use safe_load for security

    # Load the model and tokenizer
    model, tokenizer = get_model_stuff(config, do_lora=False)

    # Load and preprocess the dataset
    dataset = load_dataset(tokenizer, split='train')

    # Train the model using TRL's SFTTrainer
    train_with_trl(model, tokenizer, dataset, config)



if __name__ == "__main__":
    main()





