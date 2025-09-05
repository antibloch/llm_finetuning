from model.model_unsloth import *
from dataset.csqa import load_dataset
from utils.param_counter import *
from train.torch_trainer_adv import *
from evaluation.evaluate import *
import yaml


def main():
    config_path = "config/config.yaml"

    # Open the config file and load its contents
    with open(config_path, "r") as file:
        config = yaml.safe_load(file) # Use safe_load for security

    # Load the model and tokenizer
    model, tokenizer = get_model_stuff(config, do_lora=False)

    # Load and preprocess the dataset
    dataset = load_dataset(tokenizer, split='train')
    eval_dataset = load_dataset(tokenizer, split='validation')

    # Train the model using TRL's SFTTrainer
    trained_model = train_with_torch(model, tokenizer, dataset, config, do_instrument=True)


    # EVALUATION AFTER TRAINING
    print("\n" + "="*60)
    print("EVALUATION AFTER FINE-TUNING")
    print("="*60)
    # Evaluate the trained model
    results = evaluate_trained_model(
        trained_model, 
        tokenizer, 
        eval_dataset, 
        batch_size=2,
        max_samples=100  # num samples to evaluate
    )
    
    # Print results
    print_evaluation_results(results)


if __name__ == "__main__":
    main()





