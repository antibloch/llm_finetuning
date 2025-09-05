from model.model_hf import *
from dataset.csqa import dataset_loader
from utils.param_counter import *
from train.trl_trainer import *
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
    dataset = dataset_loader(tokenizer, split='train')
    eval_dataset = dataset_loader(tokenizer, split='validation')

    print("Dataset sizes:")
    print(f"  Training: {len(dataset)} samples")
    print(f"  Evaluation: {len(eval_dataset)} samples")

    # EVALUATION BEFORE TRAINING
    print("\n" + "="*60)
    print("EVALUATION BEFORE FINE-TUNING (Baseline)")
    print("="*60)
    
    baseline_results = evaluate_trained_model(
        model, 
        tokenizer, 
        eval_dataset, 
        batch_size=config.get('per_device_batch_size', 8),
        max_samples=100,  # num samples to evaluate
        show_examples=5,  # print 5 random input-output pairs
        max_new_tokens=10,  # Short generation for answer
        temperature=0.0,    # Temperature (high temp for creative (less greedy))
        top_p=0.9           # Nucleus sampling (small value for less diversity and more focused generations)
    )
    
    print_evaluation_results(baseline_results, model_name=f"{config['MODEL_NAME']} (Baseline)")



if __name__ == "__main__":
    main()





