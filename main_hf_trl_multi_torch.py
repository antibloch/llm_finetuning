from model.model_hf import *
from dataset.csqa import dataset_loader
from utils.param_counter import *
from train.trl_trainer_multi_torch import *
from evaluation.evaluate import *
import yaml
import argparse



def main(args):
    if args.lora:
        do_lora = True
    else:
        do_lora = False
        
    config_path = "config/config.yaml"

    # Open the config file and load its contents
    with open(config_path, "r") as file:
        config = yaml.safe_load(file) # Use safe_load for security

    # Load the model and tokenizer
    model, tokenizer = get_model_stuff(config, do_lora=do_lora)

    # Load and preprocess the dataset
    dataset = dataset_loader(tokenizer, split='train')
    eval_dataset = dataset_loader(tokenizer, split='validation')

    print("Dataset sizes:")
    print(f"  Training: {len(dataset)} samples")
    print(f"  Evaluation: {len(eval_dataset)} samples")


    # Train the model using TRL's SFTTrainer
    trainer = train_with_trl(model, tokenizer, dataset, config=None, do_instrument=True)

    # Extract the trained model for evaluation
    trained_model = trainer.model

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
        max_samples=100,  # num samples to evaluate
        show_examples=5,  # print 5 random input-output pairs
        max_new_tokens=10,  # Short generation for answer
        temperature=0.0,    # Temperature (high temp for creative (less greedy))
        top_p=0.9           # Nucleus sampling (small value for less diversity and more focused generations)
    )
    
    # Print results
    print_evaluation_results(results)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model using LoRA and TRL.")
    parser.add_argument('--lora', action='store_true', help="Use LoRA for fine-tuning.")
    args = parser.parse_args()
    main(args)





