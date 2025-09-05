# Save this file as prepare_csqa_formatted.py
import json
from pathlib import Path
import sys

from datasets import load_dataset
from tqdm import tqdm
from jsonargparse import CLI

def format_commonsense_qa(examples, eos_token):
    """Format CommonsenseQA examples using the specified template."""
    # Define the prompt template for your multiple-choice task
    mcq_prompt_template = """Below is a multiple-choice question. Your task is to select the correct answer from the choices provided.

### Question:
{}

### Choices:
{}

### Answer:
{}"""

    questions = examples["question"]
    choices_list = examples["choices"]
    answer_keys = examples["answerKey"]
    
    texts = []
    # Iterate through each example in the batch
    for question, choices, answer_key in zip(questions, choices_list, answer_keys):
        # Format the choices into a single string (e.g., "A. choice1\nB. choice2")
        formatted_choices = "\n".join([f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])])
        
        # Find the full text of the correct answer
        # First, find the index of the correct label (e.g., index of 'A')
        correct_answer_index = choices["label"].index(answer_key)
        # Then, get the text at that index
        correct_answer_text = choices["text"][correct_answer_index]
        
        # Format the full prompt
        text = mcq_prompt_template.format(question, formatted_choices, correct_answer_text) + eos_token
        texts.append(text)
        
    return {"text": texts}

def prepare(
    destination_path: Path = Path("data/commonsense_qa"),
    test_split_size: int = 500,
    eos_token: str = "",
) -> None:
    """
    Prepares the CommonsenseQA dataset for fine-tuning using the new formatting approach.
    The dataset is downloaded, formatted, and saved as train and validation JSON files.
    """
    destination_path.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("commonsense_qa", split="train")
    dataset = dataset.train_test_split(test_size=test_split_size, seed=42)
    train_set = dataset["train"]
    val_set = dataset["test"]

    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")

    print("Formatting and saving the training set...")
    format_and_save(train_set, destination_path / "train.json", eos_token)

    print("Formatting and saving the validation set...")
    format_and_save(val_set, destination_path / "val.json", eos_token)

    print("Data preparation complete. ✅")

def format_and_save(dataset_split, file_path, eos_token=""):
    """Formats a dataset split using the new template and saves it to a JSON file."""
    # Apply the formatting function to the entire dataset split
    formatted_dataset = dataset_split.map(
        lambda x: format_commonsense_qa(x, eos_token), 
        batched=True,
        desc=f"Processing {file_path.name}"
    )
    
    # Extract the formatted text and save as JSON
    output_data = []
    for sample in tqdm(formatted_dataset, desc=f"Saving {file_path.name}"):
        output_data.append({
            "text": sample["text"]
        })

    with open(file_path, "w") as f:
        json.dump(output_data, f, indent=4)

def load_formatted_dataset(tokenizer=None, split='train'):
    """
    Load and format the CommonsenseQA dataset (compatible with the original function signature).
    """
    # Load the CommonsenseQA dataset from Hugging Face
    dataset = load_dataset("commonsense_qa", split=split)
    
    # Get the EOS token from the tokenizer
    eos_token = ""
    if tokenizer is not None:
        eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else ""
    
    # Format the dataset using the provided function
    dataset = dataset.map(lambda x: format_commonsense_qa(x, eos_token), batched=True)
    
    return dataset

if __name__ == "__main__":
    CLI(prepare)