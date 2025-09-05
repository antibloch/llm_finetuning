import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
from typing import Tuple, Optional

def print_beautiful_example(input_prompt, generated_output, predicted_answer, correct_answer, example_num, is_correct):
    """
    Print a beautifully formatted example showing input prompt and model output.
    """
    # Color codes for terminal output
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    # Choose color based on correctness
    result_color = GREEN if is_correct else RED
    result_symbol = "✓" if is_correct else "✗"
    
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}{CYAN}EXAMPLE {example_num}{RESET}")
    print(f"{BOLD}{'='*80}{RESET}")
    
    # Parse and display the input prompt nicely
    print(f"{BOLD}{BLUE}INPUT PROMPT:{RESET}")
    print(f"{'-'*40}")
    
    # Split the prompt to extract question and choices
    try:
        parts = input_prompt.split("### Question:")
        if len(parts) > 1:
            question_part = parts[1].split("### Choices:")[0].strip()
            choices_part = parts[1].split("### Choices:")[1].split("### Answer:")[0].strip()
            
            print(f"{BOLD}Question:{RESET}")
            print(f"  {question_part}")
            print(f"\n{BOLD}Choices:{RESET}")
            for line in choices_part.split('\n'):
                if line.strip():
                    print(f"  {line.strip()}")
        else:
            # Fallback if parsing fails
            print(f"  {input_prompt[:200]}...")
    except:
        print(f"  {input_prompt[:200]}...")
    
    print(f"\n{BOLD}{YELLOW}MODEL OUTPUT:{RESET}")
    print(f"{'-'*40}")
    print(f"  Generated: '{generated_output}'")
    
    print(f"\n{BOLD}EVALUATION:{RESET}")
    print(f"{'-'*40}")
    print(f"  Predicted Answer: {BOLD}{predicted_answer}{RESET}")
    print(f"  Correct Answer:   {BOLD}{correct_answer}{RESET}")
    print(f"  Result: {result_color}{BOLD}{result_symbol} {'CORRECT' if is_correct else 'INCORRECT'}{RESET}")
    
    print(f"{BOLD}{'='*80}{RESET}")

def extract_answer_from_generation(text, choices=['A', 'B', 'C', 'D', 'E']):
    """
    Extract the answer choice from generated text.
    """
    text = text.strip().upper()
    
    # Look for patterns like "A", "The answer is A", "A)", etc.
    for choice in choices:
        patterns = [
            rf'\b{choice}\b',  # Standalone letter
            rf'\b{choice}\)',  # Letter with parenthesis
            rf'\b{choice}\.',  # Letter with period
            rf'ANSWER\s*:?\s*{choice}',  # "Answer: A"
            rf'THE\s+ANSWER\s+IS\s+{choice}',  # "The answer is A"
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return choice
    
    # Fallback: return first letter found
    for choice in choices:
        if choice in text:
            return choice
    
    return None

def evaluate_trained_model(model, tokenizer, test_dataset, batch_size=8, max_samples=None, show_examples=3):
    """
    Evaluate a trained model (in memory) on CSQA dataset.
    
    Args:
        model: Trained PyTorch model
        tokenizer: Associated tokenizer
        test_dataset: Test/validation dataset
        batch_size: Batch size for evaluation
        max_samples: Limit number of samples (for quick testing)
        show_examples: Number of examples to display beautifully (default: 3)
    
    Returns:
        Dict with evaluation results
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Limit samples if specified
    if max_samples:
        test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))
    
    print(f"Evaluating on {len(test_dataset)} samples...")
    print(f"Will display first {show_examples} examples with detailed formatting")
    
    # Create dataloader
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch_texts = batch['text']
            
            # Get answer keys - handle different possible field names
            batch_answers = None
            for key in ['answerKey', 'answer', 'label']:
                if key in batch:
                    batch_answers = batch[key]
                    break
            
            if batch_answers is None:
                print("Warning: No answer key found in dataset")
                break
            
            # Tokenize inputs
            inputs = tokenizer(
                batch_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=2048
            ).to(device)
            
            # 1. ACCURACY EVALUATION: Generate responses
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,  # Short generation for answer
                    do_sample=False,    # Deterministic for evaluation
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=1.0,
                    top_p=1.0
                )
            
            # Extract generated text (remove input)
            generated_texts = []
            for i, output in enumerate(outputs):
                input_length = inputs['input_ids'][i].shape[0]
                generated_ids = output[input_length:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                generated_texts.append(generated_text)
            
            # Check answers
            for i, (generated, correct_answer) in enumerate(zip(generated_texts, batch_answers)):
                predicted_answer = extract_answer_from_generation(generated)
                
                if predicted_answer == correct_answer:
                    correct += 1
                total += 1
                
                # Beautiful display of first few examples
                if total <= show_examples:
                    print_beautiful_example(
                        batch_texts[i], 
                        generated.strip(), 
                        predicted_answer, 
                        correct_answer, 
                        total,
                        predicted_answer == correct_answer
                    )
            
            # 2. PERPLEXITY EVALUATION: Compute loss
            labels = inputs['input_ids'].clone()
            labels[inputs['attention_mask'] == 0] = -100
            
            loss_outputs = model(**inputs, labels=labels)
            loss = loss_outputs.loss
            
            # Calculate number of non-padded tokens
            num_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    results = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'perplexity': perplexity,
        'avg_loss': avg_loss
    }
    
    return results

def print_evaluation_results(results, model_name="Trained Model"):
    """Print formatted evaluation results."""
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Total Samples: {results['total']}")
    print("-" * 60)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"Perplexity: {results['perplexity']:.4f}")
    print(f"Average Loss: {results['avg_loss']:.4f}")
    print("=" * 60)