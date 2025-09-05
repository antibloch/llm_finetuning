import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
from typing import Tuple, Optional

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

def evaluate_trained_model(model, tokenizer, test_dataset, batch_size=8, max_samples=None):
    """
    Evaluate a trained model (in memory) on CSQA dataset.
    SAFELY preserves the original model state.
    
    Args:
        model: Trained PyTorch model
        tokenizer: Associated tokenizer
        test_dataset: Test/validation dataset
        batch_size: Batch size for evaluation
        max_samples: Limit number of samples (for quick testing)
    
    Returns:
        Dict with evaluation results
    """
    # IMPORTANT: Save the original training state
    original_training_mode = model.training
    
    try:
        # Set to evaluation mode
        model.eval()
        device = next(model.parameters()).device
        
        # Limit samples if specified
        if max_samples:
            test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))
        
        print(f"Evaluating on {len(test_dataset)} samples...")
        
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
                    
                    # Debug: print first few examples from first batch
                    if batch_idx == 0 and i < 3:
                        print(f"Example {total}:")
                        print(f"  Generated: '{generated.strip()}'")
                        print(f"  Predicted: {predicted_answer}")
                        print(f"  Correct: {correct_answer}")
                        print(f"  Match: {predicted_answer == correct_answer}")
                        print()
                
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
    
    finally:
        # CRITICAL: Restore the original training state
        if original_training_mode:
            model.train()
        else:
            model.eval()
        
        print(f"Model state restored to: {'training' if original_training_mode else 'evaluation'}")

def evaluate_with_model_copy(model, tokenizer, test_dataset, batch_size=8, max_samples=None):
    """
    Alternative: Evaluate using a deep copy of the model (memory intensive).
    Use this if you want absolute guarantee of no state changes.
    """
    import copy
    
    print("Creating model copy for evaluation (this may take time and memory)...")
    
    # Create a deep copy of the model
    model_copy = copy.deepcopy(model)
    
    try:
        # Evaluate using the copy
        results = evaluate_trained_model(model_copy, tokenizer, test_dataset, batch_size, max_samples)
        return results
    finally:
        # Clean up the copy
        del model_copy
        torch.cuda.empty_cache()
        print("Model copy deleted and cache cleared")

def safe_model_evaluation_context(model):
    """
    Context manager for safe model evaluation.
    Usage:
        with safe_model_evaluation_context(model):
            # Do evaluation here
            results = some_evaluation_function(model, ...)
    """
    class ModelEvaluationContext:
        def __init__(self, model):
            self.model = model
            self.original_training_mode = None
        
        def __enter__(self):
            self.original_training_mode = self.model.training
            self.model.eval()
            return self.model
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.original_training_mode:
                self.model.train()
            else:
                self.model.eval()
    
    return ModelEvaluationContext(model)

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

# Example usage of the context manager approach:
def evaluate_with_context_manager(model, tokenizer, test_dataset, batch_size=8, max_samples=None):
    """
    Example of using the context manager for evaluation.
    """
    with safe_model_evaluation_context(model):
        # Your evaluation code here
        results = evaluate_trained_model(model, tokenizer, test_dataset, batch_size, max_samples)
    
    return results