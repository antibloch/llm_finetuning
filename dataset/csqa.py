from datasets import load_dataset


def format_commonsense_qa(examples, eos_token):
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




def dataset_loader(tokenizer, split='train'):
    # Load the CommonsenseQA dataset from Hugging Face
    dataset = load_dataset("commonsense_qa", split=split)
    
    # Get the EOS token from the tokenizer
    eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else ""
    
    # Format the dataset using the provided function
    dataset = dataset.map(lambda x: format_commonsense_qa(x, eos_token), batched=True)
    
    return dataset