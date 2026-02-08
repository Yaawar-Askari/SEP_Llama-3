import torch
import json
import logging
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath("semantic_uncertainty"))

# Reuse OATML model wrapper for loading, but we will access the raw model inside it
from uncertainty.models.huggingface_models import HuggingfaceModel
from sep_utils import format_llama3_prompt, setup_simple_logger


INPUT_LABEL_FILE = "sep_filtered_labels.json"
OUTPUT_TENSOR_FILE = "sep_dataset.pt"
MODEL_NAME = "Meta-Llama-3-8B-Instruct"

def get_last_token_hidden_state(model_instance, input_ids, attention_mask):
    """
    Extracts the hidden state of the last token of the PROMPT.
    Reusable logic for inference hooks.
    """
    # Access the underlying HF model from the OATML wrapper
    hf_model = model_instance.model 
    
    with torch.no_grad():
        # Pass output_hidden_states=True explicitly
        outputs = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False
        )
    
    # Last layer hidden states
    # shape: (batch, seq_len, hidden_dim)
    last_hidden_state = outputs.hidden_states[-1]
    
    # Get last token index (account for padding)
    # For batch_size=1, it's just -1, but let's be robust
    last_token_indices = attention_mask.sum(1) - 1
    
    # Gather features
    # shape: (batch, hidden_dim)
    features = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=hf_model.device),
        last_token_indices
    ]
    
    return features.cpu()

def main():
    setup_simple_logger()
    
    # 1. Load Validated Prompts
    logging.info("Loading filtered labels...")
    with open(INPUT_LABEL_FILE, "r") as f:
        dataset = json.load(f)
        
    # 2. Load Model
    logging.info(f"Loading Model: {MODEL_NAME}...")
    # We use the wrapper to handle loading, device map, quantization etc.
    wrapper = HuggingfaceModel(MODEL_NAME, stop_sequences=[], max_new_tokens=1)
    
    X_list = []
    y_list = []
    
    logging.info("Extracting features...")
    for item in tqdm(dataset):
        # CRITICAL: Re-construct prompt exactly as before
        document = item['document']
        prompt_text = format_llama3_prompt(document)
        
        # Tokenize
        inputs = wrapper.tokenizer(prompt_text, return_tensors="pt").to("cuda")
        
        # Extract
        features = get_last_token_hidden_state(
            wrapper, 
            inputs.input_ids, 
            inputs.attention_mask
        )
        
        X_list.append(features)
        y_list.append(item['label'])
        
    # 3. Aggregate and Save
    if len(X_list) == 0:
        logging.error("No features extracted!")
        return

    X = torch.vstack(X_list)
    y = torch.tensor(y_list)
    
    logging.info(f"Feature Matrix: {X.shape}")
    logging.info(f"Labels: {y.shape}")
    
    torch.save({"X": X, "y": y}, OUTPUT_TENSOR_FILE)
    logging.info(f"Saved dataset to {OUTPUT_TENSOR_FILE}")

if __name__ == "__main__":
    main()