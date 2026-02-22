import torch
import json
import logging
import os 
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath("semantic_uncertainty"))
from semantic_uncertainty.uncertainty.models.huggingface_models import HuggingfaceModel
from sep_utils import format_prompt, setup_simple_logger
from common_utils import MODEL_NAME

INPUT_LABEL_FILE = "sep_nli_labels.json"
OUTPUT_TENSOR_FILE = "sep_all_layers.pt"

def get_all_layers_hidden_states(model_instance, input_ids, attention_mask):
    """Extract hidden states from ALL layers for the last token of the prompt."""
    hf_model = model_instance.model 
    
    with torch.no_grad():
        outputs = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False
        )
    
    # outputs.hidden_states is a tuple of 33 tensors (embeddings + 32 layers)
    # We stack them into shape (Num_Layers, Batch, Seq, Dim) -> (Num_Layers, Dim)
    
    last_token_idx = attention_mask.sum(1) - 1
    collected_layers = []
    
    for layer_tensor in outputs.hidden_states:
        # layer_tensor: (Batch, Seq, Dim)
        # Extract last token feature
        feature = layer_tensor[0, last_token_idx, :].cpu() # Shape (1, Dim)
        collected_layers.append(feature)
        
    # Stack: (Num_Layers, Dim)
    return torch.vstack(collected_layers)

def main():
    setup_simple_logger()
    
    logging.info("Loading filtered labels...")
    with open(INPUT_LABEL_FILE, "r") as f:
        dataset = json.load(f)
        
    logging.info(f"Loading Model: {MODEL_NAME}...")
    # Token limit doesn't strictly matter here as we process prompts, but keeping safety
    wrapper = HuggingfaceModel(MODEL_NAME, stop_sequences=[], max_new_tokens=1)
    
    X_list = [] # Will become (N_samples, N_layers, Dim)
    y_list = []
    
    logging.info(f"Extracting all layers for {len(dataset)} samples...")
    for item in tqdm(dataset):
        prompt_text = format_prompt(wrapper.tokenizer, item['document'])
        inputs = wrapper.tokenizer(prompt_text, return_tensors="pt").to("cuda")
        
        # Shape: (33, 4096)
        features = get_all_layers_hidden_states(wrapper, inputs.input_ids, inputs.attention_mask)
        
        X_list.append(features)
        y_list.append(item['label'])
        
    # Stack into final tensor
    # X shape: (N_samples, N_layers, Hidden_Dim)
    X = torch.stack(X_list)
    y = torch.tensor(y_list)
    
    logging.info(f"Final Tensor Shape: {X.shape}")
    torch.save({"X": X, "y": y}, OUTPUT_TENSOR_FILE)
    logging.info(f"Saved to {OUTPUT_TENSOR_FILE}")

if __name__ == "__main__":
    main()