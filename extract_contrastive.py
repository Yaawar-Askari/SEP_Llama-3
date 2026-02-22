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

# INPUT: Uses your new BALANCED NLI labels
INPUT_LABEL_FILE = "sep_nli_labels.json" 
OUTPUT_TENSOR_FILE = "sep_contrastive_data.pt"

def get_contrastive_states(model_instance, prompt_text, full_text):
    """
    Extracts hidden states from ALL layers for:
    1. The last token of the PROMPT.
    2. The last token of the GENERATION.
    """
    hf_model = model_instance.model 
    tokenizer = model_instance.tokenizer
    
    # Tokenize prompt separately (without generation)
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    prompt_len = prompt_inputs.input_ids.shape[1]

    # Tokenize full text
    full_inputs = tokenizer(full_text, return_tensors="pt").to("cuda")
    input_ids = full_inputs.input_ids

    
    # Last token of prompt is at index `prompt_len - 1`
    # Last token of generation is at index `-1`
    
    with torch.no_grad():
        outputs = hf_model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False
        )
    
    prompt_layers = []
    gen_layers = []
    
    for layer_tensor in outputs.hidden_states:
        # layer_tensor shape: (Batch, Seq, Dim) -> Take batch 0
        seq_tensor = layer_tensor[0] 
        
        # Safety check for length (handle rare truncation edge cases)
        if seq_tensor.shape[0] < prompt_len:
            h_prompt = seq_tensor[-1, :].cpu()
            h_gen = seq_tensor[-1, :].cpu()
        else:
            h_prompt = seq_tensor[prompt_len - 1, :].cpu()
            h_gen = seq_tensor[-1, :].cpu()
        
        prompt_layers.append(h_prompt)
        gen_layers.append(h_gen)
        
    # Stack -> (Num_Layers, Dim)
    return torch.stack(prompt_layers), torch.stack(gen_layers)

def main():
    setup_simple_logger()
    
    if not os.path.exists(INPUT_LABEL_FILE):
        logging.error(f"Input file {INPUT_LABEL_FILE} not found.")
        return

    logging.info("Loading balanced labels...")
    with open(INPUT_LABEL_FILE, "r") as f:
        dataset = json.load(f)
        
    logging.info(f"Loading Model: {MODEL_NAME}...")
    wrapper = HuggingfaceModel(MODEL_NAME, stop_sequences=[], max_new_tokens=1)
    
    data_store = []
    
    logging.info(f"Extracting contrastive features for {len(dataset)} samples...")
    for item in tqdm(dataset):
        prompt_text = format_prompt(wrapper.tokenizer, item['document'])
        
        # We re-generate 1 sample to get the 'end' state corresponding to the prompt.
        # (Ideally we reuse the text, but this ensures state alignment).
        gen_text, _, _ = wrapper.predict(prompt_text, temperature=0.7)
        gen_text = gen_text.strip()
        full_text = prompt_text + gen_text
        
        h_p, h_g = get_contrastive_states(wrapper, prompt_text, full_text)
        
        data_store.append({
            "h_prompt": h_p, 
            "h_gen": h_g,    
            "label": item['label']
        })
        
    # Collate into tensors
    X_prompt = torch.stack([d['h_prompt'] for d in data_store])
    X_gen = torch.stack([d['h_gen'] for d in data_store])
    y = torch.tensor([d['label'] for d in data_store])
    
    logging.info(f"Contrastive Tensors Saved: {X_prompt.shape}")
    
    torch.save({
        "X_prompt": X_prompt, 
        "X_gen": X_gen, 
        "y": y
    }, OUTPUT_TENSOR_FILE)
    logging.info(f"Saved to {OUTPUT_TENSOR_FILE}")

if __name__ == "__main__":
    main()