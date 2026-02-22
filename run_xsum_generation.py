import os
import torch
import random
import pickle
import logging
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import sys

sys.path.append(os.path.abspath("semantic_uncertainty"))

# Reuse OATML model wrapper
from semantic_uncertainty.uncertainty.models.huggingface_models import HuggingfaceModel
from sep_utils import format_prompt, setup_simple_logger
from common_utils import MODEL_NAME, NUM_SAMPLES_XSUM, NUM_GENERATIONS_XSUM, TEMPERATURE, SEED

# Configuration
# Update this line
OUTPUT_FILE = "sep_xsum_generations.pkl"

def main():
    setup_simple_logger()
    
    # 1. Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # 2. Load Data
    logging.info("Loading XSum dataset...")
    dataset = load_dataset("xsum", split="train")
    
    # Sample indices
    indices = random.sample(range(len(dataset)), NUM_SAMPLES_XSUM)
    logging.info(f"Sampled {len(indices)} indices.")

    # 3. Initialize Model (Reusing OATML Class)
    # stop_sequences='default' loads '.' and '\n' from base_model.py which might be too aggressive for summaries.
    # We pass empty list or rely on EOS.
    logging.info(f"Loading Model: {MODEL_NAME}...")
    model = HuggingfaceModel(
        model_name=MODEL_NAME, 
        stop_sequences=[],  
        max_new_tokens=50
    )
    model.token_limit = 8192

    data_store = []

    # 4. Generation Loop
    logging.info("Starting generation...")
    for i, idx in tqdm(enumerate(indices), total=NUM_SAMPLES_XSUM):
        document = dataset[idx]['document']
        
        # CRITICAL: Apply shared formatting
        prompt_text = format_prompt(model.tokenizer, document)
        
        # Wrapper for multiple generations
        generations = []
        for gen_i in range(NUM_GENERATIONS_XSUM):
            # OATML `predict` returns: (answer, log_likelihoods, hidden_states)
            # We strictly reuse their predict method.
            try:
                answer, _, _ = model.predict(prompt_text, temperature=TEMPERATURE)
                generations.append(answer)
            except Exception as e:
                logging.error(f"Generation failed for idx {idx}: {e}")
                generations.append("") # Placeholder

        data_store.append({
            "original_index": idx,
            "document": document,
            "prompt_used": prompt_text, # Saved for verification
            "generations": generations
        })
        
        # Periodic save
        if (i + 1) % 100 == 0:
            with open(OUTPUT_FILE, "wb") as f:
                pickle.dump(data_store, f)

    # Final Save
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data_store, f)
    logging.info(f"Finished. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()