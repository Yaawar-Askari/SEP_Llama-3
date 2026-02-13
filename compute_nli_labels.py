import torch
import pickle
import json
import numpy as np
import logging
import os
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Config
INPUT_FILE = "sep_xsum_generations.pkl"
OUTPUT_FILE = "sep_nli_labels.json"
NLI_MODEL = "microsoft/deberta-large-mnli"

def setup_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def check_entailment(model, tokenizer, prem, hyp):
    """Returns True if premise entails hypothesis."""
    # Encode
    input = tokenizer(prem, hyp, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        output = model(**input)
    
    # DeBERTa-MNLI: 0=Contradiction, 1=Neutral, 2=Entailment
    prediction = torch.argmax(output.logits, dim=1).item()
    return prediction == 2

def compute_semantic_entropy(generations, model, tokenizer):
    """
    Clusters generations by semantic meaning using RELAXED logic.
    """
    clusters = [] # List of lists of indices
    
    for i, gen in enumerate(generations):
        placed = False
        for cluster in clusters:
            # Check if this generation is semantically equivalent to the cluster representative
            representative = generations[cluster[0]]
            
            # --- RELAXED LOGIC ---
            # If A implies B OR B implies A, we treat them as the same cluster.
            # This handles cases where one summary is just a shorter version of the other.
            is_entailed_a = check_entailment(model, tokenizer, representative, gen)
            is_entailed_b = check_entailment(model, tokenizer, gen, representative)
            
            if is_entailed_a or is_entailed_b: 
                cluster.append(i)
                placed = True
                break
        
        if not placed:
            clusters.append([i])
            
    # --- CRITICAL RESTORED PART ---
    # Calculate Entropy: - Sum(p * log(p))
    n = len(generations)
    if n == 0:
        return 0.0
        
    probs = np.array([len(c) / n for c in clusters])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    return entropy

def main():
    setup_logger()
    
    if not os.path.exists(INPUT_FILE):
        logging.error(f"Input file {INPUT_FILE} not found.")
        return

    logging.info(f"Loading NLI Model: {NLI_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to("cuda")
    
    logging.info("Loading generations...")
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)
        
    filtered_data = []
    
    logging.info("Computing Semantic Entropy with NLI (Relaxed)...")
    
    # Stats counters
    stats = {"confident": 0, "hallucinated": 0, "discarded": 0}
    
    for item in tqdm(data):
        gens = [g for g in item['generations'] if g.strip()]
        if len(gens) < 2: 
            stats["discarded"] += 1
            continue
        
        entropy = compute_semantic_entropy(gens, model, tokenizer)
        
        # --- THRESHOLDS ---
        # With relaxed logic, we expect lower entropy scores overall.
        # 0.0 = Perfectly consistent (1 cluster)
        # > 0.5 = At least one major split (e.g. 4 vs 1 split yields ~0.72 entropy)
        
        label = -1
        if entropy < 0.2: 
            label = 0 # Confident
            stats["confident"] += 1
        elif entropy > 0.5: 
            label = 1 # Hallucinated
            stats["hallucinated"] += 1
        else:
            stats["discarded"] += 1
            
        if label != -1:
            filtered_data.append({
                "original_index": item['original_index'],
                "document": item['document'],
                "label": label,
                "score": entropy,
                # Save generations for inspection/contrastive extraction later
                "generations": item['generations'] 
            })
            
    logging.info(f"Processing Complete. Stats: {stats}")
    logging.info(f"Retained {len(filtered_data)} samples.")
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(filtered_data, f, indent=2)

if __name__ == "__main__":
    main()