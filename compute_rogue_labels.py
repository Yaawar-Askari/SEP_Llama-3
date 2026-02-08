import pickle
import numpy as np
import json
from rouge_score import rouge_scorer
from sep_utils import setup_simple_logger
import logging

INPUT_FILE = "sep_xsum_generations.pkl"
OUTPUT_FILE = "sep_filtered_labels.json"

def compute_pairwise_rouge(generations):
    """Compute average pairwise Rouge-L F1 score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    n = len(generations)
    if n < 2: return 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            # score returns (precision, recall, fmeasure)
            s = scorer.score(generations[i], generations[j])
            scores.append(s['rougeL'].fmeasure)
            
    return np.mean(scores)

def main():
    setup_simple_logger()
    logging.info("Loading generations...")
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)
        
    filtered_data = []
    stats = {"confident": 0, "hallucinated": 0, "discarded": 0}
    
    logging.info(f"Processing {len(data)} items...")
    
    for item in data:
        gens = item['generations']
        # Filter empty generations if any errors occurred
        gens = [g for g in gens if g.strip()]
        
        if len(gens) < 2:
            stats["discarded"] += 1
            continue
            
        avg_rouge = compute_pairwise_rouge(gens)
        
        label = -1
        if avg_rouge > 0.7:
            label = 0 # Confident (Low Entropy)
            stats["confident"] += 1
        elif avg_rouge < 0.3:
            label = 1 # Hallucinated (High Entropy)
            stats["hallucinated"] += 1
        else:
            stats["discarded"] += 1
            
        if label != -1:
            filtered_data.append({
                "original_index": item['original_index'],
                "document": item['document'],
                # We do not need generations anymore, just the prompt source
                "label": label,
                "score": avg_rouge
            })
            
    logging.info(f"Results: {stats}")
    logging.info(f"Retained {len(filtered_data)} samples.")
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(filtered_data, f, indent=2)

if __name__ == "__main__":
    main()