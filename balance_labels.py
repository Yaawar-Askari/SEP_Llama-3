import json
import numpy as np
import logging

INPUT_FILE = "sep_nli_labels.json"
OUTPUT_FILE = "sep_nli_labels.json" # Overwrite with fixed labels

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def main():
    logging.info(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
        
    # 1. Extract all scores to analyze distribution
    scores = np.array([item['score'] for item in data])
    
    # 2. Compute Statistics
    median_score = np.median(scores)
    mean_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    print("\n" + "="*40)
    print("ENTROPY SCORE DISTRIBUTION")
    print("="*40)
    print(f"Count:  {len(scores)}")
    print(f"Min:    {min_score:.4f}")
    print(f"Max:    {max_score:.4f}")
    print(f"Mean:   {mean_score:.4f}")
    print(f"Median: {median_score:.4f}  <-- NEW THRESHOLD")
    print("="*40 + "\n")
    
    # 3. Apply Median Split
    # <= Median -> Class 0 (Confident)
    # > Median  -> Class 1 (Hallucinated)
    
    new_counts = {0: 0, 1: 0}
    
    for item in data:
        if item['score'] <= median_score:
            item['label'] = 0
            new_counts[0] += 1
        else:
            item['label'] = 1
            new_counts[1] += 1
            
    logging.info(f"New Class Balance: {new_counts}")
    
    # 4. Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Saved balanced labels to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()