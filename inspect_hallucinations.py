import pickle
import json
import random
import textwrap

# Files
GEN_FILE = "sep_xsum_generations.pkl"
LABEL_FILE = "sep_nli_labels.json"

def main():
    print(f"Loading labels from {LABEL_FILE}...")
    with open(LABEL_FILE, 'r') as f:
        labels = json.load(f) 

    print(f"Loading text generations from {GEN_FILE}...")
    with open(GEN_FILE, 'rb') as f:
        gen_data = pickle.load(f)

    # Create a lookup map: original_index -> list of generations
    # (Because the label file might be a subset of the full data)
    gen_map = {item['original_index']: item['generations'] for item in gen_data}

    # Separate indices by class
    hallucinated_indices = [i for i, x in enumerate(labels) if x['label'] == 1]
    confident_indices = [i for i, x in enumerate(labels) if x['label'] == 0]

    print(f"\nStats: {len(hallucinated_indices)} Hallucinated | {len(confident_indices)} Confident")

    def print_example(idx, category):
        item = labels[idx]
        original_idx = item['original_index']
        gens = gen_map.get(original_idx, [])
        
        print("\n" + "="*80)
        print(f"CATEGORY: {category} (Entropy Score: {item['score']:.4f})")
        print("="*80)
        
        print(f"\n[SOURCE DOCUMENT]:")
        # Print first 400 chars of source to give context
        doc_snippet = textwrap.shorten(item['document'], width=400, placeholder="...")
        print(doc_snippet)
        
        print(f"\n[MODEL GENERATIONS]:")
        # Print the first 3 generations to compare
        for i, g in enumerate(gens[:3]):
            clean_gen = g.strip().replace("\n", " ")
            print(f"  {i+1}. {clean_gen}")
            
        print("-" * 80)

    # --- SHOW EXAMPLES ---
    print("\n\n>>> INSPECTING 'HALLUCINATED' SAMPLES (Model was Confused) <<<")
    # Check if we have enough samples
    count = min(3, len(hallucinated_indices))
    for i in random.sample(hallucinated_indices, count):
        print_example(i, "HALLUCINATED")

    print("\n\n>>> INSPECTING 'CONFIDENT' SAMPLES (Model was Consistent) <<<")
    count = min(3, len(confident_indices))
    for i in random.sample(confident_indices, count):
        print_example(i, "CONFIDENT")

if __name__ == "__main__":
    main()