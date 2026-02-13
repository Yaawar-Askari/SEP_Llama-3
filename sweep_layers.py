import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

DATA_FILE = "sep_all_layers.pt"
PLOT_FILE = "layer_sweep_auroc.png"

def main():
    logging.info(f"Loading {DATA_FILE}...")
    data = torch.load(DATA_FILE)
    X_all = data['X'].numpy() # (N, 33, 4096)
    y = data['y'].numpy()
    
    num_layers = X_all.shape[1]
    aurocs = []
    
    logging.info(f"Sweeping {num_layers} layers...")
    
    # Iterate through each layer
    for layer_idx in range(num_layers):
        X_layer = X_all[:, layer_idx, :]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_layer, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize (Critical for convergence)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train Probe
        clf = LogisticRegression(solver='liblinear', max_iter=500, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_probs = clf.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_probs)
        aurocs.append(score)
        
        if layer_idx % 5 == 0:
            print(f"Layer {layer_idx}: AUROC = {score:.4f}")

    # Identify Best Layer
    best_layer = np.argmax(aurocs)
    best_score = aurocs[best_layer]
    print(f"\nBEST RESULT: Layer {best_layer} with AUROC {best_score:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_layers), aurocs, marker='o', linestyle='-', color='b')
    plt.axvline(best_layer, color='r', linestyle='--', label=f'Best Layer ({best_layer})')
    plt.title("Llama-3 Hallucination Detection by Layer")
    plt.xlabel("Layer Index (0 = Embeddings, 32 = Output)")
    plt.ylabel("Test AUROC")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(PLOT_FILE)
    print(f"Saved sweep plot to {PLOT_FILE}")

if __name__ == "__main__":
    main()