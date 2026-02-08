import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import logging

# Setup Logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

DATA_FILE = "sep_dataset.pt"

def main():
    # 1. Load Data
    logging.info(f"Loading {DATA_FILE}...")
    try:
        data = torch.load(DATA_FILE)
    except FileNotFoundError:
        logging.error(f"File {DATA_FILE} not found. Did you run extract_features.py?")
        return

    X = data['X'].numpy()
    y = data['y'].numpy()

    logging.info(f"Dataset Shape: X={X.shape}, y={y.shape}")
    
    # Check Class Distribution
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    logging.info(f"Class Distribution: {dist} (0=Confident, 1=Hallucinated)")

    if len(unique) < 2:
        logging.error("Only one class detected! Cannot train classifier. Try adjusting thresholds in compute_rouge_labels.py")
        return

    # 2. Preprocessing
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalize (Crucial for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train Logistic Regression
    logging.info("Training Logistic Regression Probe...")
    # C=1.0 is standard L2 regularization. 'liblinear' is good for small datasets.
    clf = LogisticRegression(random_state=42, C=1.0, solver='liblinear', max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    # 4. Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba)

    print("\n" + "="*30)
    print("PROBE RESULTS")
    print("="*30)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUROC:    {auroc:.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Confident", "Hallucinated"]))
    print("="*30)

if __name__ == "__main__":
    main()