"""Train semantic entropy probes with ID and OOD evaluation.

Modes:
  id     -- In-distribution evaluation on a single dataset
  ood    -- Train on one dataset, evaluate on another
  matrix -- Full 4x4 cross-dataset AUROC matrix

Usage:
    python train_probe.py --mode id --dataset squad
    python train_probe.py --mode ood --train_dataset squad --eval_dataset trivia_qa
    python train_probe.py --mode matrix
"""
import os
import torch
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import logging

from common_utils import QA_DATASETS, OUTPUT_BASE, MODEL_NAME, NLI_MODEL

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Semantic Entropy Probes")
    parser.add_argument("--mode", choices=["id", "ood", "matrix"], default="id",
                        help="Evaluation mode")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset for ID mode")
    parser.add_argument("--train_dataset", type=str, default=None,
                        help="Training dataset for OOD mode")
    parser.add_argument("--eval_dataset", type=str, default=None,
                        help="Evaluation dataset for OOD mode")
    return parser.parse_args()


# ============================================================
# Core utilities (unchanged from previous version)
# ============================================================

def best_split(entropy):
    """
    OATML best_split: find threshold minimizing sum-of-squared reconstruction
    error (1D k-means with k=2).
    """
    ents = entropy.numpy() if isinstance(entropy, torch.Tensor) else entropy
    splits = np.linspace(1e-10, ents.max(), 100)
    best_mse = np.inf
    best_threshold = splits[0]

    for split in splits:
        low_idxs = ents < split
        high_idxs = ents >= split

        if low_idxs.sum() == 0 or high_idxs.sum() == 0:
            continue

        low_mean = np.mean(ents[low_idxs])
        high_mean = np.mean(ents[high_idxs])

        mse = (np.sum((ents[low_idxs] - low_mean) ** 2) +
               np.sum((ents[high_idxs] - high_mean) ** 2))

        if mse < best_mse:
            best_mse = mse
            best_threshold = split

    return best_threshold


def decide_layer_range(per_layer_aurocs, num_layers):
    """
    OATML decide_layer_range: find the contiguous range of >= 5 layers
    with the highest average AUROC.
    """
    best_mean = -np.inf
    best_range = (0, 5)

    for i in range(num_layers):
        for j in range(i + 1, num_layers + 1):
            if j - i < 5:
                continue
            mean_auroc = np.mean(per_layer_aurocs[i:j])
            if mean_auroc > best_mean:
                best_mean = mean_auroc
                best_range = (i, j)

    return best_mean, best_range


# ============================================================
# Data loading
# ============================================================

def load_dataset_features(dataset_name):
    """Load pre-computed features and binarize labels for a dataset."""
    data_file = os.path.join(OUTPUT_BASE, dataset_name, "all_layers.pt")
    logging.info(f"Loading {data_file}...")
    data = torch.load(data_file, weights_only=False)

    X_tbg = data['X_tbg']      # (N, num_layers, hidden_dim)
    X_slt = data['X_slt']      # (N, num_layers, hidden_dim)
    entropy = data['entropy']   # (N,) continuous

    # Binarize using THIS dataset's own best_split threshold
    threshold = best_split(entropy)
    y = (entropy >= threshold).long().numpy()

    num_samples, num_layers, hidden_dim = X_tbg.shape
    logging.info(f"  {dataset_name}: {num_samples} samples, {num_layers} layers, "
                 f"{hidden_dim}-dim, threshold={threshold:.4f}, "
                 f"low={np.sum(y==0)}, high={np.sum(y==1)}")

    return X_tbg, X_slt, y, threshold, entropy


# ============================================================
# Per-layer sweep and probe training
# ============================================================

def sweep_layers_on_split(X_np, y, num_layers, token_name=""):
    """Per-layer AUROC sweep with internal train/test split."""
    aurocs = []
    for layer_idx in range(num_layers):
        X_layer = X_np[layer_idx]

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_layer, y, test_size=0.1, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.2, random_state=42
        )

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_proba)
        aurocs.append(score)

        if layer_idx % 5 == 0 and token_name:
            logging.info(f"  {token_name} Layer {layer_idx:2d}: AUROC = {score:.4f}")

    return aurocs


def sweep_layers_full_train(X_train_np, y_train, X_eval_np, y_eval, num_layers):
    """Per-layer AUROC: train on full train set, evaluate on full eval set."""
    aurocs = []
    for layer_idx in range(num_layers):
        clf = LogisticRegression()
        clf.fit(X_train_np[layer_idx], y_train)
        y_proba = clf.predict_proba(X_eval_np[layer_idx])[:, 1]
        score = roc_auc_score(y_eval, y_proba)
        aurocs.append(score)
    return aurocs


def train_concat_probe_id(X_np, y, r_start, r_end, token_name):
    """Train probe with internal train/val/test split (ID evaluation)."""
    X_concat = np.concatenate([X_np[l] for l in range(r_start, r_end)], axis=1)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_concat, y, test_size=0.1, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42
    )

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_val_proba = clf.predict_proba(X_val)[:, 1]
    val_auroc = roc_auc_score(y_val, y_val_proba)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    test_auroc = roc_auc_score(y_test, y_proba)

    return {
        'range': (r_start, r_end),
        'concat_dim': X_concat.shape[1],
        'val_auroc': val_auroc,
        'test_acc': test_acc,
        'test_auroc': test_auroc,
        'y_test': y_test,
        'y_pred': y_pred,
    }


def train_concat_probe_ood(X_train_np, y_train, X_eval_np, y_eval, r_start, r_end):
    """Train on full train set, evaluate on full eval set (OOD evaluation)."""
    X_train_concat = np.concatenate(
        [X_train_np[l] for l in range(r_start, r_end)], axis=1)
    X_eval_concat = np.concatenate(
        [X_eval_np[l] for l in range(r_start, r_end)], axis=1)

    clf = LogisticRegression()
    clf.fit(X_train_concat, y_train)

    y_pred = clf.predict(X_eval_concat)
    y_proba = clf.predict_proba(X_eval_concat)[:, 1]
    eval_acc = accuracy_score(y_eval, y_pred)
    eval_auroc = roc_auc_score(y_eval, y_proba)

    return eval_auroc, eval_acc


# ============================================================
# Mode: In-Distribution
# ============================================================

def main_id(dataset_name):
    """Full ID evaluation for a single dataset."""
    X_tbg, X_slt, y, threshold, entropy = load_dataset_features(dataset_name)
    num_samples, num_layers, hidden_dim = X_tbg.shape

    results = {}
    for token_name, X_raw in [("TBG", X_tbg), ("SLT", X_slt)]:
        logging.info(f"\n{'='*40}")
        logging.info(f"[{dataset_name}] Processing {token_name}")
        logging.info(f"{'='*40}")

        X_np = X_raw.numpy().transpose(1, 0, 2)  # (num_layers, N, hidden_dim)

        # Per-layer sweep
        aurocs = sweep_layers_on_split(X_np, y, num_layers, token_name)

        # Best contiguous range
        best_mean, (r_start, r_end) = decide_layer_range(aurocs, num_layers)
        logging.info(f"{token_name} best range: [{r_start}, {r_end}) "
                     f"mean AUROC = {best_mean:.4f}")

        # Concatenated probe
        result = train_concat_probe_id(X_np, y, r_start, r_end, token_name)
        result['per_layer_aurocs'] = aurocs
        result['range_mean'] = best_mean
        results[token_name] = result

    # Print results
    print(f"\n{'='*60}")
    print(f"SEP RESULTS — {dataset_name} (In-Distribution)")
    print(f"{'='*60}")
    print(f"Model:        {MODEL_NAME}")
    print(f"NLI Model:    {NLI_MODEL}")
    print(f"SE threshold: {threshold:.4f} (best_split)")
    print(f"Samples:      {num_samples}")
    print(f"Class dist:   low={np.sum(y==0)}, high={np.sum(y==1)}")
    print()

    for tn in ["TBG", "SLT"]:
        r = results[tn]
        r_start, r_end = r['range']
        print(f"--- {tn} ---")
        print(f"  Layer range:   [{r_start}, {r_end})")
        print(f"  Feature dim:   {r['concat_dim']}")
        print(f"  Val AUROC:     {r['val_auroc']:.4f}")
        print(f"  Test AUROC:    {r['test_auroc']:.4f}")
        print(f"  Test Accuracy: {r['test_acc']:.4f}")
        print()

    # Per-layer table
    print("Per-layer AUROC:")
    print(f"  {'Layer':>5}  {'TBG':>8}  {'SLT':>8}")
    for i in range(num_layers):
        tbg_m = "*" if results['TBG']['range'][0] <= i < results['TBG']['range'][1] else " "
        slt_m = "*" if results['SLT']['range'][0] <= i < results['SLT']['range'][1] else " "
        print(f"  {i:>5}  {results['TBG']['per_layer_aurocs'][i]:>8.4f}{tbg_m} "
              f"{results['SLT']['per_layer_aurocs'][i]:>8.4f}{slt_m}")
    print("  (* = in best range)")

    # Classification reports
    for tn in ["TBG", "SLT"]:
        r = results[tn]
        print(f"\nClassification Report ({tn}):")
        print(classification_report(r['y_test'], r['y_pred'],
                                    target_names=["Low SE", "High SE"]))
    print("=" * 60)

    return results


# ============================================================
# Mode: Out-of-Distribution
# ============================================================

def main_ood(train_dataset, eval_dataset):
    """Train probe on train_dataset, evaluate on eval_dataset."""
    logging.info(f"OOD: train={train_dataset}, eval={eval_dataset}")

    # Load both datasets
    X_tbg_train, X_slt_train, y_train, thresh_train, _ = load_dataset_features(train_dataset)
    X_tbg_eval, X_slt_eval, y_eval, thresh_eval, _ = load_dataset_features(eval_dataset)

    num_layers = X_tbg_train.shape[1]

    results = {}
    for token_name, X_train_raw, X_eval_raw in [
        ("TBG", X_tbg_train, X_tbg_eval),
        ("SLT", X_slt_train, X_slt_eval),
    ]:
        logging.info(f"\n[{train_dataset}->{eval_dataset}] {token_name}")

        X_train_np = X_train_raw.numpy().transpose(1, 0, 2)
        X_eval_np = X_eval_raw.numpy().transpose(1, 0, 2)

        # Per-layer sweep: train on full train, eval on full eval
        aurocs = sweep_layers_full_train(X_train_np, y_train, X_eval_np, y_eval, num_layers)

        # Best layer range from cross-dataset per-layer AUROCs
        best_mean, (r_start, r_end) = decide_layer_range(aurocs, num_layers)
        logging.info(f"  {token_name} best range: [{r_start}, {r_end}) "
                     f"mean AUROC = {best_mean:.4f}")

        # Concatenated probe: train on full train, eval on full eval
        eval_auroc, eval_acc = train_concat_probe_ood(
            X_train_np, y_train, X_eval_np, y_eval, r_start, r_end)

        results[token_name] = {
            'range': (r_start, r_end),
            'range_mean': best_mean,
            'eval_auroc': eval_auroc,
            'eval_acc': eval_acc,
            'per_layer_aurocs': aurocs,
        }

    # Print results
    print(f"\n{'='*60}")
    print(f"SEP OOD RESULTS — Train: {train_dataset} -> Eval: {eval_dataset}")
    print(f"{'='*60}")
    print(f"Train threshold: {thresh_train:.4f}, Eval threshold: {thresh_eval:.4f}")
    print(f"Train samples:   {len(y_train)}, Eval samples: {len(y_eval)}")
    print()
    for tn in ["TBG", "SLT"]:
        r = results[tn]
        r_start, r_end = r['range']
        print(f"--- {tn} ---")
        print(f"  Layer range: [{r_start}, {r_end})")
        print(f"  Eval AUROC:  {r['eval_auroc']:.4f}")
        print(f"  Eval Acc:    {r['eval_acc']:.4f}")
    print("=" * 60)

    return results


# ============================================================
# Mode: Matrix (all cross-dataset pairs)
# ============================================================

def main_matrix():
    """Run all 4x4 train/eval combinations and print AUROC matrix."""
    # Check which datasets have data
    available = []
    for ds in QA_DATASETS:
        data_file = os.path.join(OUTPUT_BASE, ds, "all_layers.pt")
        if os.path.exists(data_file):
            available.append(ds)
        else:
            logging.warning(f"Skipping {ds}: {data_file} not found")

    if len(available) < 2:
        logging.error(f"Need at least 2 datasets for matrix evaluation. Found: {available}")
        return

    logging.info(f"Available datasets: {available}")

    # Pre-load all datasets
    all_data = {}
    for ds in available:
        X_tbg, X_slt, y, threshold, entropy = load_dataset_features(ds)
        all_data[ds] = {
            'X_tbg': X_tbg, 'X_slt': X_slt, 'y': y,
            'threshold': threshold, 'entropy': entropy,
        }

    # Compute matrix for both token positions
    for token_name in ["TBG", "SLT"]:
        auroc_matrix = np.zeros((len(available), len(available)))
        range_matrix = {}

        for i, train_ds in enumerate(available):
            X_train_raw = all_data[train_ds][f'X_{token_name.lower()}']
            y_train = all_data[train_ds]['y']
            num_layers = X_train_raw.shape[1]
            X_train_np = X_train_raw.numpy().transpose(1, 0, 2)

            for j, eval_ds in enumerate(available):
                X_eval_raw = all_data[eval_ds][f'X_{token_name.lower()}']
                y_eval = all_data[eval_ds]['y']
                X_eval_np = X_eval_raw.numpy().transpose(1, 0, 2)

                if train_ds == eval_ds:
                    # ID: use internal splits for layer selection + evaluation
                    aurocs = sweep_layers_on_split(X_train_np, y_train, num_layers)
                    _, (r_start, r_end) = decide_layer_range(aurocs, num_layers)

                    # Still compute ID AUROC with concat probe internal split
                    result = train_concat_probe_id(X_train_np, y_train, r_start, r_end, "")
                    auroc_matrix[i][j] = result['test_auroc']
                    range_matrix[(train_ds, eval_ds)] = (r_start, r_end)
                else:
                    # OOD: sweep across datasets
                    aurocs = sweep_layers_full_train(
                        X_train_np, y_train, X_eval_np, y_eval, num_layers)
                    _, (r_start, r_end) = decide_layer_range(aurocs, num_layers)

                    eval_auroc, _ = train_concat_probe_ood(
                        X_train_np, y_train, X_eval_np, y_eval, r_start, r_end)
                    auroc_matrix[i][j] = eval_auroc
                    range_matrix[(train_ds, eval_ds)] = (r_start, r_end)

                logging.info(f"  {token_name} {train_ds:>10} -> {eval_ds:<10}: "
                             f"AUROC={auroc_matrix[i][j]:.4f} "
                             f"layers=[{range_matrix[(train_ds,eval_ds)][0]},"
                             f"{range_matrix[(train_ds,eval_ds)][1]})")

        # Print matrix
        print(f"\n{'='*70}")
        print(f"CROSS-DATASET AUROC MATRIX — {token_name}")
        print(f"{'='*70}")
        print(f"Model: {MODEL_NAME} | NLI: {NLI_MODEL}")
        print()

        # Header
        label = "Train \\ Eval"
        header = f"{label:>14}"
        for ds in available:
            header += f"  {ds:>12}"
        print(header)
        print("-" * len(header))

        # Rows
        for i, train_ds in enumerate(available):
            row = f"{train_ds:>14}"
            for j, eval_ds in enumerate(available):
                val = auroc_matrix[i][j]
                marker = " *" if train_ds == eval_ds else "  "
                row += f"  {val:>10.4f}{marker}"
            print(row)

        print()
        print("  (* = in-distribution, others = OOD)")

        # Summary stats
        diag = np.diag(auroc_matrix)
        off_diag = auroc_matrix[~np.eye(len(available), dtype=bool)]
        print(f"\n  ID mean AUROC:  {np.mean(diag):.4f} (diagonal)")
        print(f"  OOD mean AUROC: {np.mean(off_diag):.4f} (off-diagonal)")
        print(f"  OOD/ID ratio:   {np.mean(off_diag)/np.mean(diag):.4f}")

        # Per-dataset thresholds
        print(f"\n  SE thresholds:")
        for ds in available:
            t = all_data[ds]['threshold']
            n = len(all_data[ds]['y'])
            low = np.sum(all_data[ds]['y'] == 0)
            high = np.sum(all_data[ds]['y'] == 1)
            print(f"    {ds:>12}: threshold={t:.4f}, N={n}, low={low}, high={high}")

    print(f"\n{'='*70}")


# ============================================================
# Main dispatch
# ============================================================

def main():
    args = parse_args()

    if args.mode == "id":
        if args.dataset is None:
            logging.error("--dataset required for ID mode")
            return
        main_id(args.dataset)

    elif args.mode == "ood":
        if args.train_dataset is None or args.eval_dataset is None:
            logging.error("--train_dataset and --eval_dataset required for OOD mode")
            return
        main_ood(args.train_dataset, args.eval_dataset)

    elif args.mode == "matrix":
        main_matrix()


if __name__ == "__main__":
    main()
