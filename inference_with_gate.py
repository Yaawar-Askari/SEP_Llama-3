"""SEP-Triggered Lookback Gating — Inference Script

Pipeline (per sample):
  1. Load pre-computed TBG embedding from generations.pkl (no LLM re-run needed).
  2. SEP probe scores the embedding → uncertainty ∈ [0, 1].
  3. If uncertainty > sep_threshold → TRIGGER:
       - Attach LookbackGateController hooks to upper LLM layers.
       - Re-generate with output_attentions=True so hooks can read per-head attention.
       - Per-head Lookback Ratio: LR[l,h] = Σ(attn to prompt) / Σ(attn to all).
       - Gate = σ(LR × alpha)  applied to pre-o_proj activations per head.
       - Generation uses the suppressed-head output.
  4. If uncertainty ≤ sep_threshold → passthrough (reuse existing most_likely_answer).
  5. Compute accuracy (F1 ≥ 50%) for both paths and compare.

Saved to: output/{dataset}/gated_results.pkl

Usage:
    python inference_with_gate.py --dataset squad
    python inference_with_gate.py --dataset trivia_qa --alpha 15.0 --sep_threshold 0.6
    python inference_with_gate.py --dataset squad --token_type SLT
"""
import os
import sys
import gc
import pickle
import logging
import argparse
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "semantic_uncertainty"))

from uncertainty.models.huggingface_models import HuggingfaceModel
from common_utils import (
    MODEL_NAME, QA_DATASETS, OUTPUT_BASE,
    TEMPERATURE_LOW, MAX_NEW_TOKENS,
)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


# ------------------------------------------------------------------ #
# Lookback Gate — PyTorch forward hooks, no transformers fork needed  #
# ------------------------------------------------------------------ #

class LookbackGateController:
    """Attaches forward hooks to LlamaAttention layers in the upper third of
    the network (or a custom range).  When self.triggered is True, each
    generated token's o_proj input is re-scaled per-head by σ(LR × alpha),
    where LR = attention on prompt tokens / total attention.

    Hook design:
      Hook-A  (forward_hook  on self_attn)     → cache attn_weights for this step
      Hook-B  (forward_pre_hook on o_proj)     → apply per-head gate using cached weights

    NOTE: requires output_attentions=True in model.generate() so that
    LlamaAttention actually computes and returns attention weights.
    """

    def __init__(self, model, context_length, alpha=10.0, layer_range=None):
        self.triggered       = False
        self.context_length  = context_length
        self.alpha           = alpha
        self._attn_cache     = {}   # layer_idx → Tensor (B, H, q, kv)
        self._hooks          = []

        layers = model.model.layers
        n      = len(layers)

        if layer_range is None:
            # Upper third — where "answer-extraction" attention heads live
            # (Yu et al. EMNLP 2024; upper layers ~21-31 for Llama-2-7b)
            layer_range = range(n * 2 // 3, n)

        for idx in layer_range:
            attn_mod = layers[idx].self_attn

            # A: capture attention weights produced by this layer
            hA = attn_mod.register_forward_hook(
                self._make_capture_hook(idx)
            )
            # B: gate the pre-projection hidden states
            hB = attn_mod.o_proj.register_forward_pre_hook(
                self._make_gate_pre_hook(idx)
            )
            self._hooks.extend([hA, hB])

        logging.info(f"LookbackGateController: hooks on layers "
                     f"{list(layer_range)}, alpha={alpha}, "
                     f"context_length={context_length}")

    # ---- Hook A: capture attn_weights -------------------------------- #

    def _make_capture_hook(self, layer_idx):
        def hook(module, inp, output):
            # output = (attn_output, attn_weights_or_None, past_key_value)
            self._attn_cache[layer_idx] = output[1]   # None if output_attentions=False
        return hook

    # ---- Hook B: per-head gate ---------------------------------------- #

    def _make_gate_pre_hook(self, layer_idx):
        def pre_hook(module, inp):
            if not self.triggered:
                return   # no-op

            attn_weights = self._attn_cache.get(layer_idx)
            if attn_weights is None:
                return   # output_attentions was False — skip

            # Shape: (B, num_heads, q_len, kv_len)
            q_len = attn_weights.shape[2]

            #  Skip the prompt-processing step (q_len > 1 means full prompt pass
            #  with KV cache; we only gate individual generated-token steps).
            if q_len > 1:
                return

            x = inp[0]   # (B=1, q_len=1, hidden)  — pre-o_proj tensor
            B, _, hidden = x.shape
            num_heads = attn_weights.shape[1]
            head_dim  = hidden // num_heads
            ctx       = self.context_length

            # Lookback Ratio per head (Chuang et al. Eq. 1)
            # attn_weights[:, :, -1, :] = attention FROM last query token
            attn_row  = attn_weights[0, :, -1, :]          # (H, kv_len)
            attn_ctx  = attn_row[:, :ctx].sum(-1)           # (H,)  attention on prompt
            attn_new  = attn_row[:, ctx:].sum(-1)           # (H,)  attention on generated
            lr        = attn_ctx / (attn_ctx + attn_new + 1e-10)   # (H,)  ∈ [0, 1]

            # Gate: heads with low LR (ignoring prompt) are suppressed
            gate = torch.sigmoid(lr * self.alpha)           # (H,) ∈ ~(0, 1)

            # Apply per-head scaling to pre-projection activations
            x_r   = x.view(B, 1, num_heads, head_dim)
            gate  = gate.to(x.device).view(1, 1, num_heads, 1)
            x_out = (x_r * gate).view(B, 1, hidden)

            return (x_out,)     # returned tuple replaces the hook input

        return pre_hook

    # ---- Control -------------------------------------------- #

    def trigger(self):
        """Activate gating for the next generation call."""
        self.triggered = True
        self._attn_cache.clear()

    def reset(self):
        """Deactivate gating (passthrough mode)."""
        self.triggered = False
        self._attn_cache.clear()

    def remove(self):
        """Detach all hooks (call once, after all inference is done)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ------------------------------------------------------------------ #
# Accuracy helper (matches run_qa_generation.py)                      #
# ------------------------------------------------------------------ #

def compute_f1(prediction, ground_truth):
    pred_tokens = prediction.lower().split()
    gt_tokens   = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common  = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_accuracy(prediction, answers):
    if not answers:
        return 0.0
    max_f1 = max(compute_f1(prediction, ans) for ans in answers)
    return 1.0 if max_f1 * 100 >= 50.0 else 0.0


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(description="SEP-Triggered Lookback Gating Inference")
    parser.add_argument("--dataset", required=True, choices=QA_DATASETS)
    parser.add_argument("--token_type", choices=["TBG", "SLT"], default="TBG",
                        help="Which SEP probe to use (default: TBG)")
    parser.add_argument("--sep_threshold", type=float, default=0.5,
                        help="SEP uncertainty score above which gating is triggered "
                             "(default: 0.5 — fires for 'High SE' predictions)")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Gate sharpness: gate = sigmoid(LR × alpha). "
                             "Higher → harder gating (default: 10.0)")
    parser.add_argument("--layer_range", type=str, default=None,
                        help="Comma-separated start,end for gated layers, e.g. '21,32'. "
                             "Default: upper third of all layers.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset

    # ---- Paths ----
    out_dir     = os.path.join(OUTPUT_BASE, dataset)
    gen_file    = os.path.join(out_dir, "generations.pkl")
    probe_file  = os.path.join(out_dir, f"sep_probe_{args.token_type}.pkl")
    result_file = os.path.join(out_dir, "gated_results.pkl")

    for path, desc in [(gen_file, "generations.pkl"),
                       (probe_file, f"sep_probe_{args.token_type}.pkl")]:
        if not os.path.exists(path):
            logging.error(f"{desc} not found at {path}. "
                          f"Run run_qa_generation.py and "
                          f"train_probe.py --save_probe first.")
            return

    # ---- Load ----
    logging.info(f"Loading generations from {gen_file} ...")
    with open(gen_file, "rb") as f:
        gen_data = pickle.load(f)

    logging.info(f"Loading SEP probe from {probe_file} ...")
    with open(probe_file, "rb") as f:
        probe_bundle = pickle.load(f)

    clf       = probe_bundle['clf']
    r_start   = probe_bundle['r_start']
    r_end     = probe_bundle['r_end']
    se_thresh = probe_bundle['threshold']   # SE binarization threshold (informational)

    logging.info(f"Probe: layers [{r_start},{r_end}), SE threshold={se_thresh:.4f}")
    logging.info(f"Gate trigger: SEP uncertainty > {args.sep_threshold}")

    # ---- Parse custom layer range ----
    layer_range = None
    if args.layer_range is not None:
        s, e = args.layer_range.split(",")
        layer_range = range(int(s), int(e))

    # ---- Load LLM ----
    logging.info(f"Loading model: {MODEL_NAME} ...")
    hf_model = HuggingfaceModel(
        model_name=MODEL_NAME,
        stop_sequences='default',
        max_new_tokens=MAX_NEW_TOKENS,
    )
    raw_model   = hf_model.model      # the actual HF CausalLM
    tokenizer   = hf_model.tokenizer
    stop_seqs   = hf_model.stop_sequences

    # ---- Score all samples with SEP probe (no LLM needed here) ----
    logging.info("Scoring all samples with SEP probe ...")
    sep_scores   = []
    valid_items  = []

    for item in gen_data:
        tbg = item.get('tbg_embedding')   # (num_layers, 1, hidden_dim)
        slt = item.get('slt_embedding')

        emb = tbg if args.token_type == "TBG" else slt
        if emb is None:
            sep_scores.append(None)
            valid_items.append(False)
            continue

        # Squeeze middle dim: (num_layers, 1, hidden_dim) → (num_layers, hidden_dim)
        emb_sq = emb.squeeze(1) if emb.dim() == 3 else emb   # (L, D)

        # Concatenate layers in probe range  → (1, feature_dim)
        feature = np.concatenate(
            [emb_sq[l].numpy() for l in range(r_start, r_end)], axis=0
        )[np.newaxis, :]

        score = clf.predict_proba(feature)[0, 1]   # probability of High-SE
        sep_scores.append(score)
        valid_items.append(True)

    n_triggered      = sum(1 for s in sep_scores if s is not None and s > args.sep_threshold)
    n_passthrough    = sum(1 for s, v in zip(sep_scores, valid_items)
                           if v and s <= args.sep_threshold)
    n_invalid        = sum(1 for v in valid_items if not v)

    logging.info(f"SEP scored {len(gen_data)} samples: "
                 f"triggered={n_triggered}, passthrough={n_passthrough}, "
                 f"no-embedding={n_invalid}")

    # ---- Attach gate hooks (they are no-ops until controller.trigger() is called) ----
    # We need one forward pass to know num_layers; take it from the probe bundle.
    controller = LookbackGateController(
        model=raw_model,
        context_length=0,    # placeholder — updated per-sample
        alpha=args.alpha,
        layer_range=layer_range,
    )

    # ---- Inference loop ----
    results = []
    acc_gate  = []
    acc_pass  = []

    logging.info("Running gated inference ...")
    for i, item in enumerate(tqdm(gen_data)):
        score      = sep_scores[i]
        answers    = item.get('answers', [])
        existing   = item.get('most_likely_answer', "")
        prompt     = item.get('prompt_used', "")

        if score is None or prompt == "":
            # No embedding or no prompt — just carry forward existing answer
            results.append({**item, 'gated_answer': existing,
                            'sep_score': score, 'gate_triggered': False})
            acc_pass.append(compute_accuracy(existing, answers))
            continue

        triggered = score > args.sep_threshold

        if not triggered:
            # --- Passthrough: reuse existing answer from generations.pkl ---
            results.append({**item, 'gated_answer': existing,
                            'sep_score': float(score), 'gate_triggered': False})
            acc_pass.append(compute_accuracy(existing, answers))
        else:
            # --- Triggered: re-generate with Lookback Gate active ---
            inputs = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to("cuda")

            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']

            n_prompt_tokens = inputs['input_ids'].shape[1]

            # Update context_length on the controller before triggering
            controller.context_length = n_prompt_tokens
            controller.trigger()

            try:
                with torch.no_grad():
                    out = raw_model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,           # greedy during gated pass
                        temperature=1.0,
                        output_attentions=True,    # required for hook-A to capture weights
                        output_scores=False,
                        output_hidden_states=False,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # Decode only newly generated tokens
                gen_tokens = out.sequences[0][n_prompt_tokens:]
                gated_ans  = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                # Strip stop sequences
                for stop in stop_seqs:
                    if gated_ans.endswith(stop):
                        gated_ans = gated_ans[:-len(stop)].strip()
                        break

            except Exception as e:
                logging.error(f"Sample {i}: gated generation failed — {e}")
                gated_ans = existing     # fall back to existing answer

            finally:
                controller.reset()

            results.append({**item, 'gated_answer': gated_ans,
                            'sep_score': float(score), 'gate_triggered': True})
            acc_gate.append(compute_accuracy(gated_ans, answers))

            if (i + 1) % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    # ---- Detach hooks ----
    controller.remove()

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"GATED INFERENCE RESULTS — {dataset}")
    print(f"{'='*60}")
    print(f"Model:         {MODEL_NAME}")
    print(f"SEP probe:     {args.token_type}  (layers [{r_start},{r_end}))")
    print(f"Gate trigger:  SEP score > {args.sep_threshold}")
    print(f"Gate alpha:    {args.alpha}")
    print()
    print(f"Total samples:      {len(results)}")
    print(f"  Triggered (gated): {n_triggered}")
    print(f"  Passthrough:       {n_passthrough}")
    print(f"  No embedding:      {n_invalid}")
    print()

    if acc_gate:
        print(f"Accuracy (gated samples):")
        print(f"  Before gating (original answer): "
              f"{np.mean([compute_accuracy(r['most_likely_answer'], r['answers']) for r in results if r['gate_triggered']]):.4f}")
        print(f"  After  gating (new answer):      {np.mean(acc_gate):.4f}")

    if acc_pass:
        print(f"Accuracy (passthrough, no change): {np.mean(acc_pass):.4f}")

    all_gated_acc = [compute_accuracy(r['gated_answer'], r['answers']) for r in results]
    all_orig_acc  = [compute_accuracy(r['most_likely_answer'], r['answers']) for r in results
                     if r['most_likely_answer'] != ""]
    print()
    print(f"Overall accuracy (original answers):  {np.mean(all_orig_acc):.4f}")
    print(f"Overall accuracy (after gating):      {np.mean(all_gated_acc):.4f}")
    print("=" * 60)

    # ---- Save ----
    with open(result_file, "wb") as f:
        pickle.dump(results, f)
    logging.info(f"Saved {len(results)} results → {result_file}")


if __name__ == "__main__":
    main()
