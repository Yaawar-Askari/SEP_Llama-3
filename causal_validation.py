"""Causal Validation of SEP-Triggered Lookback Gating

Two tests adapted from the project plan (Yu et al. + Chuang et al.):

Test 1 – Knockout Test
    Hypothesis: Heads flagged as "hallucinating" by the gate actually CAUSE the
    degraded output. If we zero them out completely (gate=0, harder than the
    sigmoid gate), accuracy should drop further or stay the same, not improve.
    
    More importantly: gated re-generations should be BETTER than the original
    answers on triggered samples. The knockout comparison exposes the delta.

Test 2 – Blindness Test
    Hypothesis: Heads with HIGH Lookback Ratio (strongly attending to the
    prompt/context) are the "grounding" heads. Suppressing them (forcing gate=0
    for *high-LR* heads instead of low-LR heads) should DEGRADE output quality,
    demonstrating these heads are causally responsible for correct grounding.

Inputs expected in output/{dataset}/:
    - gated_results.pkl      (from inference_with_gate.py)
    - sep_probe_{token}.pkl  (from train_probe.py --save_probe)

Usage:
    python causal_validation.py --dataset squad
    python causal_validation.py --dataset trivia_qa --num_samples 50
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
    MAX_NEW_TOKENS,
)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)


# ------------------------------------------------------------------ #
# Forced Gate Controller                                               #
# ------------------------------------------------------------------ #

class ForcedGateController:
    """Like LookbackGateController but applies a FORCED gate strategy:
    
      mode='zero_all'   : gate = 0 for ALL heads in target layers (knockout)
      mode='zero_high'  : gate = 0 for heads with LR ≥ lr_cutoff (blindness)
      mode='zero_low'   : gate = 0 for heads with LR < lr_cutoff  (normal gate, hard version)

    This bypasses the sigmoid and applies binary gating to cleanly test causality.
    """

    def __init__(self, model, context_length, layer_range=None,
                 mode='zero_all', lr_cutoff=0.5):
        self.triggered      = False
        self.context_length = context_length
        self.mode           = mode
        self.lr_cutoff      = lr_cutoff
        self._attn_cache    = {}
        self._hooks         = []

        layers = model.model.layers
        n = len(layers)
        if layer_range is None:
            layer_range = range(n * 2 // 3, n)

        for idx in layer_range:
            attn = layers[idx].self_attn
            self._hooks.append(attn.register_forward_hook(
                self._make_capture_hook(idx)
            ))
            self._hooks.append(attn.o_proj.register_forward_pre_hook(
                self._make_forced_gate_hook(idx)
            ))

    def _make_capture_hook(self, idx):
        def hook(module, inp, output):
            self._attn_cache[idx] = output[1]
        return hook

    def _make_forced_gate_hook(self, idx):
        def pre_hook(module, inp):
            if not self.triggered:
                return
            attn_weights = self._attn_cache.get(idx)
            if attn_weights is None:
                return
            if attn_weights.shape[2] > 1:   # Skip prompt-processing step
                return

            x         = inp[0]
            B, _, H_  = x.shape
            num_heads = attn_weights.shape[1]
            head_dim  = H_ // num_heads
            ctx       = self.context_length

            attn_row = attn_weights[0, :, -1, :]              # (H, kv)
            attn_ctx = attn_row[:, :ctx].sum(-1)              # (H,)
            attn_new = attn_row[:, ctx:].sum(-1)              # (H,)
            lr       = attn_ctx / (attn_ctx + attn_new + 1e-10)  # (H,) ∈ [0,1]

            gate = torch.ones_like(lr)   # default: let through

            if self.mode == 'zero_all':
                gate = torch.zeros_like(lr)

            elif self.mode == 'zero_high':
                # Blindness test: suppress context-attending (high-LR) heads
                gate[lr >= self.lr_cutoff] = 0.0

            elif self.mode == 'zero_low':
                # Hard knockout: suppress low-LR (hallucinating) heads
                gate[lr < self.lr_cutoff] = 0.0

            x_r   = x.view(B, 1, num_heads, head_dim)
            gate  = gate.to(x.device).view(1, 1, num_heads, 1)
            x_out = (x_r * gate).view(B, 1, H_)
            return (x_out,)
        return pre_hook

    def trigger(self):
        self.triggered = True
        self._attn_cache.clear()

    def reset(self):
        self.triggered = False
        self._attn_cache.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def compute_f1(pred, gt):
    p, g = pred.lower().split(), gt.lower().split()
    if not p or not g:
        return 0.0
    common   = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p)
    rec  = num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def compute_acc(pred, answers):
    if not answers:
        return 0.0
    return 1.0 if max(compute_f1(pred, a) for a in answers) * 100 >= 50.0 else 0.0


def generate_with_controller(raw_model, tokenizer, prompt,
                              controller, stop_seqs, existing_fallback):
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to("cuda")
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    n_prompt = inputs['input_ids'].shape[1]
    controller.context_length = n_prompt
    controller.trigger()

    try:
        with torch.no_grad():
            out = raw_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                output_attentions=True,
                output_scores=False,
                output_hidden_states=False,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_tok = out.sequences[0][n_prompt:]
        ans = tokenizer.decode(gen_tok, skip_special_tokens=True).strip()
        for stop in stop_seqs:
            if ans.endswith(stop):
                ans = ans[:-len(stop)].strip()
                break
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        ans = existing_fallback
    finally:
        controller.reset()

    return ans


# ------------------------------------------------------------------ #
# Parse args                                                           #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(description="Causal Validation of Lookback Gate")
    parser.add_argument("--dataset", required=True, choices=QA_DATASETS)
    parser.add_argument("--token_type", choices=["TBG", "SLT"], default="TBG")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Samples to test per condition (default: 100)")
    parser.add_argument("--lr_cutoff", type=float, default=0.5,
                        help="LR threshold for zero_high/zero_low modes (default: 0.5)")
    parser.add_argument("--layer_range", type=str, default=None,
                        help="Comma-separated start,end, e.g. '21,32'")
    return parser.parse_args()


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    ds = args.dataset

    out_dir      = os.path.join(OUTPUT_BASE, ds)
    gated_file   = os.path.join(out_dir, "gated_results.pkl")
    probe_file   = os.path.join(out_dir, f"sep_probe_{args.token_type}.pkl")
    val_out_file = os.path.join(out_dir, "causal_validation.pkl")

    for path, desc in [(gated_file, "gated_results.pkl"),
                       (probe_file, f"sep_probe_{args.token_type}.pkl")]:
        if not os.path.exists(path):
            logging.error(f"{desc} not found at {path}.")
            return

    with open(gated_file, "rb") as f:
        gated_data = pickle.load(f)

    with open(probe_file, "rb") as f:
        probe_bundle = pickle.load(f)     # contains threshold etc.

    layer_range = None
    if args.layer_range:
        s, e = args.layer_range.split(",")
        layer_range = range(int(s), int(e))

    # ================================================================ #
    # Sample selection                                                   #
    # Test 1 (knockout): samples that WERE triggered by SEP              #
    # Test 2 (blindness): samples that were NOT triggered                #
    # ================================================================ #

    triggered_items   = [r for r in gated_data if r.get('gate_triggered', False)]
    passthrough_items = [r for r in gated_data if not r.get('gate_triggered', False)
                         and r.get('prompt_used', '') != '']

    n = min(args.num_samples, len(triggered_items), len(passthrough_items))
    if n == 0:
        logging.error("Not enough samples for validation. "
                      "Run inference_with_gate.py first.")
        return

    knock_items  = triggered_items[:n]
    blind_items  = passthrough_items[:n]

    logging.info(f"Test 1 (knockout):  {len(knock_items)} triggered samples")
    logging.info(f"Test 2 (blindness): {len(blind_items)} passthrough samples")

    # ---- Load LLM ----
    logging.info(f"Loading model: {MODEL_NAME} ...")
    hf_model   = HuggingfaceModel(
        model_name=MODEL_NAME,
        stop_sequences='default',
        max_new_tokens=MAX_NEW_TOKENS,
    )
    raw_model = hf_model.model
    tokenizer = hf_model.tokenizer
    stop_seqs = hf_model.stop_sequences

    # ================================================================ #
    # TEST 1: KNOCKOUT (zero_all)                                        #
    # Expected: accuracy should drop relative to gated answer,           #
    # confirming the sigmoid gate is better than hard zeroing AND that   #
    # the gated generation is better than the original.                  #
    # ================================================================ #

    logging.info("\n=== TEST 1: KNOCKOUT (zero all upper-layer heads) ===")
    knockout_ctrl = ForcedGateController(
        raw_model, context_length=0,
        layer_range=layer_range, mode='zero_all'
    )

    knock_results = []
    for item in tqdm(knock_items, desc="Knockout"):
        prompt  = item['prompt_used']
        answers = item.get('answers', [])
        orig    = item['most_likely_answer']
        gated   = item['gated_answer']

        ko_ans = generate_with_controller(
            raw_model, tokenizer, prompt,
            knockout_ctrl, stop_seqs, orig
        )
        knock_results.append({
            'question':         item.get('question', ''),
            'answers':          answers,
            'original_answer':  orig,
            'gated_answer':     gated,
            'knockout_answer':  ko_ans,
            'acc_original':     compute_acc(orig,   answers),
            'acc_gated':        compute_acc(gated,  answers),
            'acc_knockout':     compute_acc(ko_ans, answers),
        })
        gc.collect()
        torch.cuda.empty_cache()

    knockout_ctrl.remove()

    # ================================================================ #
    # TEST 2: BLINDNESS (zero high-LR / context-attending heads)         #
    # Expected: accuracy degrades for passthrough samples (which were    #
    # confident), proving high-LR heads are causally important.          #
    # ================================================================ #

    logging.info("\n=== TEST 2: BLINDNESS (zero high-LR grounding heads) ===")
    blindness_ctrl = ForcedGateController(
        raw_model, context_length=0,
        layer_range=layer_range, mode='zero_high', lr_cutoff=args.lr_cutoff
    )

    blind_results = []
    for item in tqdm(blind_items, desc="Blindness"):
        prompt  = item['prompt_used']
        answers = item.get('answers', [])
        orig    = item['most_likely_answer']

        blind_ans = generate_with_controller(
            raw_model, tokenizer, prompt,
            blindness_ctrl, stop_seqs, orig
        )
        blind_results.append({
            'question':          item.get('question', ''),
            'answers':           answers,
            'original_answer':   orig,
            'blindness_answer':  blind_ans,
            'acc_original':      compute_acc(orig,      answers),
            'acc_blindness':     compute_acc(blind_ans, answers),
        })
        gc.collect()
        torch.cuda.empty_cache()

    blindness_ctrl.remove()

    # ================================================================ #
    # Results                                                            #
    # ================================================================ #

    print(f"\n{'='*65}")
    print(f"CAUSAL VALIDATION RESULTS — {ds}")
    print(f"{'='*65}")
    print(f"Model: {MODEL_NAME}")
    print(f"LR cutoff (blindness test): {args.lr_cutoff}")
    print()

    # Test 1
    orig_acc_ko  = np.mean([r['acc_original'] for r in knock_results])
    gated_acc_ko = np.mean([r['acc_gated']    for r in knock_results])
    ko_acc       = np.mean([r['acc_knockout'] for r in knock_results])

    print(f"--- TEST 1: Knockout (N={len(knock_results)} triggered samples) ---")
    print(f"  Original answer accuracy (pre-gate):   {orig_acc_ko:.4f}")
    print(f"  Gated answer accuracy (sigmoid gate):  {gated_acc_ko:.4f}  "
          f"({'↑ IMPROVED' if gated_acc_ko > orig_acc_ko else '↓ degraded'})")
    print(f"  Knockout accuracy (hard zero_all):     {ko_acc:.4f}  "
          f"({'↓ worse than gate' if ko_acc < gated_acc_ko else '→ similar'})")
    print(f"  → {'CAUSAL: sigmoid gate > hard zero confirms soft suppression is optimal'}"
          if gated_acc_ko > ko_acc else
          f"  → NOTE: hard zero matched or beat sigmoid gate — consider lower alpha")

    print()

    # Test 2
    orig_acc_bl = np.mean([r['acc_original']  for r in blind_results])
    blind_acc   = np.mean([r['acc_blindness'] for r in blind_results])

    print(f"--- TEST 2: Blindness (N={len(blind_results)} passthrough samples) ---")
    print(f"  Original answer accuracy:            {orig_acc_bl:.4f}")
    print(f"  After blinding grounding heads:      {blind_acc:.4f}  "
          f"({'↓ DEGRADED — grounding heads are causal!' if blind_acc < orig_acc_bl else '→ no change'})")
    print(f"  → {'CAUSAL: zeroing high-LR heads hurts accuracy, confirming their role in grounding'}"
          if blind_acc < orig_acc_bl else
          f"  → NOTE: blindness test inconclusive — try a lower lr_cutoff")

    print()
    print("=" * 65)

    # ---- Save ----
    val_output = {
        'knockout_results':  knock_results,
        'blindness_results': blind_results,
        'config': vars(args),
    }
    with open(val_out_file, "wb") as f:
        pickle.dump(val_output, f)
    logging.info(f"Saved validation results → {val_out_file}")


if __name__ == "__main__":
    main()
