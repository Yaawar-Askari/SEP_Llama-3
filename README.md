# Semantic Entropy Probes (SEP) for Llama-3 Hallucination Detection

[![arXiv](https://img.shields.io/badge/arXiv-2406.15927-b31b1b.svg)](https://arxiv.org/abs/2406.15927)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/get-started/locally/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository implements **Semantic Entropy Probes (SEPs)** for **Meta-Llama-3-8B-Instruct**, providing a complete pipeline to detect hallucinations in abstractive summarization using the [XSum](https://huggingface.co/datasets/xsum) dataset.

Adapted from the original [OATML/semantic-entropy-probes](https://github.com/OATML/semantic-entropy-probes) repository, modernized to support Llama-3, FP16 precision, and Hugging Face gated model access.

## Objective

Large Language Models (LLMs) often hallucinate -- producing fluent but factually incorrect text. This project distinguishes between **Confident** (accurate) and **Confused** (hallucinated) generations by measuring **Semantic Entropy** (consistency across multiple stochastic samples) and training a linear probe on the model's hidden states to predict this uncertainty.

SEPs approximate semantic entropy from a single forward pass through the model's hidden states, eliminating the need for multi-sampling at inference time and reducing computational overhead to near zero.

## Pipeline Architecture

The pipeline consists of four sequential stages:

```
XSum Documents
      |
      v
[1. Generation]         run_xsum_generation.py
      |                   - 5 stochastic summaries per document (T=0.7)
      |                   - Output: sep_xsum_generations.pkl
      v
[2. Labeling]           compute_rouge_labels.py
      |                   - Pairwise Rouge-L overlap scoring
      |                   - Label 0 (Confident): Rouge-L > 0.7
      |                   - Label 1 (Hallucinated): Rouge-L < 0.3
      |                   - Output: sep_filtered_labels.json
      v
[3. Feature Extraction] extract_features.py
      |                   - Prompt-only forward pass
      |                   - Last-token hidden state from final layer
      |                   - Output: sep_dataset.pt
      v
[4. Probe Training]     train_probe.py
                          - Logistic Regression on hidden states
                          - 80/20 train/test split with StandardScaler
                          - Reports Accuracy and AUROC
```

### Stage Details

**1. Generation (`run_xsum_generation.py`)** -- Samples 1,000 documents from XSum and generates 5 stochastic summaries per document using `Meta-Llama-3-8B-Instruct` at Temperature 0.7. Uses the OATML `HuggingfaceModel` wrapper for inference. Saves results incrementally every 100 samples.

**2. Labeling (`compute_rouge_labels.py`)** -- Measures semantic consistency via pairwise Rouge-L F1 scores across the 5 generations. Samples with high agreement (> 0.7) are labeled Confident (0); those with low agreement (< 0.3) are labeled Hallucinated (1). Ambiguous samples are discarded.

**3. Feature Extraction (`extract_features.py`)** -- Performs a prompt-only forward pass on each filtered sample and extracts the hidden state of the last token from the final transformer layer (dimension 4096 for Llama-3-8B). Saves features and labels as a PyTorch tensor file.

**4. Probe Training (`train_probe.py`)** -- Trains an L2-regularized Logistic Regression classifier (via scikit-learn) on the extracted hidden states. Features are normalized with `StandardScaler`. Outputs a classification report with per-class precision/recall and AUROC score.

## Installation & Setup

### Prerequisites

- Python 3.10+
- PyTorch with CUDA support
- GPU with at least 24 GB VRAM (for FP16 inference of the 8B model)
- Access to [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) on Hugging Face (gated model)

### Install Dependencies

```bash
pip install torch transformers datasets rouge-score scikit-learn accelerate matplotlib
```

Or use the provided conda environment:

```bash
conda env update -f sep_enviroment.yaml
conda activate se_probes
```

### Authenticate with Hugging Face

Llama-3 is a gated model. You must accept the license agreement and authenticate:

```bash
huggingface-cli login
```

## Usage

### Run the Full Pipeline

The generation step is the most time-consuming (~2 hours for 1,000 samples on an A100). Run everything sequentially:

```bash
nohup sh -c "python run_xsum_generation.py && python compute_rouge_labels.py && python extract_features.py && python train_probe.py" > pipeline.log 2>&1 &
```

Monitor progress:

```bash
tail -f pipeline.log
```

### Step-by-Step Execution

```bash
# 1. Generate stochastic summaries
python run_xsum_generation.py
# Output: sep_xsum_generations.pkl

# 2. Compute uncertainty labels via Rouge-L
python compute_rouge_labels.py
# Output: sep_filtered_labels.json

# 3. Extract hidden state features
python extract_features.py
# Output: sep_dataset.pt

# 4. Train and evaluate the probe
python train_probe.py
# Output: Classification report and AUROC printed to stdout
```

## Project Structure

```
semantic-entropy-probes/
├── run_xsum_generation.py          # Stage 1: Stochastic generation on XSum
├── compute_rouge_labels.py         # Stage 2: Rouge-L labeling
├── extract_features.py             # Stage 3: Hidden state extraction
├── train_probe.py                  # Stage 4: Logistic regression probe
├── sep_utils.py                    # Shared utilities (prompt formatting, logging)
├── sep_enviroment.yaml             # Conda environment specification
├── semantic_uncertainty/           # Core library (forked from OATML)
│   ├── generate_answers.py         #   Multi-dataset answer generation
│   ├── compute_uncertainty_measures.py  #   Semantic entropy computation
│   ├── analyze_results.py          #   Aggregate metrics
│   └── uncertainty/                #   Model wrappers, data utils, measures
│       ├── models/                 #     HuggingFace model interface
│       ├── data/                   #     Dataset loaders (TriviaQA, SQuAD, etc.)
│       └── uncertainty_measures/   #     SE, p_ik, p_true implementations
├── semantic_entropy_probes/        # Original SEP notebook and pre-trained models
│   └── train-latent-probe.ipynb    #   Interactive probe training notebook
├── slurm/
│   └── run.sh                      # SLURM batch job script
└── LICENSE
```

## Key Modifications for Llama-3

- **Precision**: Enforced `torch.float16` to fit the 8B parameter model on consumer GPUs (24 GB VRAM).
- **Prompt Formatting**: `sep_utils.py` handles Llama-3's specific `<|begin_of_text|>`, `<|start_header_id|>`, and `<|eot_id|>` special token structure, ensuring identical formatting across generation and feature extraction.
- **Task**: Abstractive summarization on XSum, extending the original QA-focused pipeline.
- **Labeling**: Rouge-L-based proxy for semantic entropy, replacing the entailment-based clustering used in the original paper.
- **Error Handling**: Robust handling of empty generations and token limit overflows during generation.

## Credits

Based on the paper:

> **Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs**
> Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, Yarin Gal
> arXiv:2406.15927 (2024)

Original repository: [OATML/semantic-entropy-probes](https://github.com/OATML/semantic-entropy-probes)

```bibtex
@misc{kossen2024semanticentropyprobesrobust,
      title={Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs},
      author={Jannik Kossen and Jiatong Han and Muhammed Razzak and Lisa Schut and Shreshth Malik and Yarin Gal},
      year={2024},
      eprint={2406.15927},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.15927},
}
```
