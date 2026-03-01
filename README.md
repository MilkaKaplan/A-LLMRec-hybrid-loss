# Hybrid MSE + InfoNCE Matching Loss for A-LLMRec

**Course project** — Ben-Gurion University of the Negev, 2026  
**Student:** Milka Kaplan

## Based on

> Sein Kim, Hongseok Kang, Seungyoon Choi, Donghyun Kim, Minchul Yang, Chanyoung Park.  
> **"Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System"**  
> KDD 2024.  
> Original code: https://github.com/ghdtjr/A-LLMRec

---

## What We Did

We identified a limitation in A-LLMRec's Stage 1 alignment objective: the MSE-only matching loss produces over-smoothed representations. We proposed a **hybrid matching loss** combining MSE with a CLIP-style InfoNCE contrastive objective:

```
L_hybrid = α_mse · L_MSE + α_InfoNCE · L_InfoNCE
```

We implemented and evaluated three configurations on Amazon All Beauty and Movies and TV datasets using **OPT-350m** on a single **NVIDIA RTX 3090**.

---

## Key Files

| File | Description |
|------|-------------|
| `models/a_llmrec_model_HYBRID.py` | Modified model with hybrid loss |
| `models/a_llmrec_model_BASELINE.py` | Original MSE-only baseline |
| `hybrid_loss_stage1.py` | HybridMatchingLoss implementation |
| `run_hybrid_beauty_s2.sh` | Job script: Hybrid Stage 2 (Beauty) |
| `run_hybrid_tau05_beauty.sh` | Job script: Hybrid τ=0.5 (Beauty) |

---

## Experimental Results

### Reproduction (Baseline)

| Dataset | Model | Hit@1 | Original Hit@1 |
|---------|-------|-------|----------------|
| Movies and TV | SASRec (CF backbone) | NDCG@10 = 0.6743 | — |
| Movies and TV | A-LLMRec Baseline (OPT-350m) | 0.2684 | 0.6237 (OPT-6.7B) |
| All Beauty | A-LLMRec Baseline (OPT-350m) | 0.01684 | 0.5809 (OPT-6.7B) |

Performance gap vs. original is due to OPT-350m vs. OPT-6.7B (~19× smaller model).

### Hybrid Loss Experiments (All Beauty)

| Method | Hit@1 | Δ vs Baseline |
|--------|-------|---------------|
| Baseline (MSE only, B=8) | 0.01684 | — |
| Hybrid v1 (τ=0.07, B=8) | 0.01676 | −0.05% |
| Hybrid v2 (τ=0.5, B=8) | 0.01594 | −5.35% |
| Hybrid v1 (τ=0.07, 10 Stage 2 epochs) | 0.01457 | −13.5% |
| Hybrid v3 (τ=0.07, eff. B=32, grad. accum) | 0.00650 | −61.4% |

---

## Key Finding

All hybrid configurations underperform the MSE-only baseline. The failure is **not** due to insufficient negatives — gradient accumulation (eff. B=32) performs worst of all.

The cause is **gradient conflict**: MSE pulls embeddings together while InfoNCE pushes them apart. At B=8, InfoNCE signal is too weak (log(8) ≈ 2.08 nats) and stuck at the entropy ceiling. When amplified via gradient accumulation, it actively destabilizes training.

---

## Setup

```bash
# Clone original repo
git clone https://github.com/ghdtjr/A-LLMRec.git

# Replace model file with hybrid version
cp models/a_llmrec_model_HYBRID.py A-LLMRec/models/a_llmrec_model.py

# Run Stage 1 + Stage 2 + Inference
sbatch run_hybrid_beauty_s2.sh
```

**Hardware:** NVIDIA RTX 3090 (24GB)  
**Environment:** Python 3.8, PyTorch, conda env `allm_rec`  
**Datasets:** Amazon All Beauty, Amazon Movies and TV

