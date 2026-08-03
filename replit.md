# CAVYAA 1.0

**Tumor-Informed & Timeline-Aware Minimal Residual Disease Tracking**

Research prototype for computational oncology algorithm development using **synthetic data only**. Not clinically validated. See README.md for full scientific limitations.

## Stack

- Python 3.12
- PyTorch (CPU; CUDA used automatically when available)
- scikit-learn, numpy, pandas, matplotlib, tensorboard, umap-learn

## How to run

### Train the model
```bash
python train.py
```
Writes checkpoints, metrics JSON, TensorBoard logs, and figures under `artifacts/`.

### Run inference on a saved checkpoint
```bash
python infer.py --checkpoint artifacts/checkpoints/best_model.pt
```

### Run tests
```bash
python -m pytest tests/ -v
```

## Project layout

| File | Purpose |
|---|---|
| `config.py` | Dataclass configuration (data, model, loss, training, viz) |
| `synthetic_data.py` | Synthetic longitudinal cohort simulator |
| `dataset.py` | PyTorch Dataset and DataLoader helpers |
| `preprocessing.py` | Patient-wise splitting, imputation, scaling |
| `layers.py` | Residual blocks and gradient reversal |
| `encoders.py` | Fragment, protein, timeline, cancer encoders |
| `fusion.py` | Attention-gated cross-modal fusion |
| `vae.py` | β-VAE latent bottleneck |
| `domain_adaptation.py` | Sequencer domain classifier + GRL schedule |
| `model.py` | End-to-end CavyaaModel |
| `losses.py` | Multi-objective loss + KL annealing |
| `metrics.py` | Classification and calibration metrics |
| `trainer.py` | Training loop, checkpointing, TensorBoard, early stopping |
| `evaluation.py` | Prediction and metric evaluation helpers |
| `visualization.py` | Diagnostic plot generation |
| `utils.py` | Reproducibility, logging, JSON/CSV, device helpers |
| `train.py` | Full training entry point |
| `infer.py` | Checkpoint inference example |

## Notes

- No secrets or external services required — everything runs on synthetic data.
- Artifacts (checkpoints, figures, metrics) are written to `artifacts/` which is gitignored.
- The test suite runs in ~15 seconds on CPU.

## User preferences

- Wants to build a patient-facing "healing cancer journey" app on top of this ML backend (follow-up task).
