# Changelog

## Unreleased

- Added `CODE_AUDIT.md` with severity-ranked findings from a full module review.
- Removed fabricated dependency-free training metrics, placeholder plots, and fake checkpoints from `train.py`.
- Added explicit dependency checks to `train.py` and `infer.py` so missing scientific packages fail clearly.
- Improved synthetic data generation with documented low-rank feature correlations, AR(1)-style longitudinal noise, heteroscedastic measurement noise, patient-level latent effects, and time/inflammation-associated missingness.
- Added explicit model initialization and additional normalization in the recurrence head and β-VAE bottleneck.
- Replaced deprecated CUDA AMP usage with `torch.amp.autocast` and `torch.amp.GradScaler`.
- Added finite-loss, finite-output, and gradient-norm validation during training.
- Added AdamW epsilon/beta settings, linear warmup plus cosine decay, best/last checkpoint saving, full resume state, CSV history logging, and TensorBoard scalar logging.
- Added configurable DataLoader persistent workers and prefetch behavior.
- Added optional `torch.compile()` support through configuration.
- Updated visualizations to use UMAP when installed and t-SNE fallback otherwise, without mislabeled claims.
- Added pytest tests covering dataset behavior, model forward pass, loss/backward, GRL, VAE, training/checkpoint/evaluation, synthetic generation, NaN checks, and device compatibility.
- Expanded README scientific-transparency documentation and dependency metadata.
- Added `FINAL_REPORT.md` summarizing architecture review, fixes, testing, limitations, and future work.
