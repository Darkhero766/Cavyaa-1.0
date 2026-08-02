"""Core CAVYAA tests.

Tests that require scientific dependencies are skipped automatically when the
execution environment does not provide those dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from config import DataConfig, LossConfig, ModelConfig  # noqa: E402
from dataset import CavyaaDataset, make_loader  # noqa: E402
from domain_adaptation import DomainClassifier  # noqa: E402
from evaluation import evaluate  # noqa: E402
from losses import cavyaa_loss  # noqa: E402
from model import CavyaaModel  # noqa: E402
from preprocessing import Preprocessor, split_by_patient  # noqa: E402
from synthetic_data import generate_synthetic_cohort  # noqa: E402
from trainer import Trainer  # noqa: E402
from utils import get_device, set_seed  # noqa: E402
from vae import BetaVAE  # noqa: E402


def small_arrays(rows: int = 8) -> dict[str, np.ndarray]:
    """Create deterministic arrays for dataset and model tests."""
    rng = np.random.default_rng(3)
    return {
        "fragment": rng.normal(size=(rows, 48)).astype("float32"),
        "protein": rng.normal(size=(rows, 36)).astype("float32"),
        "timeline": rng.normal(size=(rows, 2)).astype("float32"),
        "cancer": rng.integers(0, 5, size=rows).astype("int64"),
        "domain": rng.integers(0, 4, size=rows).astype("int64"),
        "label": rng.integers(0, 2, size=rows).astype("float32"),
        "patient_id": np.array([f"p{i}" for i in range(rows)]),
    }


def test_synthetic_generator_has_expected_columns_and_no_empty_patients() -> None:
    """Synthetic generator returns longitudinal rows with required fields."""
    cfg = DataConfig(n_patients=25, min_draws=3, max_draws=4, seed=10)
    frame = generate_synthetic_cohort(cfg)
    assert frame["patient_id"].nunique() == 25
    assert frame.groupby("patient_id").size().min() >= 3
    assert {"weeks_post_operation", "sequencer", "recurrence_label"}.issubset(frame.columns)


def test_dataset_shapes_and_loader() -> None:
    """Dataset and DataLoader preserve tensor shapes and dtypes."""
    dataset = CavyaaDataset(small_arrays())
    sample = dataset[0]
    assert sample["fragment"].shape == (48,)
    assert sample["protein"].shape == (36,)
    batch = next(iter(make_loader(dataset, batch_size=4, shuffle=False)))
    assert batch["timeline"].shape == (4, 2)


def test_forward_pass_and_no_nan() -> None:
    """Model forward pass returns finite logits and latent tensors."""
    model = CavyaaModel(ModelConfig())
    batch = next(iter(make_loader(CavyaaDataset(small_arrays()), batch_size=4, shuffle=False)))
    outputs = model(batch)
    assert outputs["recurrence_logit"].shape == (4,)
    for tensor in outputs.values():
        assert torch.isfinite(tensor).all()


def test_loss_is_finite_and_backward_computes_gradients() -> None:
    """Composite loss is finite and produces gradients."""
    model = CavyaaModel(ModelConfig())
    batch = next(iter(make_loader(CavyaaDataset(small_arrays()), batch_size=4, shuffle=False)))
    outputs = model(batch)
    loss, parts = cavyaa_loss(outputs, batch, beta=0.01, config=LossConfig())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in parts.values())
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_gradient_reversal_changes_gradient_sign() -> None:
    """GRL reverses latent gradients while keeping forward values usable."""
    classifier = DomainClassifier(latent_dim=3, hidden_dim=6, n_domains=2, dropout=0.0)
    x = torch.randn(5, 3, requires_grad=True)
    y = torch.tensor([0, 1, 0, 1, 0])
    classifier.set_lambda(1.0)
    loss = torch.nn.functional.cross_entropy(classifier(x), y)
    loss.backward()
    reversed_grad = x.grad.clone()
    x.grad.zero_()
    classifier.set_lambda(-1.0)
    loss = torch.nn.functional.cross_entropy(classifier(x), y)
    loss.backward()
    assert torch.sum(reversed_grad * x.grad) < 0


def test_vae_reparameterization_and_shapes() -> None:
    """VAE returns correctly shaped latent variables and reconstructions."""
    vae = BetaVAE(hidden_dim=16, latent_dim=5, fragment_dim=7, protein_dim=9)
    outputs = vae(torch.randn(4, 16))
    assert outputs["z"].shape == (4, 5)
    assert outputs["fragment_reconstruction"].shape == (4, 7)
    assert outputs["protein_reconstruction"].shape == (4, 9)


def test_training_step_checkpoint_loading_and_evaluation(tmp_path: Path) -> None:
    """A tiny training run saves and reloads a checkpoint and computes real metrics."""
    set_seed(4)
    cfg = DataConfig(n_patients=35, min_draws=2, max_draws=3, seed=4)
    frame = generate_synthetic_cohort(cfg)
    train_frame, val_frame, _ = split_by_patient(frame, 0.2, 0.2, 4)
    pre = Preprocessor.fit(train_frame)
    train_loader = make_loader(CavyaaDataset(pre.transform(train_frame)), batch_size=16, shuffle=True)
    val_loader = make_loader(CavyaaDataset(pre.transform(val_frame)), batch_size=16, shuffle=False)
    from config import ExperimentConfig, TrainConfig

    exp = ExperimentConfig(
        train=TrainConfig(epochs=1, checkpoint_dir=tmp_path, log_dir=tmp_path / "tb", history_csv=tmp_path / "history.csv")
    )
    model = CavyaaModel(exp.model)
    trainer = Trainer(model, exp, torch.device("cpu"))
    history, checkpoint = trainer.fit(train_loader, val_loader)
    assert checkpoint.exists()
    assert history
    model2 = CavyaaModel(exp.model)
    trainer2 = Trainer(model2, exp, torch.device("cpu"))
    restored = trainer2.load_checkpoint(checkpoint)
    assert restored
    metrics = evaluate(model2, val_loader, torch.device("cpu"))
    assert "roc_auc" in metrics and "ece" in metrics


def test_device_compatibility() -> None:
    """Device helper returns a torch device and model can move to it."""
    device = get_device()
    model = CavyaaModel(ModelConfig()).to(device)
    assert next(model.parameters()).device.type == device.type
