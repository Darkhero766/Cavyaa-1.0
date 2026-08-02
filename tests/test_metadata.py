"""Dependency-free repository metadata tests."""

from __future__ import annotations

from pathlib import Path


def test_scientific_transparency_documents_exist() -> None:
    """Audit and report files exist even when scientific dependencies are unavailable."""
    for name in ["CODE_AUDIT.md", "FINAL_REPORT.md", "CHANGELOG.md", "README.md"]:
        assert Path(name).exists()
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "synthetic data only" in readme.lower()
    assert "clinically validated" in readme
