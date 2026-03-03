"""Tests for training run artifact logging."""

from types import SimpleNamespace


def _assert_metric_artifacts(run_dir):
    metrics_csv = run_dir / "metrics.csv"
    metrics_jsonl = run_dir / "metrics.jsonl"
    assert metrics_csv.exists()
    assert metrics_jsonl.exists()
    assert metrics_csv.read_text(encoding="utf-8").strip().splitlines()[0] == "step,split,metric,value"
    assert len(metrics_jsonl.read_text(encoding="utf-8").strip().splitlines()) >= 1


def test_train_synthetic_writes_metrics_artifacts(tmp_path):
    from presto.scripts.train_synthetic import run

    run_dir = tmp_path / "synthetic_run"
    args = SimpleNamespace(
        epochs=1,
        batch_size=4,
        lr=1e-3,
        d_model=64,
        n_layers=1,
        n_heads=4,
        n_binding=12,
        n_elution=8,
        n_tcr=8,
        data_dir=str(tmp_path / "synthetic_data"),
        checkpoint=str(tmp_path / "synthetic.pt"),
        seed=7,
        run_dir=str(run_dir),
    )
    run(args)
    _assert_metric_artifacts(run_dir)
    metrics_csv = run_dir / "metrics.csv"
    contents = metrics_csv.read_text(encoding="utf-8")
    assert "uw_weight_binding" in contents



