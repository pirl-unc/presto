"""Synthetic software regression checks before touching original fit targets."""

import copy
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import fit as runner
from admit import derive
from common import ROOT, read, request, write
from model import fit, predict, sequence_features, weights


def fixture():
    sequences = [
        "AAAK",
        "AAAKK",
        "AALK",
        "AALLK",
        "AGGK",
        "AGGGK",
        "AEEK",
        "AEEEK",
        "ARRK",
        "ARRRK",
        "ATTK",
        "ATTTK",
    ]
    return [
        {
            "measurement_id": f"s{i}_d{j}",
            "sequence": seq,
            "gene": f"G{i // 2}",
            "x": float(j),
            "y": 2 + 0.7 * j + 0.2 * len(seq),
            "dose_fmol": 10.0**j,
        }
        for i, seq in enumerate(sequences)
        for j in (-1, 0, 1)
    ]


def main():
    req = request()
    rows = fixture()
    simple = [{**r, "y": 2 + 0.7 * r["x"]} for r in rows]
    model = fit(simple, "slope")
    assert np.allclose(model["coefficients"], [2, 0.7], rtol=0, atol=1e-12)
    unit = fit([{**r, "y": 3 + r["x"]} for r in rows], "unit")
    assert np.allclose(unit["coefficients"], [3], rtol=0, atol=1e-12)
    duplicate = [
        *rows,
        *[
            {**r, "measurement_id": r["measurement_id"] + "duplicate"}
            for r in rows
            if r["sequence"] == "AAAK"
        ],
    ]
    assert np.isclose(weights(duplicate).sum(), 1)
    for seq in {r["sequence"] for r in duplicate}:
        assert np.isclose(
            sum(
                w
                for r, w in zip(duplicate, weights(duplicate), strict=True)
                if r["sequence"] == seq
            ),
            1 / 12,
        )
    assert np.allclose(fit(rows, "slope")["coefficients"], fit(duplicate, "slope")["coefficients"])
    seq_model = fit(rows, "sequence", 0.1)
    assert all(
        len(v) == len(seq_model["active_features"])
        for v in (seq_model["feature_means"], seq_model["feature_scales"])
    )
    assert np.isfinite(predict(seq_model, rows)).all()
    assert seq_model["regularized_rank"] == seq_model["n_parameters"]
    assert np.allclose(sequence_features("AAK")[:2], [2 / 3, 0])
    valid = {
        "decision": "accepted",
        "target_value": 10.0,
        "amount.nominal_amount_fmol": 1.0,
        "target_unit": "source_arbitrary_summed_intensity",
        "amount.measurement_unit": "fmol",
        "standard.peptide_sequence": "AAK",
        "standard.gene_symbol_raw": "G",
    }
    assert derive(valid)["y"] == 1.0
    for field in ("target_value", "amount.nominal_amount_fmol"):
        for value in (None, 0, -1, float("nan"), float("inf")):
            try:
                derive({**valid, field: value})
            except ValueError:
                pass
            else:
                raise AssertionError("Accepted invalid quantity")
    with tempfile.TemporaryDirectory(prefix="nontryptic-response-fixture-") as tmp:
        root = Path(tmp)
        (root / "reproduce").mkdir()
        (root / "results").mkdir()
        write(root / "reproduce/launch.json", {"alphas": [0.1, 1.0]})
        write(root / "rows.json", rows)
        folds = [
            {
                "held_gene": g,
                "held_sequences": sorted({r["sequence"] for r in rows if r["gene"] == g}),
            }
            for g in sorted({r["gene"] for r in rows})
        ]
        write(root / "folds.json", folds)
        r = {
            "admitted": str(root / "rows.json"),
            "folds": str(root / "folds.json"),
            "output_subdir": "first",
        }
        with patch.object(runner, "ROOT", root), patch.object(runner, "request", lambda: r):
            runner.main()
        altered = copy.deepcopy(rows)
        for row in altered:
            if row["gene"] == "G0":
                row["y"] += 100 * row["x"] + 1000
        write(root / "altered.json", altered)
        r = {**r, "admitted": str(root / "altered.json"), "output_subdir": "second"}
        with patch.object(runner, "ROOT", root), patch.object(runner, "request", lambda: r):
            runner.main()
        a, b = [read(root / "results" / name / "models.json") for name in ("first", "second")]
        assert [m for m in a if m["outer_gene"] == "G0"] == [
            m for m in b if m["outer_gene"] == "G0"
        ]
        for m in a:
            assert m["outer_gene"] not in m["training_genes"]
            if m["level"] == "inner":
                assert m["inner_gene"] not in m["training_genes"]
    result = {
        "status": "passed",
        "fixture": "synthetic_software_only_no_biological_source_access",
        "checks": [
            "known_slope",
            "unit_slope",
            "equal_peptide_dose_weight",
            "duplicate_slot_invariance",
            "finite_ridge_and_active_columns",
            "invalid_quantity_rejection",
            "full_runner_outer_target_mutation_invariance",
            "inner_and_outer_gene_exclusion",
        ],
    }
    write(ROOT / "results" / req["result_file"], result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
