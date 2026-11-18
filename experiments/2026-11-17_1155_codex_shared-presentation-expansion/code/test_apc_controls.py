"""Controls must retain pairing, outcomes and peptide-specific source evidence."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "apc_controls", Path(__file__).with_name("apc_controls.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def examples():
    return [
        {
            "split": "validation",
            "value": label,
            "peptide": f"PEPTIDE-{i}-{label}",
            "mhc": [{"mhc_class": cls}],
            "provenance": {"pair_id": f"{cls}:{i}", "study_id": f"study-{i // 3}"},
            "context": {
                "source_n": f"FLANK-{i}-{label}",
                "acquisition": {"x": i},
                "cell_type": f"cell-{i}",
                "interventions": {"knockout:B2M": {"state": "present" if i % 2 else "absent"}},
            },
        }
        for cls in ("I", "II")
        for i in range(9)
        for label in (0, 1)
    ]


def test_controls_preserve_pairs_source_context_and_grouped_signatures():
    source = examples()
    lookup = {r["provenance"]["pair_id"]: r for r in source}
    changed, report = module.controlled(source, "within-study-shuffle")
    assert 0 < report["changed_pairs_before_vocab_projection"] < 18
    assert report["non_apc_invariants_verified"]
    for original, row in zip(source, changed, strict=True):
        assert row["context"]["source_n"] == original["context"]["source_n"]
        assert row["context"]["acquisition"] == original["context"]["acquisition"]
        donor = lookup[row["provenance"]["apc_input_control"]["donor_pair"]]
        assert donor["mhc"] == row["mhc"]
        assert donor["provenance"]["study_id"] == row["provenance"]["study_id"]
        assert module.signature(row["context"]) == module.signature(donor["context"])
    again, same = module.controlled(source, "within-study-shuffle")
    assert changed == again and report == same


def test_unknown_is_not_absent_and_original_is_unchanged():
    source = examples()
    unknown, _ = module.controlled(source, "unknown")
    original, report = module.controlled(source, "original")
    assert report["changed_pairs_before_vocab_projection"] == 0
    for before, unchanged, masked in zip(source, original, unknown, strict=True):
        assert module.signature(before["context"]) == module.signature(unchanged["context"])
        assert masked["context"]["cell_type"] is None
        assert masked["context"]["interventions"] == {}
        assert masked["context"]["source_n"] == before["context"]["source_n"]
