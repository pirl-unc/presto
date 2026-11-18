"""Human-readable APC support from the exact protected native TRAIN audit."""

import json
from pathlib import Path

import pandas as pd
from report import table

E = Path(__file__).resolve().parents[1]


def main():
    audit = json.loads((E / "results/context-conditional-support.json").read_text())
    train = audit["populations"]["expanded-train.jsonl"]
    source = pd.DataFrame(train["factor_support"])
    # Line-level groups omit rows with no reported line; cell-type groups retain unknowns.
    rows = []
    typed = source[source.scope == "study_cell_type"]
    for (cls, factor), group in typed.groupby(["class", "factor"]):
        lines = source[
            (source.scope == "study_reported_cell_line")
            & (source["class"] == cls) & (source.factor == factor)
        ]
        for col in ("present_positive_rows", "absent_positive_rows"):
            assert group[col].sum() >= lines[col].sum(), (cls, factor, col)
        rows.append({
            "class": cls,
            "factor": factor,
            "explicit_present_positives": int(group.present_positive_rows.sum()),
            "explicit_absent_positives": int(group.absent_positive_rows.sum()),
            "study_line_groups_with_both_states": int(lines.both_explicit_states.sum()),
            "present_positives_without_reported_line": int(
                group.present_positive_rows.sum() - lines.present_positive_rows.sum()
            ),
            "pmids": ", ".join(sorted(group.study.astype(str).unique())),
        })
    summary = pd.DataFrame(rows)
    out = E / "results/analysis"
    summary.to_csv(out / "apc-training-support.csv", index=False)
    text = [
        "# APC annotations in actual native TRAIN", "",
        f"Protected native TRAIN contains {train['positive_rows']:,} MS-positive rows.", "",
        "Counts below are after the declared training filters and leakage exclusions. "
        "They are not the full Hitlist export counts. Each reported present/absent annotation "
        "is retained literally; an unknown exposure is not counted as an absent control. "
        "Concurrent factors can occur in the same row, so factor counts must not be summed "
        "into a dataset size. Backgrounds inherit their paired positive's APC state.", "",
        table(summary), "",
        "A study/reported-line group with both annotations is a candidate contrast, not proof "
        "of an isolated intervention. Some reported lines are broad cell labels. Rows without "
        "reported lines are excluded from line-level contrasts. Co-exposures, candidate MHC "
        "sets and source arm attribution still matter. HLA-DM and CIITA "
        "expression annotations without explicit controls support conditional prediction, "
        "not calibrated overexpression effects. Drug-absent controls are not fabricated. "
        "The model has no measured expression or absolute presentation-yield training targets.", "",
        "Exact source hashes, conditions, concurrent signatures and cell counts: "
        "[context-conditional-support.json](../context-conditional-support.json). "
        "Upstream curation: [Hitlist #905](https://github.com/pirl-unc/hitlist/issues/905).", "",
    ]
    (out / "apc-training-support.md").write_text("\n".join(text))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
