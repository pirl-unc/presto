"""Training-only checks of the proposed process's mathematical shape limits."""

import gzip
import json
import math
from collections import Counter, defaultdict

import numpy as np

from verify import FAMILY, read_groups


def main():
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    groups = [g for g in read_groups(cfg) if g["split"] == "train"]
    envelope = []
    concavity = []
    for g in groups:
        percurve = defaultdict(list)
        for index, source in enumerate(g["rows"]):
            curve, t, ref, signal, reference = source
            b = int(curve["start_0based"] > 0) + int(
                curve["end_0based"] < len(g["sequence"])
            )
            upper = b * math.log10(t / ref)
            response = math.log10(signal / reference)
            envelope.append(
                dict(
                    curve_id=curve["curve_id"],
                    condition_id=g["condition"],
                    construct_id=g["parent"],
                    peptide=curve["peptide_IL_sequence"],
                    time=t,
                    reference_time=ref,
                    required_boundaries=b,
                    target=response,
                    upper_bound=upper,
                    excess=response - upper,
                    weight=float(g["w"][index]),
                )
            )
            percurve[curve["curve_id"]].append(source)
        for curveid, sources in percurve.items():
            curve = sources[0][0]
            points = {sources[0][2]: math.log10(sources[0][4])}
            points.update({s[1]: math.log10(s[3]) for s in sources})
            times = sorted(points)
            for a, b, c in zip(times, times[1:], times[2:]):
                chord = points[a] + (b - a) / (c - a) * (points[c] - points[a])
                concavity.append(
                    dict(
                        curve_id=curveid,
                        condition_id=g["condition"],
                        construct_id=g["parent"],
                        peptide=curve["peptide_IL_sequence"],
                        times=[a, b, c],
                        chord_minus_middle=chord - points[b],
                    )
                )

    def statistics(rows, value, timekey):
        pairs = defaultdict(list)
        for r in rows:
            pairs[r["condition_id"], r["peptide"], str(r[timekey])].append(r[value])
        return {
            "observations": len(rows),
            "by_threshold": {
                str(threshold): {
                    "fraction": sum(r[value] > threshold for r in rows) / len(rows),
                    "paired_events": len(pairs),
                    "both_labels_violate": sum(
                        len(v) >= 2 and all(x > threshold for x in v)
                        for v in pairs.values()
                    ),
                    "by_parent": {
                        parent: sum(
                            r[value] > threshold
                            for r in rows
                            if r["construct_id"] == parent
                        )
                        / sum(r["construct_id"] == parent for r in rows)
                        for parent in sorted({r["construct_id"] for r in rows})
                    },
                }
                for threshold in (0.0, 0.3, 0.5)
            },
        }

    lower = sum(r["weight"] * max(r["excess"], 0) ** 2 for r in envelope)
    result = dict(
        training_only=True,
        envelope=statistics(envelope, "excess", "time"),
        concavity=statistics(concavity, "chord_minus_middle", "times"),
        minimum_training_mse_from_upper_envelope=lower,
        minimum_rmse_from_upper_envelope=math.sqrt(lower),
        not_a_biological_cause_attribution=True,
        no_rows_excluded=True,
    )
    with gzip.open(FAMILY / "results/training-shape-observations.jsonl.gz", "wt") as f:
        for kind, rows in [("envelope", envelope), ("concavity", concavity)]:
            for r in rows:
                f.write(json.dumps(dict(kind=kind, **r), sort_keys=True) + "\n")
    (FAMILY / "results/training-shape-constraints.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
