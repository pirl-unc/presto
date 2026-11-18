"""Complete-pair presentation metrics shared by evaluation workflows."""

from collections import defaultdict
import math


def paired_metrics(records):
    groups = defaultdict(dict)
    for record in records:
        if record["endpoint"] != "elution_observation":
            continue
        observation = record["observation"]
        pair = observation.get("provenance", {}).get("pair_id")
        if not pair:
            continue
        classes = set(record["mhc_classes"])
        if len(classes) != 1:
            raise ValueError("pair metrics require one declared MHC class per observation")
        key = (next(iter(classes)), pair)
        label = record["target"]
        if label not in (0, 1) or label in groups[key]:
            raise ValueError("pair requires exactly one positive and one background")
        score = record["prediction"]
        if not math.isfinite(score):
            raise ValueError("pair metrics require finite predictions")
        groups[key][label] = score
    scores = defaultdict(list)
    for (cls, _), values in groups.items():
        if set(values) != {0, 1}:
            raise ValueError("incomplete presentation pair")
        delta = values[1] - values[0]
        scores[cls].append(float(delta > 0) + 0.5 * (delta == 0))
    return {
        cls: {"pairs": len(values), "win_rate": sum(values) / len(values)}
        for cls, values in sorted(scores.items())
    }
