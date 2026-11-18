"""Experiment receipt around the canonical explicit APC-category projection."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict

from presto.data.modular import Observation
from presto.data.modular_context import project_apc_context


class Projection:
    def __init__(self, config):
        self.vocabulary = {
            name: tuple(config.get(name, ()))
            for name in (
                "cell_types",
                "cytokines",
                "processing_pathways",
                "interventions",
            )
        }
        self.rows = 0
        self.changed = 0
        self.claims = Counter()
        self.digest = hashlib.sha256()

    def record(self, original):
        context = Observation.from_dict(original).context
        projected = project_apc_context(context, **self.vocabulary)
        result = original
        self.rows += 1
        if projected is not context:
            self.changed += 1
            result = {**original, "context": asdict(projected)}
            omitted = projected.provenance["unrepresented_apc_context"][-1]
            for family, values in omitted.items():
                if isinstance(values, dict):
                    self.claims.update(f"{family}:{name}" for name in values)
                elif values is not None:
                    self.claims[f"{family}:{values}"] += 1
        encoded = json.dumps(result, separators=(",", ":")) + "\n"
        self.digest.update(encoded.encode())
        return result, encoded

    def receipt(self):
        return {
            "policy": "unsupported_categorical_apc_claims_to_unknown_preserve_provenance",
            "vocabulary": self.vocabulary,
            "rows": self.rows,
            "changed_rows": self.changed,
            "omitted_claim_counts": dict(sorted(self.claims.items())),
            "projected_jsonl_sha256": self.digest.hexdigest(),
        }
