"""Declared unobserved ligand backgrounds with positive lineage and split embargoes.

These labels encode a sampled comparison population, never measured non-binding
or non-presentation. Protein mappings select a background population; they do not
assign the true source of a multiply mapping positive peptide.
"""

from dataclasses import replace
from functools import lru_cache
import hashlib
import random

from .modular import AA, Observation


class LigandBackgroundSampler:
    """Length-matched positions from mapped proteins, otherwise a declared proteome.

    A proteome supplies ``proteins`` and exact ``lookup(peptide)``. Protein IDs are
    sampled uniformly, then positions uniformly. Existing positive sequences and
    sequences assigned to another split are excluded. The caller fixes row order
    and persists the resulting membership; this class does not assign partitions.
    """

    def __init__(self, proteome, *, seed, reference, positive_peptides, peptide_splits):
        if type(seed) is not int or not isinstance(reference, str) or not reference.strip():
            raise ValueError("integer seed and frozen background-policy reference required")
        self.proteome = proteome
        self.seed = seed
        self.reference = reference
        self.positives = positive_peptides
        self.peptide_splits = peptide_splits
        self._eligible_proteins = lru_cache(maxsize=512)(self._eligible_proteins)
        self._parents = lru_cache(maxsize=65536)(self._parents)

    def _eligible_proteins(self, length):
        return tuple(sorted(k for k, v in self.proteome.proteins.items() if len(v) >= length))

    def _parents(self, peptide):
        return tuple(sorted({protein for protein, _ in self.proteome.lookup(peptide)}))

    def sample(self, positive: Observation, *, replicate=0):
        if (
            positive.endpoint != "elution_observation"
            or positive.value != 1
            or positive.split not in {"train", "validation", "test"}
            or not positive.group_id
            or type(replicate) is not int
            or replicate < 0
        ):
            raise ValueError("partitioned positive ligand and nonnegative replicate required")
        length = len(positive.peptide)
        parent_ids = self._parents(positive.peptide)
        population = "same_protein" if parent_ids else "reference_proteome"
        eligible = parent_ids or self._eligible_proteins(length)
        if not eligible:
            raise ValueError("no protein supports the requested background length")
        token = f"{self.reference}|{self.seed}|{positive.observation_id}|{replicate}"
        digest = hashlib.sha256(token.encode()).hexdigest()
        rng = random.Random(int(digest, 16))
        # Rejection proposals are uniform within a protein. Exhaustive fallback
        # samples the valid positions uniformly and detects true exhaustion.
        remaining = eligible
        while remaining:
            protein_id = remaining[rng.randrange(len(remaining))]
            sequence = self.proteome.proteins[protein_id]
            positions = len(sequence) - length + 1

            def permitted(position, sequence=sequence):
                peptide = sequence[position : position + length]
                return (
                    not set(peptide) - set(AA)
                    and peptide not in self.positives
                    and self.peptide_splits.get(peptide, positive.split) == positive.split
                )

            position = None
            for _ in range(32):
                proposed = rng.randrange(positions)
                if permitted(proposed):
                    position = proposed
                    break
            if position is None:
                valid = [i for i in range(positions) if permitted(i)]
                if not valid:
                    remaining = tuple(key for key in remaining if key != protein_id)
                    continue
                position = rng.choice(valid)
            peptide = sequence[position : position + length]
            result = replace(
                positive,
                observation_id="ligand-background:" + digest,
                source="hitlist_ligand_background",
                peptide=peptide,
                value=0.0,
                provenance={
                    "measurement_namespace": "generated_ligand_background_v1",
                    "original_measurement_id": digest,
                    "label_kind": "generated_unobserved_background",
                    "measured_outcome": False,
                    "policy_reference": self.reference,
                    "seed": self.seed,
                    "replicate": replicate,
                    "parent_observation_id": positive.observation_id,
                    "study_id": positive.provenance["study_id"],
                    "source_contributor_id": positive.provenance.get("source_contributor_id"),
                    "background_kind": population,
                    "protein_id": protein_id,
                    "protein_position": position,
                    "positive_candidate_proteins": list(parent_ids),
                    "length_matched": True,
                    "raw_record": positive.provenance.get("raw_record", {}),
                    "interpretation": (
                        "Unobserved sampled comparator; not a measured biological negative"
                    ),
                },
            )
            result.validate()
            self.peptide_splits[peptide] = positive.split
            return result
        raise ValueError("background population exhausted after positive and split exclusion")
