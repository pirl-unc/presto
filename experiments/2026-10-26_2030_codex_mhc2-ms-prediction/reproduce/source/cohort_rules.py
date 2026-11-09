"""Deterministic molecular components and explicit generated backgrounds."""

from collections import Counter, defaultdict
import hashlib
import random


def kmers(peptide):
    return {peptide[i : i + 9] for i in range(len(peptide) - 8)}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def components(peptides):
    """Transitive shared-nine-residue components, independent of input ordering."""
    peptides = sorted(set(peptides))
    parent = {peptide: peptide for peptide in peptides}

    def find(peptide):
        while parent[peptide] != peptide:
            parent[peptide] = parent[parent[peptide]]
            peptide = parent[peptide]
        return peptide

    owner = {}
    for peptide in peptides:
        for segment in sorted(kmers(peptide)):
            if segment in owner:
                left, right = find(peptide), find(owner[segment])
                parent[max(left, right)] = min(left, right)
            else:
                owner[segment] = peptide
    return {peptide: find(peptide) for peptide in peptides}


def dp_split(component):
    fraction = int(digest("presto-ms-dp-v1|" + component)[:16], 16) / 2**64
    return "train" if fraction < 0.70 else "validation" if fraction < 0.85 else "test"


def partition(dp_peptides, dq_peptides, initializer_segments):
    """Freeze DP components and remove declared initializer/source exposures."""
    dp_components = components(dp_peptides)
    contaminated = {
        component
        for peptide, component in dp_components.items()
        if kmers(peptide) & initializer_segments
    }
    assignments, dispositions = {}, []
    for peptide, component in sorted(dp_components.items()):
        split = dp_split(component)
        reason = (
            "initializer_exposed_evaluation_component"
            if split != "train" and component in contaminated
            else "included"
        )
        dispositions.append(
            dict(study="35251023", peptide=peptide, component=component, split=split, reason=reason)
        )
        if reason == "included":
            assignments[("35251023", peptide)] = (split, "dp:" + digest(component))
    dp_segments = {segment for peptide in dp_components for segment in kmers(peptide)}
    dq_components = components(dq_peptides)
    for peptide, component in sorted(dq_components.items()):
        segments = kmers(peptide)
        reason = (
            "dp_source_overlap"
            if segments & dp_segments
            else "initializer_overlap"
            if segments & initializer_segments
            else "included"
        )
        dispositions.append(
            dict(
                study="34433824",
                peptide=peptide,
                component=component,
                split="external",
                reason=reason,
            )
        )
        if reason == "included":
            assignments[("34433824", peptide)] = ("external", "dq8:" + digest(component))
    return assignments, dispositions


def backgrounds(assignments, known_positive, initializer_segments):
    """One permutation per peptide; reject declared positive/dependency collisions."""
    by_peptide = {}
    segments_to_splits = defaultdict(set)
    for (_, peptide), (split, _) in assignments.items():
        if peptide in by_peptide and by_peptide[peptide] != split:
            raise ValueError("positive peptide crosses splits")
        by_peptide[peptide] = split
        for segment in kmers(peptide):
            segments_to_splits[segment].add(split)
    used, result = set(), {}
    for peptide, split in sorted(by_peptide.items(), key=lambda item: (item[1], item[0])):
        seed = int(digest("presto-ms-background-v1|" + peptide), 16)
        rng = random.Random(seed)
        for attempt in range(1, 1001):
            residues = list(peptide)
            rng.shuffle(residues)
            candidate = "".join(residues)
            segments = kmers(candidate)
            if candidate == peptide or candidate in known_positive or candidate in used:
                continue
            if any(segments_to_splits[segment] - {split} for segment in segments):
                continue
            if split != "train" and segments & initializer_segments:
                continue
            assert Counter(candidate) == Counter(peptide)
            used.add(candidate)
            result[peptide] = dict(peptide=candidate, seed=str(seed), attempt=attempt, split=split)
            for segment in segments:
                segments_to_splits[segment].add(split)
            break
        else:
            raise ValueError(
                f"no valid background after1000 attempts for positive hash {digest(peptide)}"
            )
    return result
