"""Shared sequence protection and source-matched presentation backgrounds.

A reference occurrence is a possible source, not a measured protein assignment.
All matching proteins define the split family. Ambiguous junctions stay unknown.
"""

from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import math
from pathlib import Path

from .modular import AA


def stable_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class RelatedSequences:
    """Exact, shared contiguous 8mer, or equal-length Hamming distance <= 1."""

    def __init__(self, peptides):
        self.peptides = set(peptides)
        self.cores = {p[i : i + 8] for p in self.peptides for i in range(len(p) - 7)}
        self.halves = defaultdict(set)
        for peptide in self.peptides:
            midpoint = len(peptide) // 2
            for side, part in ((0, peptide[:midpoint]), (1, peptide[midpoint:])):
                self.halves[len(peptide), side, part].add(peptide)

    def reason(self, peptide):
        if peptide in self.peptides:
            return "exact"
        if any(peptide[i : i + 8] in self.cores for i in range(len(peptide) - 7)):
            return "shared8mer"
        midpoint = len(peptide) // 2
        for side, part in ((0, peptide[:midpoint]), (1, peptide[midpoint:])):
            for other in self.halves.get((len(peptide), side, part), ()):
                if sum(a != b for a, b in zip(peptide, other, strict=True)) <= 1:
                    return "hamming1"
        return None


@dataclass(frozen=True)
class Protein:
    identifier: str
    gene: str
    sequence: str


def read_gencode_proteins(path):
    """Read the explicit GENCODE protein/transcript/gene header contract."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    proteins, header, chunks = {}, None, []

    def finish():
        if header is None:
            return
        fields = header.split("|")
        if len(fields) < 3 or not fields[0].startswith("ENSP") or not fields[2].startswith("ENSG"):
            raise ValueError("expected explicit GENCODE protein and gene identifiers")
        sequence = "".join(chunks)
        if fields[0] in proteins and proteins[fields[0]].sequence != sequence:
            raise ValueError("conflicting reference sequence for protein ID")
        proteins[fields[0]] = Protein(fields[0], fields[2], sequence)

    with opener(path, "rt") as stream:
        for line in stream:
            line = line.strip()
            if line.startswith(">"):
                finish()
                header, chunks = line[1:], []
            elif line:
                chunks.append(line)
    finish()
    if not proteins:
        raise ValueError("empty protein reference")
    return proteins


def map_peptides(peptides, proteins):
    """Find every exact reference occurrence, with no per-peptide hit cap."""
    import ahocorasick

    automaton = ahocorasick.Automaton()
    for peptide in sorted(set(peptides)):
        if not peptide or set(peptide) - set(AA):
            raise ValueError("mapping requires canonical nonempty peptide sequences")
        automaton.add_word(peptide, peptide)
    hits = defaultdict(list)
    if not len(automaton):
        return hits
    automaton.make_automaton()
    for identifier, protein in sorted(proteins.items()):
        for stop, peptide in automaton.iter(protein.sequence):
            hits[peptide].append((identifier, stop + 1 - len(peptide)))
    return hits


def protein_families(hits):
    """Union every possible source of any peptide, without choosing an isoform."""
    parents = {}

    def find(value):
        parents.setdefault(value, value)
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    for occurrences in hits.values():
        ids = sorted({identifier for identifier, _ in occurrences})
        if not ids:
            continue
        for identifier in ids:
            left, right = find(ids[0]), find(identifier)
            if left != right:
                parents[max(left, right)] = min(left, right)
    return {identifier: find(identifier) for identifier in parents}


def junction(sequence, start, length, flank_length=15):
    n_flank = sequence[max(0, start - flank_length) : start]
    c_flank = sequence[start + length : start + length + flank_length]
    return {
        "source_n": None if set(n_flank) - set(AA) else n_flank,
        "source_c": None if set(c_flank) - set(AA) else c_flank,
        "n_terminus": start == 0,
        "c_terminus": start + length == len(sequence),
    }


def source_junction(peptide, occurrences, proteins):
    """Preserve unanimous junctions; mask disagreements without dropping labels."""
    contexts = []
    for identifier, start in occurrences:
        sequence = proteins[identifier].sequence
        if sequence[start : start + len(peptide)] != peptide:
            raise ValueError("source occurrence disagrees with reference")
        contexts.append(junction(sequence, start, len(peptide)))
    if not contexts:
        raise ValueError("source peptide has no reference occurrence")
    if all(value == contexts[0] for value in contexts):
        return contexts[0], "unanimous_reference_junction"
    return {
        "source_n": None,
        "source_c": None,
        "n_terminus": None,
        "c_terminus": None,
    }, "ambiguous_reference_junction"


def background_candidate(peptide, occurrences, proteins, *, observed, protected, key):
    """Deterministic same-length candidate from a possible source; no hit cap.

    Excludes exact known observations and related protected sequences. Caller
    applies the same missing-junction policy to both members of an ambiguous pair.
    """
    identifiers = sorted({identifier for identifier, _ in occurrences})
    ordered = sorted(identifiers, key=lambda name: stable_digest(key + ":protein:" + name))
    for identifier in ordered:
        sequence = proteins[identifier].sequence
        count = len(sequence) - len(peptide) + 1
        if count <= 0:
            continue
        seed = int(stable_digest(key + ":positions:" + identifier), 16)
        start_at, stride = seed % count, (seed // count) % count or 1
        while math.gcd(stride, count) != 1:
            stride += 1
        for offset in range(count):
            start = (start_at + offset * stride) % count
            candidate = sequence[start : start + len(peptide)]
            if candidate == peptide or candidate in observed or set(candidate) - set(AA):
                continue
            if protected.reason(candidate):
                continue
            return candidate, identifier, start, junction(sequence, start, len(peptide))
    return None
