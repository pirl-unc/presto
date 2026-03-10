#!/usr/bin/env python3
"""
Audit MHC-I groove extraction across all sequences in the MHC index.

Approach:
1. Find Cys-Cys pairs that define Ig-fold disulfide bonds
2. Use the alpha2 Cys pair to INFER mature start (back-calculate from
   the known ~100 mature-position of that Cys)
3. Split the groove into alpha1 and alpha2 domains
4. The alpha1/alpha2 split in MHC-I is structurally cognate to the
   alpha1(alpha chain)/beta1(beta chain) split in MHC-II

Key insight: we do NOT hard-code "90 aa" for alpha1. Instead we let
the Cys pair position define the boundary. If the alpha1 domain is
85 aa in sharks or 92 aa in fish, the Cys anchor finds it.

Second key insight: we do NOT rely on a hydrophobic-core heuristic to
find signal peptides. Instead, the Cys pair position tells us where the
mature protein starts: if the alpha2 first Cys is at raw position 124
but should be at mature position ~100, the signal peptide is ~24 aa.
"""

import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Cys pair finder
# ---------------------------------------------------------------------------

def find_cys_pairs(
    seq: str, min_sep: int = 48, max_sep: int = 72
) -> list[tuple[int, int, int]]:
    """
    Find all Cys-Cys pairs with separation in [min_sep, max_sep].
    Returns list of (cys1_idx, cys2_idx, separation).
    Sorted by cys1_idx (N-terminal first).
    """
    cyss = [i for i, aa in enumerate(seq) if aa == "C"]
    pairs = []
    for i, c1 in enumerate(cyss):
        for c2 in cyss[i + 1 :]:
            d = c2 - c1
            if min_sep <= d <= max_sep:
                pairs.append((c1, c2, d))
            elif d > max_sep:
                break
    return pairs


# ---------------------------------------------------------------------------
# MHC-I groove extraction
# ---------------------------------------------------------------------------

# The alpha2 Ig-fold disulfide has sep ~63 (range 55-71).
# The alpha3 Ig-fold disulfide has sep ~56 (range 48-64).
# These ranges overlap, so we rely on N-to-C ordering:
#   alpha2 Cys1 comes first, alpha3 Cys1 comes later.
# We also know alpha2 Cys1 is roughly at mature position ~100,
# and alpha3 Cys1 is roughly at mature position ~202.

IG_SEP_MIN = 48
IG_SEP_MAX = 72

# The alpha2 first Cys is at approximately mature position 101
# (0-indexed). This is extremely conserved across all jawed vertebrates.
ALPHA2_CYS1_MATURE_POS = 101

# The alpha2 first Cys sits ~10 residues into the alpha2 domain.
# So alpha2 starts at cys1 - 10 (approximately).
ALPHA2_CYS1_OFFSET_INTO_DOMAIN = 10

# Alpha2 ends ~20 residues after the second Cys of the pair.
ALPHA2_END_AFTER_CYS2 = 20

# For filtering: the alpha2 cys1 must be in this raw-sequence range
# to qualify. A full precursor with signal peptide puts it ~125;
# a mature protein puts it ~101. Fragments may be smaller.
ALPHA2_CYS1_RAW_MIN = 60   # must be at least 60 residues in
ALPHA2_CYS1_RAW_MAX = 180  # shouldn't be this late

# Fallback: alpha3 Cys1 is at mature position ~203 (conserved Ig-fold
# disulfide). If alpha2 pair is missing but alpha3 is found, we can
# back-calculate mature_start and use fixed domain boundaries.
ALPHA3_CYS1_MATURE_POS = 203
ALPHA3_CYS1_RAW_MIN = 180  # alpha3 pair must be past alpha2 region


@dataclass
class GrooveParseResult:
    allele: str = ""
    species: str = ""
    gene: str = ""
    seq_len: int = 0
    inferred_mature_start: int = 0
    n_cys_pairs: int = 0
    alpha2_cys1: Optional[int] = None
    alpha2_cys2: Optional[int] = None
    alpha2_sep: Optional[int] = None
    alpha3_cys1: Optional[int] = None
    alpha3_cys2: Optional[int] = None
    alpha3_sep: Optional[int] = None
    alpha1_seq: Optional[str] = None
    alpha2_seq: Optional[str] = None
    alpha1_len: Optional[int] = None
    alpha2_len: Optional[int] = None
    status: str = "ok"
    flags: list[str] = field(default_factory=list)


def _infer_mature_start_from_cys(cys1_raw: int, seq: str) -> int:
    """
    Given the raw position of the alpha2 first Cys, infer where the
    mature protein starts.

    If cys1_raw > ALPHA2_CYS1_MATURE_POS, the difference is the signal
    peptide length. If cys1_raw <= ALPHA2_CYS1_MATURE_POS, the sequence
    is already mature (possibly truncated at N-terminus).
    """
    sp_len = cys1_raw - ALPHA2_CYS1_MATURE_POS
    if sp_len < 0:
        # Sequence starts after the true N-terminus (truncated)
        return 0
    if sp_len > 40:
        # Signal peptides > 40 are suspicious; cap but still use
        return sp_len  # flag it but use the value
    return sp_len


def parse_mhci_groove(seq: str, allele: str = "", species: str = "", gene: str = "") -> GrooveParseResult:
    r = GrooveParseResult(allele=allele, species=species, gene=gene, seq_len=len(seq))

    # Step 1: find all Cys pairs with Ig-fold-like separation
    pairs = find_cys_pairs(seq, min_sep=IG_SEP_MIN, max_sep=IG_SEP_MAX)
    r.n_cys_pairs = len(pairs)

    if len(pairs) == 0:
        r.status = "no_cys_pairs"
        r.flags.append("no Ig-fold disulfide pairs found")
        return r

    # Step 2: identify alpha2 pair
    # The alpha2 pair is the first pair whose cys1 is plausibly in the
    # alpha2 domain (raw position >= ALPHA2_CYS1_RAW_MIN)
    alpha2_pair = None
    alpha3_pair = None
    for c1, c2, sep in pairs:
        if c1 < ALPHA2_CYS1_RAW_MIN:
            r.flags.append(f"skipped_early_cys({c1},{c2},sep={sep})")
            continue
        if c1 > ALPHA2_CYS1_RAW_MAX:
            r.flags.append(f"skipped_late_cys({c1},{c2},sep={sep})")
            continue
        if alpha2_pair is None:
            alpha2_pair = (c1, c2, sep)
        elif alpha3_pair is None:
            # alpha3 pair must be well downstream of alpha2
            if c1 > alpha2_pair[1] + 10:
                alpha3_pair = (c1, c2, sep)

    if alpha2_pair is None:
        # Fallback: look for alpha3 pair and back-calculate
        alpha3_fallback = None
        for c1, c2, sep in pairs:
            if c1 >= ALPHA3_CYS1_RAW_MIN:
                alpha3_fallback = (c1, c2, sep)
                break
        if alpha3_fallback is None:
            r.status = "no_alpha2_pair"
            r.flags.append(f"no qualifying Cys pair in range [{ALPHA2_CYS1_RAW_MIN},{ALPHA2_CYS1_RAW_MAX}]")
            return r
        # Use alpha3 pair to infer domain boundaries
        r.flags.append("alpha3_fallback")
        a3c1, a3c2, a3sep = alpha3_fallback
        r.alpha3_cys1, r.alpha3_cys2, r.alpha3_sep = a3c1, a3c2, a3sep
        r.inferred_mature_start = max(0, a3c1 - ALPHA3_CYS1_MATURE_POS)
        # Use fixed domain sizes: alpha1 = 91aa, alpha2 = 92aa
        alpha1_start = r.inferred_mature_start
        alpha2_start = alpha1_start + 91
        alpha2_end = min(a3c1 - 20, len(seq))  # alpha3 Ig starts ~20 before Cys1
        if alpha2_start >= alpha2_end or alpha2_start >= len(seq):
            r.status = "alpha3_fallback_bad_boundaries"
            r.flags.append(f"alpha3 fallback gave bad boundaries: a2=[{alpha2_start}:{alpha2_end}]")
            return r
        r.alpha1_seq = seq[alpha1_start:alpha2_start]
        r.alpha2_seq = seq[alpha2_start:alpha2_end]
        r.alpha1_len = len(r.alpha1_seq)
        r.alpha2_len = len(r.alpha2_seq)
        if r.alpha2_len < 60:
            r.flags.append(f"alpha2_short({r.alpha2_len})")
        if r.alpha2_len > 120:
            r.flags.append(f"alpha2_long({r.alpha2_len})")
        return r

    r.alpha2_cys1, r.alpha2_cys2, r.alpha2_sep = alpha2_pair
    if alpha3_pair:
        r.alpha3_cys1, r.alpha3_cys2, r.alpha3_sep = alpha3_pair

    # Step 3: infer mature start from the alpha2 Cys position
    r.inferred_mature_start = _infer_mature_start_from_cys(r.alpha2_cys1, seq)

    if r.inferred_mature_start > 40:
        r.flags.append(f"long_sp({r.inferred_mature_start})")
    if r.inferred_mature_start < 0:
        r.flags.append(f"negative_sp({r.inferred_mature_start})")
        r.inferred_mature_start = 0

    # Step 4: define domain boundaries from Cys positions
    # Alpha2 starts ~10 residues before its first Cys
    alpha2_start = r.alpha2_cys1 - ALPHA2_CYS1_OFFSET_INTO_DOMAIN
    # Alpha2 ends ~20 residues after its second Cys (or at alpha3 start, or at seq end)
    alpha2_end = min(r.alpha2_cys2 + ALPHA2_END_AFTER_CYS2, len(seq))

    # Alpha1 is from mature start to alpha2 start
    alpha1_start = r.inferred_mature_start
    alpha1_end = alpha2_start

    # Sanity: alpha1 should be positive length
    if alpha1_end <= alpha1_start:
        r.status = "alpha1_negative_length"
        r.flags.append(f"alpha1 would be [{alpha1_start}:{alpha1_end}]")
        return r

    r.alpha1_seq = seq[alpha1_start:alpha1_end]
    r.alpha2_seq = seq[alpha2_start:alpha2_end]
    r.alpha1_len = len(r.alpha1_seq)
    r.alpha2_len = len(r.alpha2_seq)

    # Sanity checks
    if r.alpha1_len < 50:
        r.flags.append(f"alpha1_short({r.alpha1_len})")
    if r.alpha1_len > 100:
        r.flags.append(f"alpha1_long({r.alpha1_len})")
    if r.alpha2_len < 60:
        r.flags.append(f"alpha2_short({r.alpha2_len})")
    if r.alpha2_len > 120:
        r.flags.append(f"alpha2_long({r.alpha2_len})")

    # Check for alpha3 as sanity
    if alpha3_pair is None and len(seq) > alpha2_end + 30:
        r.flags.append("no_alpha3_pair_but_sequence_continues")

    return r


# ---------------------------------------------------------------------------
# Main: run over all Class I sequences in the index
# ---------------------------------------------------------------------------

def main():
    index_path = "data/mhc_index.csv"

    with open(index_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter to Class I only, exclude MICA/MICB/HFE/TAP
    non_classical = {"MICA", "MICB", "HFE", "TAP1", "TAP2", "HLA-E", "HLA-F", "HLA-G"}
    class_i_rows = []
    skipped_nonclassical = 0
    for r in rows:
        if r["mhc_class"] != "I":
            continue
        gene = r["gene"]
        if any(gene.startswith(nc) or gene == nc for nc in non_classical):
            skipped_nonclassical += 1
            continue
        class_i_rows.append(r)

    print(f"Total Class I entries: {len(class_i_rows)} (skipped {skipped_nonclassical} non-classical)")
    print()

    results = []
    for row in class_i_rows:
        seq = row["sequence"].strip().upper()
        r = parse_mhci_groove(
            seq=seq,
            allele=row["allele_raw"],
            species=row["species"],
            gene=row["gene"],
        )
        results.append(r)

    # --- Summary statistics ---
    status_counts = Counter(r.status for r in results)
    print("=== Status distribution ===")
    for status, count in status_counts.most_common():
        pct = 100.0 * count / len(results)
        print(f"  {status}: {count} ({pct:.1f}%)")
    print()

    ok_results = [r for r in results if r.status == "ok"]
    print(f"Successfully parsed: {len(ok_results)} / {len(results)} ({100*len(ok_results)/len(results):.1f}%)")
    print()

    # Alpha1 length distribution
    a1_lens = [r.alpha1_len for r in ok_results]
    a2_lens = [r.alpha2_len for r in ok_results]
    print("=== Alpha1 (groove half 1) length distribution ===")
    a1_counter = Counter(a1_lens)
    for length, count in sorted(a1_counter.items()):
        bar = "#" * min(count // 10, 80)
        if count >= 10 or length < 80 or length > 100:
            print(f"  {length:3d} aa: {count:6d}  {bar}")
    print(f"  min={min(a1_lens)}, median={sorted(a1_lens)[len(a1_lens)//2]}, max={max(a1_lens)}")
    print()

    print("=== Alpha2 (groove half 2) length distribution ===")
    a2_counter = Counter(a2_lens)
    for length, count in sorted(a2_counter.items()):
        bar = "#" * min(count // 10, 80)
        if count >= 10 or length < 80 or length > 100:
            print(f"  {length:3d} aa: {count:6d}  {bar}")
    print(f"  min={min(a2_lens)}, median={sorted(a2_lens)[len(a2_lens)//2]}, max={max(a2_lens)}")
    print()

    # Alpha2 disulfide separation distribution
    a2_seps = [r.alpha2_sep for r in ok_results if r.alpha2_sep is not None]
    print("=== Alpha2 disulfide Cys-Cys separation ===")
    sep_counter = Counter(a2_seps)
    for sep, count in sorted(sep_counter.items()):
        bar = "#" * min(count // 10, 80)
        print(f"  sep={sep}: {count:6d}  {bar}")
    print()

    # Inferred mature start distribution
    starts = [r.inferred_mature_start for r in ok_results]
    print("=== Inferred mature start (signal peptide length) ===")
    start_counter = Counter(starts)
    for start, count in sorted(start_counter.items()):
        bar = "#" * min(count // 10, 80)
        if count >= 5:
            print(f"  {start:3d} aa: {count:6d}  {bar}")
    print(f"  0 (no SP / already mature): {start_counter.get(0, 0)}")
    sp_entries = [r for r in ok_results if r.inferred_mature_start > 0]
    if sp_entries:
        sp_lens = [r.inferred_mature_start for r in sp_entries]
        print(f"  With SP: {len(sp_entries)} ({100*len(sp_entries)/len(ok_results):.1f}%)")
        print(f"  SP len: min={min(sp_lens)}, median={sorted(sp_lens)[len(sp_lens)//2]}, max={max(sp_lens)}")
    print()

    # Species breakdown for failures
    failures = [r for r in results if r.status != "ok"]
    if failures:
        print(f"=== Failures ({len(failures)}) ===")
        species_fails = defaultdict(list)
        for r in failures:
            species_fails[r.species or "(empty)"].append(r)
        for sp, fails in sorted(species_fails.items(), key=lambda x: -len(x[1])):
            print(f"  {sp}: {len(fails)} failures")
            for f in fails[:3]:
                print(f"    {f.allele} (len={f.seq_len}, status={f.status}, flags={f.flags})")
            if len(fails) > 3:
                print(f"    ... and {len(fails)-3} more")
        print()

    # Flagged but successful
    flagged = [r for r in ok_results if r.flags]
    if flagged:
        print(f"=== Flagged but successful ({len(flagged)}) ===")
        flag_counts = Counter()
        for r in flagged:
            for f in r.flags:
                # Normalize flag name
                fname = f.split("(")[0]
                flag_counts[fname] += 1
        for fname, count in flag_counts.most_common():
            print(f"  {fname}: {count}")
        print()
        # Show a few examples of unusual lengths
        short_a1 = [r for r in ok_results if r.alpha1_len and r.alpha1_len < 80]
        long_a1 = [r for r in ok_results if r.alpha1_len and r.alpha1_len > 100]
        if short_a1:
            print(f"  Short alpha1 examples (<80aa): {len(short_a1)}")
            for r in short_a1[:5]:
                sp_note = f" (SP={r.inferred_mature_start})" if r.inferred_mature_start > 0 else ""
                print(f"    {r.allele} ({r.species}): a1={r.alpha1_len}, a2={r.alpha2_len}, seq_len={r.seq_len}{sp_note}")
        if long_a1:
            print(f"  Long alpha1 examples (>100aa): {len(long_a1)}")
            for r in long_a1[:5]:
                sp_note = f" (SP={r.inferred_mature_start})" if r.inferred_mature_start > 0 else ""
                print(f"    {r.allele} ({r.species}): a1={r.alpha1_len}, a2={r.alpha2_len}, seq_len={r.seq_len}{sp_note}")

    # Per-species summary for successful parses
    print()
    print("=== Per-species groove dimensions (successful parses) ===")
    species_stats = defaultdict(lambda: {"count": 0, "a1_lens": [], "a2_lens": []})
    for r in ok_results:
        sp = r.species or "(empty)"
        species_stats[sp]["count"] += 1
        species_stats[sp]["a1_lens"].append(r.alpha1_len)
        species_stats[sp]["a2_lens"].append(r.alpha2_len)

    for sp, stats in sorted(species_stats.items(), key=lambda x: -x[1]["count"]):
        if stats["count"] < 3:
            continue
        a1 = stats["a1_lens"]
        a2 = stats["a2_lens"]
        a1_med = sorted(a1)[len(a1)//2]
        a2_med = sorted(a2)[len(a2)//2]
        a1_range = f"{min(a1)}-{max(a1)}"
        a2_range = f"{min(a2)}-{max(a2)}"
        print(f"  {sp:35s}: n={stats['count']:5d}  a1={a1_med:3d} ({a1_range:>7s})  a2={a2_med:3d} ({a2_range:>7s})")


if __name__ == "__main__":
    main()
