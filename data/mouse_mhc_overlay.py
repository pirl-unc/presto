"""Build a mouse MHC sequence overlay from IMGT nomenclature + UniProt.

This module intentionally keeps provenance explicit for every emitted protein
sequence so questionable mappings can be traced and corrected later.
"""

from __future__ import annotations

import csv
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .allele_resolver import infer_mhc_class


IMGT_MOUSE_MHC_NOMENCLATURE_URL = (
    "https://www.imgt.org/IMGTrepertoireMH/LocusGenes/nomenclatures/mouse/MHC/Mu_MHCnom.html"
)
UNIPROT_MOUSE_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL_TEMPLATE = "https://www.uniprot.org/uniprotkb/{accession}"


_IMGT_TOKEN_RE = re.compile(r"\b(?:MH[12]-[A-Z0-9-]+|H2-[A-Z0-9-]+)\b")
_UNIPROT_HAP_RE = re.compile(
    r",\s*([A-Z0-9]+)-([A-Z0-9]+)\s+(?:alpha|beta|M alpha|M beta(?:\s+\d+)?)\s+chain\b"
)


_GENE_REMAP: Dict[str, str] = {
    "MH1-K1": "H2-K1",
    "MH2-AA": "H2-AA",
}

_SPECIAL_UNIPROT_GENE_QUERY: Dict[str, List[str]] = {
    "H2-K": ["H2-K1"],
    "H2-D": ["H2-D1"],
    "H2-AA": ["H2-Aa"],
    "H2-AB": ["H2-Ab1"],
    "H2-EA": ["H2-Ea"],
    "H2-EB": ["H2-Eb1"],
    "H2-DMA": ["H2-DMa"],
    "H2-DMB1": ["H2-DMb1"],
    "H2-DMB2": ["H2-DMb2"],
    "H2-OA": ["H2-Oa"],
    "H2-OB": ["H2-Ob"],
    "H2-PA": ["H2-Pa"],
    "H2-PB": ["H2-Pb"],
}

_GENE_TO_FAMILY: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"^H2-K\d*$"), "H2-K"),
    (re.compile(r"^H2-D\d*$"), "H2-D"),
    (re.compile(r"^H2-L$"), "H2-L"),
    (re.compile(r"^H2-AA$|^H2-AA\d*$|^H2-AA$|^H2-Aa$"), "H2-AA"),
    (re.compile(r"^H2-AB\d*$|^H2-Ab\d*$"), "H2-AB"),
    (re.compile(r"^H2-EA$|^H2-Ea$"), "H2-EA"),
    (re.compile(r"^H2-EB\d*$|^H2-Eb\d*$"), "H2-EB"),
    (re.compile(r"^H2-DMA$|^H2-DMa$"), "H2-DMA"),
    (re.compile(r"^H2-DMB\d*$|^H2-DMb\d*$"), "H2-DMB1"),
    (re.compile(r"^H2-Q\d*$"), "H2-Q"),
    (re.compile(r"^H2-T\d*$"), "H2-T"),
]

_PROVENANCE_COLUMNS: List[str] = [
    "selected",
    "selection_reason",
    "allele_token",
    "mhc_class",
    "species",
    "sequence",
    "seq_len",
    "imgt_gene_symbol",
    "imgt_source_url",
    "uniprot_gene_query",
    "uniprot_accession",
    "uniprot_entry_id",
    "uniprot_entry_type",
    "uniprot_record_url",
    "uniprot_protein_name",
    "uniprot_gene_names",
    "allele_derivation_rule",
    "build_timestamp_utc",
]


def _fetch_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Presto-MouseMHCOverlay/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def _normalize_imgt_gene_symbol(token: str) -> str:
    value = token.strip().upper()
    value = value.rstrip("-")
    if value in _GENE_REMAP:
        return _GENE_REMAP[value]
    return value


def _is_gene_like_imgt_symbol(symbol: str) -> bool:
    # Skip broad loci unless they are directly useful for sequence lookup.
    broad = {
        "H2-A",
        "H2-D",
        "H2-DM",
        "H2-E",
        "H2-I",
        "H2-IA",
        "H2-IE",
        "H2-K",
        "H2-M",
        "H2-O",
        "H2-P",
        "H2-Q",
        "H2-T",
    }
    if symbol in broad:
        return False
    if not symbol.startswith("H2-"):
        return False
    if symbol.endswith("-"):
        return False
    tail = symbol.split("-", 1)[1]
    return bool(tail)


def parse_imgt_mouse_mhc_genes(html: str) -> List[str]:
    tokens = {_normalize_imgt_gene_symbol(tok) for tok in _IMGT_TOKEN_RE.findall(html)}
    filtered = sorted(tok for tok in tokens if _is_gene_like_imgt_symbol(tok))
    return filtered


def fetch_imgt_mouse_mhc_genes(
    imgt_url: str = IMGT_MOUSE_MHC_NOMENCLATURE_URL,
    timeout: int = 30,
) -> List[str]:
    html = _fetch_text(imgt_url, timeout=timeout)
    return parse_imgt_mouse_mhc_genes(html)


def _candidate_uniprot_gene_queries(imgt_gene_symbol: str) -> List[str]:
    symbol = imgt_gene_symbol.strip().upper()
    candidates: List[str] = []
    for value in _SPECIAL_UNIPROT_GENE_QUERY.get(symbol, []):
        if value not in candidates:
            candidates.append(value)

    # Direct symbol is still useful for many H2-Q/H2-T genes.
    if symbol not in candidates:
        candidates.append(symbol)

    # Camel-case conversion for symbols like H2-AB -> H2-Ab, H2-DMA -> H2-DMa.
    if symbol.startswith("H2-"):
        tail = symbol[3:]
        m = re.fullmatch(r"([A-Z]+)(\d*)", tail)
        if m:
            letters, digits = m.groups()
            if letters:
                camel = letters[0] + letters[1:].lower() + digits
                candidate = f"H2-{camel}"
                if candidate not in candidates:
                    candidates.append(candidate)
                if letters in {"AB", "EB"}:
                    candidate_beta = f"H2-{letters[0]}{letters[1].lower()}1"
                    candidate_beta = f"H2-{candidate_beta.split('-', 1)[1]}"
                    if candidate_beta not in candidates:
                        candidates.append(candidate_beta)
        if tail in {"K", "D"}:
            candidate = f"H2-{tail}1"
            if candidate not in candidates:
                candidates.append(candidate)

    return candidates


def _uniprot_search(query: str, timeout: int = 30, size: int = 100) -> List[Dict[str, object]]:
    params = {
        "query": query,
        "fields": (
            "accession,id,reviewed,protein_name,gene_names,organism_name,length,sequence"
        ),
        "format": "json",
        "size": str(size),
    }
    url = UNIPROT_MOUSE_SEARCH_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Presto-MouseMHCOverlay/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.load(response)
    results = payload.get("results", [])
    if isinstance(results, list):
        return [row for row in results if isinstance(row, dict)]
    return []


def _extract_uniprot_protein_name(row: Dict[str, object]) -> str:
    desc = row.get("proteinDescription")
    if not isinstance(desc, dict):
        return ""
    rec = desc.get("recommendedName")
    if isinstance(rec, dict):
        full = rec.get("fullName")
        if isinstance(full, dict):
            value = full.get("value")
            if isinstance(value, str):
                return value
    return ""


def _extract_uniprot_gene_names(row: Dict[str, object]) -> List[str]:
    genes = row.get("genes")
    if not isinstance(genes, list):
        return []
    names: List[str] = []
    for entry in genes:
        if not isinstance(entry, dict):
            continue
        gene_name = entry.get("geneName")
        if isinstance(gene_name, dict):
            value = gene_name.get("value")
            if isinstance(value, str) and value:
                names.append(value)
        synonyms = entry.get("synonyms")
        if isinstance(synonyms, list):
            for syn in synonyms:
                if not isinstance(syn, dict):
                    continue
                value = syn.get("value")
                if isinstance(value, str) and value:
                    names.append(value)
    deduped: List[str] = []
    seen = set()
    for value in names:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _canonical_family_from_gene_symbol(gene_symbol: str) -> Optional[str]:
    for pattern, family in _GENE_TO_FAMILY:
        if pattern.match(gene_symbol):
            return family
    return None


def _derive_alleles_from_uniprot_row(
    *,
    imgt_gene_symbol: str,
    uniprot_gene_query: str,
    protein_name: str,
    uniprot_gene_names: Sequence[str],
) -> List[Tuple[str, str]]:
    family = _canonical_family_from_gene_symbol(imgt_gene_symbol)
    if not family:
        return []

    alleles: List[Tuple[str, str]] = []
    m = _UNIPROT_HAP_RE.search(protein_name or "")
    if m:
        protein_family, haplotype = m.groups()
        family_tail = family.split("-", 1)[1]
        if protein_family == family_tail[0]:
            allele = f"{family}*{haplotype.lower()}"
            alleles.append((allele, "protein_name_haplotype"))

    if not alleles:
        for name in uniprot_gene_names:
            match = re.search(r"\bH2-[A-Z](?:[A-Z0-9]+)?([bdkqrsuw0-9]+)\b", name.lower())
            if match:
                allele = f"{family}*{match.group(1)}"
                alleles.append((allele, "gene_name_haplotype_fallback"))

    deduped: List[Tuple[str, str]] = []
    seen = set()
    for allele, rule in alleles:
        key = (allele, rule)
        if key in seen:
            continue
        deduped.append((allele, rule))
        seen.add(key)
    return deduped


def _entry_is_reviewed(entry_type: str) -> bool:
    return entry_type.lower().startswith("uniprotkb reviewed")


def build_mouse_mhc_overlay(
    *,
    out_csv: str,
    out_fasta: str,
    imgt_url: str = IMGT_MOUSE_MHC_NOMENCLATURE_URL,
    reviewed_only: bool = True,
    max_genes: int = 0,
    timeout: int = 30,
) -> Dict[str, int]:
    genes = fetch_imgt_mouse_mhc_genes(imgt_url=imgt_url, timeout=timeout)
    if max_genes > 0:
        genes = genes[:max_genes]

    timestamp = datetime.now(timezone.utc).isoformat()
    rows: List[Dict[str, str]] = []
    selected_by_allele: Dict[str, Dict[str, str]] = {}

    for imgt_gene in genes:
        queries = _candidate_uniprot_gene_queries(imgt_gene)
        seen_accessions = set()
        for gene_query in queries:
            base_query = f"gene_exact:{gene_query} AND organism_id:10090"
            query = base_query + (" AND reviewed:true" if reviewed_only else "")
            uniprot_rows = _uniprot_search(query=query, timeout=timeout)
            for row in uniprot_rows:
                accession = str(row.get("primaryAccession") or "").strip()
                if not accession or accession in seen_accessions:
                    continue
                seen_accessions.add(accession)

                seq = row.get("sequence")
                sequence = ""
                if isinstance(seq, dict):
                    seq_value = seq.get("value")
                    if isinstance(seq_value, str):
                        sequence = seq_value.strip().upper()
                if not sequence:
                    continue

                entry_type = str(row.get("entryType") or row.get("entry_type") or "")
                if reviewed_only and not _entry_is_reviewed(entry_type):
                    continue

                protein_name = _extract_uniprot_protein_name(row)
                gene_names = _extract_uniprot_gene_names(row)
                alleles = _derive_alleles_from_uniprot_row(
                    imgt_gene_symbol=imgt_gene,
                    uniprot_gene_query=gene_query,
                    protein_name=protein_name,
                    uniprot_gene_names=gene_names,
                )
                if not alleles:
                    continue

                for allele_token, derivation_rule in alleles:
                    mhc_class = infer_mhc_class(allele_token)
                    out_row = {
                        "selected": "0",
                        "selection_reason": "",
                        "allele_token": allele_token,
                        "mhc_class": mhc_class,
                        "species": "mouse",
                        "sequence": sequence,
                        "seq_len": str(len(sequence)),
                        "imgt_gene_symbol": imgt_gene,
                        "imgt_source_url": imgt_url,
                        "uniprot_gene_query": gene_query,
                        "uniprot_accession": accession,
                        "uniprot_entry_id": str(row.get("uniProtkbId") or ""),
                        "uniprot_entry_type": entry_type,
                        "uniprot_record_url": UNIPROT_ENTRY_URL_TEMPLATE.format(accession=accession),
                        "uniprot_protein_name": protein_name,
                        "uniprot_gene_names": ";".join(gene_names),
                        "allele_derivation_rule": derivation_rule,
                        "build_timestamp_utc": timestamp,
                    }
                    rows.append(out_row)

                    prev = selected_by_allele.get(allele_token)
                    if prev is None:
                        out_row["selected"] = "1"
                        out_row["selection_reason"] = "first_match"
                        selected_by_allele[allele_token] = out_row
                        continue

                    prev_len = int(prev.get("seq_len") or "0")
                    cur_len = int(out_row.get("seq_len") or "0")
                    prev_reviewed = _entry_is_reviewed(prev.get("uniprot_entry_type", ""))
                    cur_reviewed = _entry_is_reviewed(out_row.get("uniprot_entry_type", ""))
                    prev_acc = prev.get("uniprot_accession", "")
                    cur_acc = out_row.get("uniprot_accession", "")

                    current_wins = (cur_reviewed, cur_len, cur_acc) > (prev_reviewed, prev_len, prev_acc)
                    if current_wins:
                        prev["selected"] = "0"
                        prev["selection_reason"] = "replaced_by_higher_rank"
                        out_row["selected"] = "1"
                        out_row["selection_reason"] = "highest_rank"
                        selected_by_allele[allele_token] = out_row
                    else:
                        out_row["selected"] = "0"
                        out_row["selection_reason"] = "lower_rank_duplicate"

    csv_path = Path(out_csv)
    fasta_path = Path(out_fasta)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)

    rows.sort(key=lambda r: (r["allele_token"], r["uniprot_accession"]))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_PROVENANCE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in _PROVENANCE_COLUMNS})

    selected = [row for row in rows if row.get("selected") == "1"]
    selected.sort(key=lambda r: r["allele_token"])
    with fasta_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            header = (
                f">{row['allele_token']} source=uniprot_mouse_overlay "
                f"accession={row['uniprot_accession']} gene={row['imgt_gene_symbol']}"
            )
            handle.write(header + "\n")
            handle.write(row["sequence"] + "\n")

    return {
        "imgt_genes": len(genes),
        "catalog_rows": len(rows),
        "selected_alleles": len(selected),
        "fasta_records": len(selected),
    }
