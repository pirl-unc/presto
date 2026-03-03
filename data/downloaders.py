"""Data downloaders for immunology datasets.

Provides functions to download immunology datasets from their official sources:
- IEDB: T-cell, B-cell, MHC ligand binding data
- CEDAR: Cancer epitope data (sibling database to IEDB, cancer-specific curation)
- VDJdb: TCR-pMHC paired data
- McPAS-TCR: Pathology-associated TCR sequences
- IMGT/HLA: HLA protein sequences
- IMGT/GENE-DB: V/D/J gene sequences
- IPD-MHC: Non-human MHC sequences
- 10x Genomics: Public single-cell immune profiling datasets
- PIRD: Pan Immune Repertoire Database

Also provides reference-based deduplication for assay data.
"""

import csv
import gzip
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Set, Tuple, Any, Iterator
import sys


# =============================================================================
# Dataset Registry
# =============================================================================

@dataclass
class DatasetInfo:
    """Information about a downloadable dataset."""
    name: str
    description: str
    url: str
    filename: str
    source: str  # iedb, vdjdb, mcpas, imgt, ipd_mhc, 10x, pird
    category: str  # binding, tcell, bcell, tcr, mhc_sequence, vdj_genes, elution
    file_format: str  # csv, tsv, zip, fasta, json
    requires_agreement: bool = False
    post_process: Optional[str] = None  # Name of post-processing function
    species: Optional[str] = None  # human, mouse, macaque, etc.
    version: Optional[str] = None  # Dataset version if known
    # Note: size is fetched dynamically via HEAD request, not stored here


# Official download URLs - comprehensive registry
# Note: IEDB/CEDAR require manual download from their websites due to session-based auth
DATASETS: Dict[str, DatasetInfo] = {
    # =========================================================================
    # IEDB Datasets (primary immunology database)
    # All IEDB data requires agreement to terms: https://www.iedb.org/terms_of_use.php
    # Download manually from: https://www.iedb.org/database_export_v3.php
    # =========================================================================
    "iedb_mhc_ligand": DatasetInfo(
        name="iedb_mhc_ligand",
        description="IEDB MHC ligand binding assays (IC50, KD, EC50) + mass spec/elution",
        url="https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip",
        filename="mhc_ligand_full_single_file.zip",
        source="iedb",
        category="binding",
        file_format="zip",
        requires_agreement=True,
        post_process="unzip",
    ),
    "iedb_tcell": DatasetInfo(
        name="iedb_tcell",
        description="IEDB T-cell epitope assays (all T-cell response data)",
        url="https://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip",
        filename="tcell_full_v3.zip",
        source="iedb",
        category="tcell",
        file_format="zip",
        requires_agreement=True,
        post_process="unzip",
    ),
    "iedb_bcell": DatasetInfo(
        name="iedb_bcell",
        description="IEDB B-cell epitope assays (antibody response data)",
        url="https://www.iedb.org/downloader.php?file_name=doc/bcell_full_v3_single_file.zip",
        filename="bcell_full_v3_single_file.zip",
        source="iedb",
        category="bcell",
        file_format="zip",
        requires_agreement=True,
        post_process="unzip",
    ),
    # CEDAR: sibling database to IEDB (cancer-specific curation, same export format)
    # Download manually from: https://cedar.iedb.org/
    "iedb_cedar_tcell": DatasetInfo(
        name="iedb_cedar_tcell",
        description="IEDB/CEDAR curated cancer T-cell epitopes",
        url="https://cedar.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip",
        filename="cedar_tcell_full_v3.zip",
        source="iedb",
        category="tcell",
        file_format="zip",
        requires_agreement=True,
        post_process="unzip",
    ),
    "iedb_cedar_bcell": DatasetInfo(
        name="iedb_cedar_bcell",
        description="IEDB/CEDAR curated cancer B-cell epitopes",
        url="https://cedar.iedb.org/downloader.php?file_name=doc/bcell_full_v3_single_file.zip",
        filename="cedar_bcell_full_single_file.zip",
        source="iedb",
        category="bcell",
        file_format="zip",
        requires_agreement=True,
        post_process="unzip",
    ),
    "iedb_cedar_mhc_ligand": DatasetInfo(
        name="iedb_cedar_mhc_ligand",
        description="IEDB/CEDAR curated cancer MHC ligand + elution data",
        url="https://cedar.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip",
        filename="cedar_mhc_ligand_full_single_file.zip",
        source="iedb",
        category="binding",
        file_format="zip",
        requires_agreement=True,
        post_process="unzip",
    ),

    # =========================================================================
    # VDJdb - TCR-pMHC paired data
    # =========================================================================
    "vdjdb": DatasetInfo(
        name="vdjdb",
        description="VDJdb TCR-pMHC paired sequences with V/J gene annotations",
        url="https://github.com/antigenomics/vdjdb-db/releases/download/2025-12-29/vdjdb-2025-12-29.zip",
        filename="vdjdb.zip",
        source="vdjdb",
        category="tcr",
        file_format="zip",
        post_process="unzip",
        version="2025-12-29",
    ),

    # =========================================================================
    # McPAS-TCR - Pathology-associated TCR sequences
    # =========================================================================
    "mcpas": DatasetInfo(
        name="mcpas",
        description="McPAS-TCR pathology-associated TCR sequences",
        url="https://gitlab.com/immunomind/immunarch/-/raw/dev-0.5.0/private/McPAS-TCR.csv.gz",
        filename="McPAS-TCR.csv.gz",
        source="mcpas",
        category="tcr",
        file_format="csv",
        post_process="gunzip",
    ),

    # =========================================================================
    # IMGT/HLA - Human HLA sequences
    # =========================================================================
    "imgt_hla": DatasetInfo(
        name="imgt_hla",
        description="IMGT/HLA human HLA protein sequences (all alleles)",
        url="https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/hla_prot.fasta",
        filename="hla_prot.fasta",
        source="imgt",
        category="mhc_sequence",
        file_format="fasta",
        species="human",
    ),
    "imgt_hla_nuc": DatasetInfo(
        name="imgt_hla_nuc",
        description="IMGT/HLA human HLA nucleotide sequences",
        url="https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/hla_nuc.fasta",
        filename="hla_nuc.fasta",
        source="imgt",
        category="mhc_sequence",
        file_format="fasta",
        species="human",
    ),
    "imgt_hla_a": DatasetInfo(
        name="imgt_hla_a",
        description="IMGT/HLA HLA-A protein sequences",
        url="https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/A_prot.fasta",
        filename="hla_a_prot.fasta",
        source="imgt",
        category="mhc_sequence",
        file_format="fasta",
        species="human",
    ),
    "imgt_hla_b": DatasetInfo(
        name="imgt_hla_b",
        description="IMGT/HLA HLA-B protein sequences",
        url="https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/B_prot.fasta",
        filename="hla_b_prot.fasta",
        source="imgt",
        category="mhc_sequence",
        file_format="fasta",
        species="human",
    ),
    "imgt_hla_c": DatasetInfo(
        name="imgt_hla_c",
        description="IMGT/HLA HLA-C protein sequences",
        url="https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/C_prot.fasta",
        filename="hla_c_prot.fasta",
        source="imgt",
        category="mhc_sequence",
        file_format="fasta",
        species="human",
    ),
    "imgt_hla_drb1": DatasetInfo(
        name="imgt_hla_drb1",
        description="IMGT/HLA HLA-DRB1 protein sequences",
        url="https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/DRB1_prot.fasta",
        filename="hla_drb1_prot.fasta",
        source="imgt",
        category="mhc_sequence",
        file_format="fasta",
        species="human",
    ),
    "imgt_hla_dqa1": DatasetInfo(
        name="imgt_hla_dqa1",
        description="IMGT/HLA HLA-DQA1 protein sequences",
        url="https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/DQA1_prot.fasta",
        filename="hla_dqa1_prot.fasta",
        source="imgt",
        category="mhc_sequence",
        file_format="fasta",
        species="human",
    ),
    "imgt_hla_dqb1": DatasetInfo(
        name="imgt_hla_dqb1",
        description="IMGT/HLA HLA-DQB1 protein sequences",
        url="https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/DQB1_prot.fasta",
        filename="hla_dqb1_prot.fasta",
        source="imgt",
        category="mhc_sequence",
        file_format="fasta",
        species="human",
    ),
    "imgt_hla_dpb1": DatasetInfo(
        name="imgt_hla_dpb1",
        description="IMGT/HLA HLA-DPB1 protein sequences",
        url="https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/DPB1_prot.fasta",
        filename="hla_dpb1_prot.fasta",
        source="imgt",
        category="mhc_sequence",
        file_format="fasta",
        species="human",
    ),
    "imgt_allele_list": DatasetInfo(
        name="imgt_allele_list",
        description="IMGT/HLA complete allele list with metadata",
        url="https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/wmda/hla_nom.txt",
        filename="hla_nom.txt",
        source="imgt",
        category="mhc_sequence",
        file_format="tsv",
        species="human",
    ),

    # =========================================================================
    # IMGT/GENE-DB - V/D/J gene sequences
    # =========================================================================
    "imgt_trav": DatasetInfo(
        name="imgt_trav",
        description="IMGT human TRAV gene sequences",
        url="https://www.imgt.org/download/GENE-DB/IMGTGENEDB-ReferenceSequences.fasta-nt-WithGaps-F+ORF+inframeP",
        filename="imgt_trav.fasta",
        source="imgt",
        category="vdj_genes",
        file_format="fasta",
        species="human",
        post_process="filter_trav",
    ),
    "imgt_trbv": DatasetInfo(
        name="imgt_trbv",
        description="IMGT human TRBV gene sequences",
        url="https://www.imgt.org/download/GENE-DB/IMGTGENEDB-ReferenceSequences.fasta-nt-WithGaps-F+ORF+inframeP",
        filename="imgt_trbv.fasta",
        source="imgt",
        category="vdj_genes",
        file_format="fasta",
        species="human",
        post_process="filter_trbv",
    ),

    # =========================================================================
    # IPD-MHC - Non-human MHC sequences (via FTP)
    # =========================================================================
    "ipd_mhc_nhp": DatasetInfo(
        name="ipd_mhc_nhp",
        description="IPD-MHC non-human primate MHC sequences",
        url="https://raw.githubusercontent.com/ANHIG/IPDMHC/Latest/MHC_prot.fasta",
        filename="ipd_mhc_prot.fasta",
        source="ipd_mhc",
        category="mhc_sequence",
        file_format="fasta",
        species="nhp",
    ),
    "ipd_mhc_nuc": DatasetInfo(
        name="ipd_mhc_nuc",
        description="IPD-MHC non-human MHC nucleotide sequences",
        url="https://raw.githubusercontent.com/ANHIG/IPDMHC/Latest/MHC_nuc.fasta",
        filename="ipd_mhc_nuc.fasta",
        source="ipd_mhc",
        category="mhc_sequence",
        file_format="fasta",
        species="nhp",
    ),

    # =========================================================================
    # 10x Genomics - Public single-cell datasets
    # Note: Some URLs require authentication or change periodically
    # =========================================================================
    "10x_pbmc_10k_tcr": DatasetInfo(
        name="10x_pbmc_10k_tcr",
        description="10x Genomics 10k PBMC with TCR/BCR",
        url="https://cf.10xgenomics.com/samples/cell-vdj/5.0.0/sc5p_v2_hs_PBMC_10k/sc5p_v2_hs_PBMC_10k_t_filtered_contig_annotations.csv",
        filename="10x_pbmc_10k_tcr.csv",
        source="10x",
        category="tcr",
        file_format="csv",
        species="human",
    ),

    # =========================================================================
    # PIRD - Pan Immune Repertoire Database
    # Note: Requires manual download from https://db.cngb.org/pird/
    # =========================================================================
    "pird_tcr": DatasetInfo(
        name="pird_tcr",
        description="PIRD curated TCR sequences (requires manual download from web UI)",
        url="https://db.cngb.org/pird/tbadb/",
        filename="pird_tcr_human.csv",
        source="pird",
        category="tcr",
        file_format="csv",
        species="human",
    ),

    # =========================================================================
    # STCRDab - Structural TCR Database
    # =========================================================================
    "stcrdab": DatasetInfo(
        name="stcrdab",
        description="STCRDab structural TCR-pMHC data",
        url="https://opig.stats.ox.ac.uk/webapps/stcrdab-stcrpred/summary/all",
        filename="stcrdab_summary.dat",
        source="stcrdab",
        category="tcr",
        file_format="csv",
        species="human",
    ),
    # =========================================================================
    # UniProt SwissProt (curated protein sequences with taxonomy)
    # =========================================================================
    "uniprot_swissprot": DatasetInfo(
        name="uniprot_swissprot",
        description="UniProt/SwissProt reviewed protein sequences with taxonomy",
        url="https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz",
        filename="uniprot_sprot.fasta.gz",
        source="uniprot",
        category="protein",
        file_format="fasta",
        post_process="parse_uniprot_swissprot",
    ),
}

IEDB_EXTRACT_ALIASES: Dict[str, Dict[str, str]] = {
    "iedb_mhc_ligand": {"mhc_ligand_full.csv": "iedb_mhc_ligand_full.csv"},
    "iedb_tcell": {"tcell_full_v3.csv": "iedb_tcell_full_v3.csv"},
    "iedb_bcell": {"bcell_full_v3.csv": "iedb_bcell_full_v3.csv"},
    "iedb_cedar_mhc_ligand": {"mhc_ligand_full.csv": "cedar_mhc_ligand_full.csv"},
    "iedb_cedar_tcell": {"tcell_full_v3.csv": "cedar_tcell_full_v3.csv"},
    "iedb_cedar_bcell": {"bcell_full_v3.csv": "cedar_bcell_full_v3.csv"},
}


# =============================================================================
# Download State Management
# =============================================================================

@dataclass
class DownloadState:
    """Tracks download state for a dataset."""
    dataset: str
    status: str  # pending, downloading, completed, failed
    url: str
    local_path: Optional[str] = None
    size_bytes: Optional[int] = None
    md5: Optional[str] = None
    downloaded_at: Optional[str] = None
    error: Optional[str] = None
    processed: bool = False
    processed_path: Optional[str] = None


@dataclass
class DownloadManifest:
    """Manifest tracking all downloads in a data directory."""
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    data_dir: str = ""
    downloads: Dict[str, DownloadState] = field(default_factory=dict)

    def save(self, path: Path):
        """Save manifest to JSON file."""
        self.updated_at = datetime.now().isoformat()
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DownloadManifest":
        """Load manifest from JSON file."""
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        manifest = cls(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            data_dir=data.get("data_dir", ""),
        )
        for name, state_data in data.get("downloads", {}).items():
            manifest.downloads[name] = DownloadState(**state_data)
        return manifest


# =============================================================================
# Download Functions
# =============================================================================

def _get_remote_size(url: str, timeout: int = 10) -> Optional[int]:
    """Get file size from remote URL via HEAD request.

    Returns size in bytes, or None if unavailable.
    """
    import ssl

    try:
        ssl_context = ssl.create_default_context()
        request = urllib.request.Request(
            url,
            method='HEAD',
            headers={"User-Agent": "Presto-Downloader/1.0"}
        )
        response = urllib.request.urlopen(request, timeout=timeout, context=ssl_context)
        content_length = response.headers.get('content-length')
        if content_length:
            return int(content_length)
    except Exception:
        pass
    return None


def _format_size(size_bytes: Optional[int]) -> str:
    """Format size in human-readable format."""
    if size_bytes is None:
        return "unknown"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _compute_md5(path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def _download_with_progress(
    url: str,
    dest: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_redirects: int = 5,
) -> int:
    """Download a file with optional progress callback.

    Args:
        url: URL to download from
        dest: Destination path
        progress_callback: Called with (bytes_downloaded, total_bytes)
        max_redirects: Maximum number of redirects to follow

    Returns:
        Total bytes downloaded
    """
    import ssl

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Create SSL context that handles most cases
    ssl_context = ssl.create_default_context()

    current_url = url
    for _ in range(max_redirects):
        # Create request with user agent
        request = urllib.request.Request(
            current_url,
            headers={
                "User-Agent": "Presto-Downloader/1.0 (Python urllib)",
                'Accept': '*/*',
            }
        )

        try:
            response = urllib.request.urlopen(request, timeout=120, context=ssl_context)
            break
        except urllib.request.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                current_url = e.headers.get('Location', current_url)
                continue
            raise
    else:
        raise Exception(f"Too many redirects for {url}")

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    try:
        with open(dest, 'wb') as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    progress_callback(downloaded, total_size)
    finally:
        response.close()

    return downloaded


def _print_progress(downloaded: int, total: int, width: int = 50):
    """Print a progress bar."""
    if total > 0:
        pct = downloaded / total
        filled = int(width * pct)
        bar = '=' * filled + '-' * (width - filled)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        print(f"\r  [{bar}] {pct*100:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end='', flush=True)
    else:
        mb_down = downloaded / (1024 * 1024)
        print(f"\r  Downloaded: {mb_down:.1f} MB", end='', flush=True)


def _unzip_file(zip_path: Path, dest_dir: Path) -> List[Path]:
    """Extract a zip file and return list of extracted files."""
    extracted = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if not name.endswith('/'):
                zf.extract(name, dest_dir)
                extracted.append(dest_dir / name)
    return extracted


def _gunzip_file(gz_path: Path, dest_dir: Path) -> Path:
    """Extract a gzip file and return the extracted file path."""
    # Remove .gz extension for output filename
    out_name = gz_path.stem  # e.g., "file.csv.gz" -> "file.csv"
    out_path = dest_dir / out_name

    with gzip.open(gz_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    return out_path


def _materialize_iedb_extract_aliases(
    dataset_name: str,
    source_dir: Path,
    extracted: List[Path],
) -> List[Path]:
    """Create deterministic source-specific aliases for IEDB/CEDAR extracts."""
    alias_map = IEDB_EXTRACT_ALIASES.get(dataset_name)
    if not alias_map:
        return []

    extracted_by_name = {path.name.lower(): path for path in extracted}
    created: List[Path] = []
    for src_name, alias_name in alias_map.items():
        src_path = extracted_by_name.get(src_name.lower(), source_dir / src_name)
        if not src_path.exists():
            continue
        alias_path = source_dir / alias_name
        shutil.copy2(src_path, alias_path)
        created.append(alias_path)
    return created


def _looks_like_html_payload(path: Path, max_bytes: int = 4096) -> bool:
    """Detect common HTML payloads accidentally downloaded as data files."""
    if not path.exists() or path.is_dir():
        return False
    with open(path, "rb") as f:
        prefix = f.read(max_bytes)
    lowered = prefix.lstrip().lower()
    if lowered.startswith(b"<!doctype html"):
        return True
    if lowered.startswith(b"<html"):
        return True
    if b"<html" in lowered[:1024] and b"<body" in lowered[:1024]:
        return True
    return False


def parse_uniprot_swissprot(fasta_gz_path: Path, dest_dir: Path) -> Path:
    """Parse UniProt SwissProt gzipped FASTA into a TSV for downstream loading.

    Extracts accession, sequence, organism name (OS field), and taxonomy ID
    (OX field) from FASTA headers, then maps to 12-class organism categories.

    Writes ``proteins.tsv`` with columns: accession, sequence, category, organism.

    Args:
        fasta_gz_path: Path to ``uniprot_sprot.fasta.gz``.
        dest_dir: Directory to write ``proteins.tsv`` into.

    Returns:
        Path to the written TSV file.
    """
    from .vocab import normalize_organism

    out_path = dest_dir / "proteins.tsv"
    n_written = 0
    n_skipped = 0

    # Regex patterns for FASTA header fields
    _re_acc = re.compile(r"^>(?:sp|tr)\|([A-Za-z0-9_]+)\|")
    _re_os = re.compile(r"\bOS=(.+?)\s*(?:OX=|GN=|PE=|SV=|$)")
    _re_ox = re.compile(r"\bOX=(\d+)")

    with gzip.open(fasta_gz_path, "rt", encoding="utf-8") as fin, \
         open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(["accession", "sequence", "category", "organism"])

        accession = None
        organism_raw = None
        seq_parts: List[str] = []

        def _flush() -> None:
            nonlocal n_written, n_skipped, accession, organism_raw, seq_parts
            if accession is None:
                return
            seq = "".join(seq_parts).strip()
            cat = normalize_organism(organism_raw)
            if cat is not None and len(seq) >= 10:
                writer.writerow([accession, seq, cat, organism_raw or ""])
                n_written += 1
            else:
                n_skipped += 1
            accession = None
            organism_raw = None
            seq_parts = []

        for line in fin:
            if line.startswith(">"):
                _flush()
                m_acc = _re_acc.match(line)
                accession = m_acc.group(1) if m_acc else line[1:].split()[0]
                m_os = _re_os.search(line)
                organism_raw = m_os.group(1).strip() if m_os else None
                seq_parts = []
            else:
                seq_parts.append(line.strip())

        _flush()  # last record

    print(f"  uniprot_swissprot: Parsed {n_written} proteins "
          f"({n_skipped} skipped, no category or too short)")
    return out_path


def _validate_download_content(dataset_name: str, local_path: Path) -> Optional[str]:
    """Validate downloaded content for known source pitfalls."""
    if dataset_name == "stcrdab" and _looks_like_html_payload(local_path):
        return (
            "Downloaded content is HTML instead of STCRDab tabular data "
            "(e.g. starts with <!DOCTYPE html>). The source URL likely returned "
            "a webpage instead of the summary export."
        )
    return None


def download_dataset(
    dataset_name: str,
    data_dir: Path,
    force: bool = False,
    agree_terms: bool = False,
    verbose: bool = True,
) -> DownloadState:
    """Download a single dataset.

    Args:
        dataset_name: Name of dataset (key in DATASETS)
        data_dir: Directory to download to
        force: Re-download even if exists
        agree_terms: Agree to IEDB terms (required for IEDB data)
        verbose: Print progress

    Returns:
        DownloadState with result
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    info = DATASETS[dataset_name]
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check IEDB/CEDAR terms
    if info.requires_agreement and not agree_terms:
        return DownloadState(
            dataset=dataset_name,
            status="failed",
            url=info.url,
            error="IEDB/CEDAR data requires --agree-iedb-terms flag. "
                  "See https://www.iedb.org/terms_of_use.php"
        )

    # Determine paths
    source_dir = data_dir / info.source
    source_dir.mkdir(parents=True, exist_ok=True)
    dest_path = source_dir / info.filename

    # Check if already downloaded
    if dest_path.exists() and not force:
        if info.post_process == "unzip" and dataset_name in IEDB_EXTRACT_ALIASES:
            aliases = IEDB_EXTRACT_ALIASES[dataset_name]
            needs_aliases = any(not (source_dir / alias_name).exists() for alias_name in aliases.values())
            if needs_aliases:
                extracted = _unzip_file(dest_path, source_dir)
                _materialize_iedb_extract_aliases(dataset_name, source_dir, extracted)
        if verbose:
            print(f"  {dataset_name}: Already exists at {dest_path}")
        return DownloadState(
            dataset=dataset_name,
            status="completed",
            url=info.url,
            local_path=str(dest_path),
            size_bytes=dest_path.stat().st_size,
            md5=_compute_md5(dest_path),
            downloaded_at=datetime.fromtimestamp(dest_path.stat().st_mtime).isoformat(),
        )

    # Download
    if verbose:
        print(f"  {dataset_name}: Downloading from {info.source}...")

    try:
        progress_cb = _print_progress if verbose else None
        size = _download_with_progress(info.url, dest_path, progress_cb)
        if verbose:
            print()  # Newline after progress bar

        state = DownloadState(
            dataset=dataset_name,
            status="completed",
            url=info.url,
            local_path=str(dest_path),
            size_bytes=size,
            md5=_compute_md5(dest_path),
            downloaded_at=datetime.now().isoformat(),
        )

        # Post-process if needed
        if info.post_process == "unzip":
            if verbose:
                print(f"  {dataset_name}: Extracting zip...")
            extracted = _unzip_file(dest_path, source_dir)
            aliases = _materialize_iedb_extract_aliases(dataset_name, source_dir, extracted)
            state.processed = True
            state.processed_path = str(source_dir)
            if verbose:
                print(f"  {dataset_name}: Extracted {len(extracted)} files")
                if aliases:
                    print(f"  {dataset_name}: Wrote {len(aliases)} source aliases")
        elif info.post_process == "gunzip":
            if verbose:
                print(f"  {dataset_name}: Extracting gzip...")
            out_path = _gunzip_file(dest_path, source_dir)
            state.processed = True
            state.processed_path = str(out_path)
            if verbose:
                print(f"  {dataset_name}: Extracted to {out_path.name}")
        elif info.post_process == "parse_uniprot_swissprot":
            if verbose:
                print(f"  {dataset_name}: Parsing SwissProt FASTA...")
            out_path = parse_uniprot_swissprot(dest_path, source_dir)
            state.processed = True
            state.processed_path = str(out_path)

        # Validate downloaded payload for known bad-content cases.
        validate_path = Path(state.processed_path) if state.processed_path else dest_path
        validation_error = _validate_download_content(dataset_name, validate_path)
        if validation_error:
            return DownloadState(
                dataset=dataset_name,
                status="failed",
                url=info.url,
                local_path=str(validate_path),
                size_bytes=state.size_bytes,
                md5=state.md5,
                downloaded_at=state.downloaded_at,
                error=validation_error,
            )

        return state

    except Exception as e:
        return DownloadState(
            dataset=dataset_name,
            status="failed",
            url=info.url,
            error=str(e),
        )


def download_all(
    data_dir: Path,
    sources: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    force: bool = False,
    agree_terms: bool = False,
    verbose: bool = True,
) -> DownloadManifest:
    """Download multiple datasets.

    Args:
        data_dir: Directory to download to
        sources: Filter by source (iedb, vdjdb, mcpas, imgt)
        categories: Filter by category (binding, tcell, bcell, tcr, mhc_sequence)
        force: Re-download even if exists
        agree_terms: Agree to IEDB terms
        verbose: Print progress

    Returns:
        DownloadManifest with all results
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"

    # Load or create manifest
    manifest = DownloadManifest.load(manifest_path)
    manifest.data_dir = str(data_dir)

    # Filter datasets
    datasets_to_download = []
    for name, info in DATASETS.items():
        if sources and info.source not in sources:
            continue
        if categories and info.category not in categories:
            continue
        datasets_to_download.append(name)

    if verbose:
        print(f"Downloading {len(datasets_to_download)} datasets to {data_dir}")
        print()

    # Download each
    for name in datasets_to_download:
        state = download_dataset(
            name,
            data_dir,
            force=force,
            agree_terms=agree_terms,
            verbose=verbose,
        )
        manifest.downloads[name] = state

        if verbose and state.status == "failed":
            print(f"  WARNING: {name} failed: {state.error}", file=sys.stderr)

    # Save manifest
    manifest.save(manifest_path)

    if verbose:
        print()
        completed = sum(1 for s in manifest.downloads.values() if s.status == "completed")
        failed = sum(1 for s in manifest.downloads.values() if s.status == "failed")
        print(f"Download complete: {completed} succeeded, {failed} failed")
        print(f"Manifest saved to: {manifest_path}")

    return manifest


def list_datasets(
    sources: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> List[DatasetInfo]:
    """List available datasets with optional filtering.

    Args:
        sources: Filter by source
        categories: Filter by category

    Returns:
        List of DatasetInfo objects
    """
    result = []
    for name, info in DATASETS.items():
        if sources and info.source not in sources:
            continue
        if categories and info.category not in categories:
            continue
        result.append(info)
    return result


def list_local_datasets(data_dir: Path) -> DownloadManifest:
    """List datasets that have been downloaded locally.

    Args:
        data_dir: Data directory to check

    Returns:
        DownloadManifest with local dataset info
    """
    manifest_path = Path(data_dir) / "manifest.json"
    return DownloadManifest.load(manifest_path)


def get_dataset_path(dataset_name: str, data_dir: Path) -> Optional[Path]:
    """Get the local path for a downloaded dataset.

    Args:
        dataset_name: Name of dataset
        data_dir: Data directory

    Returns:
        Path to dataset file, or None if not downloaded
    """
    manifest = list_local_datasets(data_dir)
    if dataset_name in manifest.downloads:
        state = manifest.downloads[dataset_name]
        if state.status == "completed" and state.local_path:
            path = Path(state.local_path)
            if path.exists():
                return path
    return None


# =============================================================================
# Reference-Based Deduplication
# =============================================================================

@dataclass
class ReferenceInfo:
    """Information about a publication reference."""
    pubmed_id: Optional[str] = None
    doi: Optional[str] = None
    authors: Optional[str] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None

    def key(self) -> str:
        """Generate a unique key for this reference."""
        if self.pubmed_id:
            return f"pmid:{self.pubmed_id}"
        if self.doi:
            return f"doi:{self.doi}"
        # Fallback to title hash
        if self.title:
            return f"title:{hashlib.md5(self.title.lower().encode()).hexdigest()[:12]}"
        return "unknown"


@dataclass
class AssayRecord:
    """Generic assay record with reference information for deduplication."""
    peptide: str
    mhc_allele: str
    value: float
    value_type: str  # IC50, KD, EC50, response, etc.
    qualifier: int = 0  # -1='<', 0='=', 1='>'
    assay_type: Optional[str] = None
    mhc_class: str = "I"
    species: str = "human"
    source: str = ""
    reference: Optional[ReferenceInfo] = None
    # Quality indicators
    n_subjects: Optional[int] = None
    assay_quality: Optional[str] = None  # high, medium, low

    def dedup_key(self) -> str:
        """Generate key for deduplication (peptide + allele + value_type)."""
        return f"{self.peptide}|{self.mhc_allele}|{self.value_type}"


def _extract_reference_from_row(row: Dict[str, str], header_map: Dict[str, int]) -> ReferenceInfo:
    """Extract reference information from a data row."""
    ref = ReferenceInfo()

    # Try various column names for PubMed ID
    for col in ['pubmed_id', 'pubmed', 'pmid', 'reference id', 'pubmed id']:
        if col in row and row[col]:
            val = row[col].strip()
            # Extract numeric ID if it has prefix
            if val.isdigit():
                ref.pubmed_id = val
            else:
                match = re.search(r'\d+', val)
                if match:
                    ref.pubmed_id = match.group()
            break

    # Try various column names for DOI
    for col in ['doi', 'reference doi']:
        if col in row and row[col]:
            ref.doi = row[col].strip()
            break

    # Try to get title
    for col in ['title', 'reference title', 'article title']:
        if col in row and row[col]:
            ref.title = row[col].strip()
            break

    # Try to get year
    for col in ['year', 'publication year', 'pub year']:
        if col in row and row[col]:
            try:
                ref.year = int(row[col].strip()[:4])
            except ValueError:
                pass
            break

    return ref


def _parse_reference_columns(header: List[str]) -> Dict[str, int]:
    """Find reference-related columns in header."""
    ref_cols = {}
    header_lower = [h.lower().strip() for h in header]

    ref_patterns = [
        ('pubmed_id', ['pubmed_id', 'pubmed', 'pmid', 'reference id']),
        ('doi', ['doi', 'reference doi']),
        ('title', ['title', 'reference title', 'article title']),
        ('authors', ['authors', 'author']),
        ('year', ['year', 'publication year', 'pub year']),
    ]

    for key, patterns in ref_patterns:
        for i, h in enumerate(header_lower):
            for pattern in patterns:
                if pattern in h:
                    ref_cols[key] = i
                    break
            if key in ref_cols:
                break

    return ref_cols


class AssayDeduplicator:
    """Deduplicate assay records by reference and measurement.

    Strategy:
    1. Group records by (peptide, allele, value_type)
    2. Within each group, prefer records from:
       - More recent publications (newer methods)
       - Higher quality assays (defined assay type > undefined)
       - Exact measurements (qualifier=0) over bounds
    3. When same reference has multiple measurements, take median
    4. When different references, keep best quality per reference
    """

    def __init__(
        self,
        prefer_recent: bool = True,
        prefer_exact: bool = True,
        aggregate_same_ref: str = "median",  # median, mean, first, best
    ):
        self.prefer_recent = prefer_recent
        self.prefer_exact = prefer_exact
        self.aggregate_same_ref = aggregate_same_ref

        # Statistics
        self.stats = {
            'total_input': 0,
            'total_output': 0,
            'duplicates_removed': 0,
            'by_same_reference': 0,
            'by_different_reference': 0,
        }

    def deduplicate(
        self,
        records: List[AssayRecord],
    ) -> List[AssayRecord]:
        """Deduplicate a list of assay records.

        Args:
            records: List of AssayRecord objects

        Returns:
            Deduplicated list of AssayRecord objects
        """
        self.stats['total_input'] = len(records)

        # Group by deduplication key
        groups: Dict[str, List[AssayRecord]] = defaultdict(list)
        for rec in records:
            groups[rec.dedup_key()].append(rec)

        # Process each group
        deduped = []
        for key, group_records in groups.items():
            if len(group_records) == 1:
                deduped.append(group_records[0])
                continue

            # Further group by reference
            by_ref: Dict[str, List[AssayRecord]] = defaultdict(list)
            for rec in group_records:
                ref_key = rec.reference.key() if rec.reference else "unknown"
                by_ref[ref_key].append(rec)

            # Aggregate within each reference
            ref_representatives = []
            for ref_key, ref_records in by_ref.items():
                if len(ref_records) == 1:
                    ref_representatives.append(ref_records[0])
                else:
                    # Multiple measurements from same reference
                    self.stats['by_same_reference'] += len(ref_records) - 1
                    agg = self._aggregate_same_reference(ref_records)
                    ref_representatives.append(agg)

            # Select best among different references
            if len(ref_representatives) == 1:
                deduped.append(ref_representatives[0])
            else:
                self.stats['by_different_reference'] += len(ref_representatives) - 1
                best = self._select_best_reference(ref_representatives)
                deduped.append(best)

        self.stats['total_output'] = len(deduped)
        self.stats['duplicates_removed'] = self.stats['total_input'] - self.stats['total_output']

        return deduped

    def _aggregate_same_reference(self, records: List[AssayRecord]) -> AssayRecord:
        """Aggregate multiple measurements from the same reference."""
        if self.aggregate_same_ref == "first":
            return records[0]

        if self.aggregate_same_ref == "best":
            # Prefer exact measurements
            exact = [r for r in records if r.qualifier == 0]
            if exact:
                return exact[0]
            return records[0]

        # Compute median or mean
        values = [r.value for r in records]
        if self.aggregate_same_ref == "median":
            values.sort()
            mid = len(values) // 2
            if len(values) % 2 == 0:
                agg_value = (values[mid - 1] + values[mid]) / 2
            else:
                agg_value = values[mid]
        else:  # mean
            agg_value = sum(values) / len(values)

        # Create aggregated record (copy first record, update value)
        result = AssayRecord(
            peptide=records[0].peptide,
            mhc_allele=records[0].mhc_allele,
            value=agg_value,
            value_type=records[0].value_type,
            qualifier=0,  # Aggregated value is exact
            assay_type=records[0].assay_type,
            mhc_class=records[0].mhc_class,
            species=records[0].species,
            source=records[0].source,
            reference=records[0].reference,
            n_subjects=sum(r.n_subjects or 1 for r in records),
        )
        return result

    def _select_best_reference(self, records: List[AssayRecord]) -> AssayRecord:
        """Select best record among different references."""
        def score(rec: AssayRecord) -> Tuple[int, int, int]:
            """Higher score = better."""
            # Year score (prefer recent)
            year_score = 0
            if self.prefer_recent and rec.reference and rec.reference.year:
                year_score = rec.reference.year - 1990  # Normalize to ~0-35

            # Qualifier score (prefer exact)
            qual_score = 0 if self.prefer_exact else 1
            if rec.qualifier == 0:
                qual_score = 2
            elif rec.qualifier != 0:
                qual_score = 1

            # Assay quality score
            assay_score = 1 if rec.assay_type else 0

            return (qual_score, year_score, assay_score)

        return max(records, key=score)

    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        return self.stats.copy()


def deduplicate_binding_file(
    input_path: Path,
    output_path: Path,
    verbose: bool = True,
) -> Dict[str, int]:
    """Deduplicate a binding data file by reference.

    Args:
        input_path: Path to input CSV/TSV file
        output_path: Path to write deduplicated output
        verbose: Print progress

    Returns:
        Statistics dict
    """
    # Read input
    records = []

    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        first_line = f.readline()
        f.seek(0)
        delimiter = '\t' if '\t' in first_line else ','
        reader = csv.DictReader(f, delimiter=delimiter)

        # Find relevant columns
        header = reader.fieldnames or []
        header_lower = {h.lower(): h for h in header}

        def get_col(row: Dict, candidates: List[str]) -> str:
            for c in candidates:
                c_lower = c.lower()
                for h_lower, h_orig in header_lower.items():
                    if c_lower in h_lower:
                        val = row.get(h_orig, '')
                        if val:
                            return val.strip()
            return ''

        for row in reader:
            peptide = get_col(row, ['peptide', 'epitope', 'linear peptide'])
            if not peptide:
                continue

            allele = get_col(row, ['allele', 'mhc allele'])
            value_str = get_col(row, ['value', 'measurement value', 'ic50', 'kd'])

            try:
                value = float(value_str.lstrip('<>='))
            except (ValueError, TypeError):
                continue

            qualifier = 0
            if value_str.startswith('<'):
                qualifier = -1
            elif value_str.startswith('>'):
                qualifier = 1

            ref = _extract_reference_from_row(row, {})

            records.append(AssayRecord(
                peptide=peptide,
                mhc_allele=allele,
                value=value,
                value_type=get_col(row, ['measurement type', 'type']) or 'IC50',
                qualifier=qualifier,
                assay_type=get_col(row, ['assay type', 'assay']),
                reference=ref,
            ))

    if verbose:
        print(f"Read {len(records)} records from {input_path}")

    # Deduplicate
    deduplicator = AssayDeduplicator()
    deduped = deduplicator.deduplicate(records)

    if verbose:
        stats = deduplicator.get_stats()
        print(f"Deduplicated to {stats['total_output']} records")
        print(f"  - Removed {stats['by_same_reference']} duplicates from same reference")
        print(f"  - Removed {stats['by_different_reference']} duplicates from different references")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            'peptide', 'mhc_allele', 'value', 'value_type', 'qualifier',
            'assay_type', 'mhc_class', 'species', 'pubmed_id', 'doi'
        ])
        for rec in deduped:
            qual_str = {-1: '<', 0: '=', 1: '>'}.get(rec.qualifier, '=')
            pmid = rec.reference.pubmed_id if rec.reference else ''
            doi = rec.reference.doi if rec.reference else ''
            writer.writerow([
                rec.peptide, rec.mhc_allele, rec.value, rec.value_type,
                qual_str, rec.assay_type or '', rec.mhc_class, rec.species,
                pmid, doi
            ])

    if verbose:
        print(f"Wrote deduplicated data to {output_path}")

    return deduplicator.get_stats()


def deduplicate_tcell_file(
    input_path: Path,
    output_path: Path,
    verbose: bool = True,
) -> Dict[str, int]:
    """Deduplicate a T-cell assay file by reference.

    For T-cell data, we deduplicate by (peptide, allele, response_type).
    Multiple positive/negative responses from same reference are aggregated
    by majority vote.

    Args:
        input_path: Path to input CSV/TSV file
        output_path: Path to write deduplicated output
        verbose: Print progress

    Returns:
        Statistics dict
    """
    records = []

    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        first_line = f.readline()
        f.seek(0)
        delimiter = '\t' if '\t' in first_line else ','
        reader = csv.DictReader(f, delimiter=delimiter)

        header = reader.fieldnames or []
        header_lower = {h.lower(): h for h in header}

        def get_col(row: Dict, candidates: List[str]) -> str:
            for c in candidates:
                c_lower = c.lower()
                for h_lower, h_orig in header_lower.items():
                    if c_lower in h_lower:
                        val = row.get(h_orig, '')
                        if val:
                            return val.strip()
            return ''

        for row in reader:
            peptide = get_col(row, ['peptide', 'epitope', 'linear peptide'])
            if not peptide:
                continue

            allele = get_col(row, ['allele', 'mhc allele'])
            response_str = get_col(row, ['response', 'outcome', 'qualitative'])

            # Parse response
            response_lower = response_str.lower()
            if response_lower in ('positive', 'pos', '1', 'true', 'yes'):
                response = 1.0
            elif response_lower in ('negative', 'neg', '0', 'false', 'no'):
                response = 0.0
            else:
                continue

            ref = _extract_reference_from_row(row, {})

            records.append(AssayRecord(
                peptide=peptide,
                mhc_allele=allele,
                value=response,
                value_type='tcell_response',
                assay_type=get_col(row, ['assay type', 'assay']),
                reference=ref,
            ))

    if verbose:
        print(f"Read {len(records)} T-cell records from {input_path}")

    # Deduplicate with majority vote for responses
    deduplicator = AssayDeduplicator(aggregate_same_ref="mean")
    deduped_raw = deduplicator.deduplicate(records)

    # Convert aggregated means back to 0/1 (majority vote)
    deduped = []
    for rec in deduped_raw:
        rec.value = 1.0 if rec.value >= 0.5 else 0.0
        deduped.append(rec)

    if verbose:
        stats = deduplicator.get_stats()
        print(f"Deduplicated to {stats['total_output']} records")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            'peptide', 'mhc_allele', 'response', 'assay_type',
            'mhc_class', 'species', 'pubmed_id', 'doi'
        ])
        for rec in deduped:
            pmid = rec.reference.pubmed_id if rec.reference else ''
            doi = rec.reference.doi if rec.reference else ''
            writer.writerow([
                rec.peptide, rec.mhc_allele, int(rec.value),
                rec.assay_type or '', rec.mhc_class, rec.species,
                pmid, doi
            ])

    if verbose:
        print(f"Wrote deduplicated T-cell data to {output_path}")

    return deduplicator.get_stats()


# =============================================================================
# Utility Functions
# =============================================================================

def get_all_sources() -> List[str]:
    """Get list of all available data sources."""
    return sorted(set(info.source for info in DATASETS.values()))


def get_all_categories() -> List[str]:
    """Get list of all available data categories."""
    return sorted(set(info.category for info in DATASETS.values()))


def get_datasets_by_source(source: str) -> List[DatasetInfo]:
    """Get all datasets from a specific source."""
    return [info for info in DATASETS.values() if info.source == source]


def get_datasets_by_category(category: str) -> List[DatasetInfo]:
    """Get all datasets of a specific category."""
    return [info for info in DATASETS.values() if info.category == category]
