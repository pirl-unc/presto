"""Data management CLI commands.

Handles downloading, listing, and processing of immunology datasets.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Dict

from ..data.downloaders import (
    DATASETS,
    DatasetInfo,
    DownloadManifest,
    download_all,
    download_dataset,
    list_datasets,
    list_local_datasets,
    get_dataset_path,
    deduplicate_binding_file,
    deduplicate_tcell_file,
    _get_remote_size,
    _format_size,
)
from ..data.cross_source_dedup import deduplicate_all
from ..data.mhc_index import (
    MHCIndexError,
    augment_mhc_index,
    build_mhc_index,
    resolve_alleles,
    summarize_mhc_index,
    validate_mhc_index,
)
from ..data.mouse_mhc_overlay import (
    IMGT_MOUSE_MHC_NOMENCLATURE_URL,
    build_mouse_mhc_overlay,
)




def cmd_data_download(args: Any) -> int:
    """Handle 'presto data download' command."""
    outdir = Path(args.outdir)
    verbose = not args.quiet

    # Determine what to download
    if args.all:
        # Download everything
        if verbose:
            print("Downloading all available datasets...")
        manifest = download_all(
            outdir,
            force=args.force,
            agree_terms=args.agree_iedb_terms,
            verbose=verbose,
        )
    elif args.dataset:
        # Download specific datasets
        if verbose:
            print(f"Downloading {len(args.dataset)} specified dataset(s)...")
        manifest = DownloadManifest(data_dir=str(outdir))
        for name in args.dataset:
            if name not in DATASETS:
                print(f"Unknown dataset: {name}", file=sys.stderr)
                print(f"Available: {', '.join(DATASETS.keys())}", file=sys.stderr)
                return 1
            state = download_dataset(
                name,
                outdir,
                force=args.force,
                agree_terms=args.agree_iedb_terms,
                verbose=verbose,
            )
            manifest.downloads[name] = state
        manifest.save(outdir / "manifest.json")
    elif args.source or args.category:
        # Download by source or category
        manifest = download_all(
            outdir,
            sources=args.source,
            categories=args.category,
            force=args.force,
            agree_terms=args.agree_iedb_terms,
            verbose=verbose,
        )
    else:
        # No selection - print help
        print("Please specify what to download:", file=sys.stderr)
        print("  --all              Download all datasets", file=sys.stderr)
        print("  --dataset NAME     Download specific dataset(s)", file=sys.stderr)
        print("  --source SOURCE    Download by source (iedb, vdjdb, mcpas, imgt)", file=sys.stderr)
        print("  --category CAT     Download by category (binding, tcell, tcr, ...)", file=sys.stderr)
        print("\nUse 'presto data list' to see available datasets.", file=sys.stderr)
        return 1

    # Check for failures
    failed = [name for name, state in manifest.downloads.items() if state.status == "failed"]
    if failed:
        return 1
    return 0


def cmd_data_list(args: Any) -> int:
    """Handle 'presto data list' command."""
    if args.local:
        # List local datasets
        datadir = Path(args.datadir)
        if not datadir.exists():
            if args.json:
                print(json.dumps({"error": f"Data directory not found: {datadir}"}))
            else:
                print(f"Data directory not found: {datadir}", file=sys.stderr)
            return 1

        manifest = list_local_datasets(datadir)

        if args.json:
            output = {
                "data_dir": str(datadir),
                "datasets": {}
            }
            for name, state in manifest.downloads.items():
                info = DATASETS.get(name)
                if args.source and info and info.source not in args.source:
                    continue
                if args.category and info and info.category not in args.category:
                    continue
                output["datasets"][name] = {
                    "status": state.status,
                    "local_path": state.local_path,
                    "size_bytes": state.size_bytes,
                    "downloaded_at": state.downloaded_at,
                    "source": info.source if info else "unknown",
                    "category": info.category if info else "unknown",
                }
            print(json.dumps(output, indent=2))
        else:
            print(f"Local datasets in {datadir}:\n")
            if not manifest.downloads:
                print("  No datasets downloaded yet.")
                print("  Use 'presto data download' to download datasets.")
                return 0

            for name, state in sorted(manifest.downloads.items()):
                info = DATASETS.get(name)
                if args.source and info and info.source not in args.source:
                    continue
                if args.category and info and info.category not in args.category:
                    continue

                status_icon = "+" if state.status == "completed" else "x"
                size = _format_size(state.size_bytes) if state.size_bytes else "unknown"
                source = info.source if info else "unknown"
                print(f"  [{status_icon}] {name}")
                print(f"      Source: {source}, Size: {size}")
                if state.local_path:
                    print(f"      Path: {state.local_path}")
                if state.status == "failed" and state.error:
                    print(f"      Error: {state.error}")
                print()

    else:
        # List available datasets
        datasets = list_datasets(sources=args.source, categories=args.category)

        if args.json:
            output = {
                "available_datasets": [
                    {
                        "name": d.name,
                        "description": d.description,
                        "source": d.source,
                        "category": d.category,
                        "format": d.file_format,
                        "requires_agreement": d.requires_agreement,
                    }
                    for d in datasets
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            print("Available datasets:\n")

            # Group by source
            by_source = {}
            for d in datasets:
                by_source.setdefault(d.source, []).append(d)

            for source, source_datasets in sorted(by_source.items()):
                print(f"  {source.upper()}:")
                for d in source_datasets:
                    agreement = " [requires --agree-iedb-terms]" if d.requires_agreement else ""
                    print(f"    {d.name}")
                    print(f"      {d.description}")
                    print(f"      Category: {d.category}, Format: {d.file_format}{agreement}")
                    print()

            print("Use 'presto data download --dataset NAME' to download specific datasets")
            print("Use 'presto data download --all --agree-iedb-terms' to download everything")

    return 0


def cmd_data_info(args: Any) -> int:
    """Handle 'presto data info' command."""
    name = args.dataset
    datadir = Path(args.datadir)

    if name not in DATASETS:
        print(f"Unknown dataset: {name}", file=sys.stderr)
        print(f"Available: {', '.join(DATASETS.keys())}", file=sys.stderr)
        return 1

    info = DATASETS[name]

    print(f"Dataset: {info.name}")
    print(f"Description: {info.description}")
    print(f"Source: {info.source}")
    print(f"Category: {info.category}")
    print(f"Format: {info.file_format}")
    # Fetch size dynamically
    remote_size = _get_remote_size(info.url)
    print(f"Remote Size: {_format_size(remote_size)}")
    print(f"URL: {info.url}")
    if info.requires_agreement:
        print(f"Requires Agreement: Yes (use --agree-iedb-terms)")
    print()

    # Check if downloaded locally
    local_path = get_dataset_path(name, datadir)
    if local_path:
        print(f"Local Status: Downloaded")
        print(f"Local Path: {local_path}")
        print(f"Local Size: {_format_size(local_path.stat().st_size)}")
    else:
        print(f"Local Status: Not downloaded")
        print(f"Use 'presto data download --dataset {name}' to download")

    return 0


def cmd_data_process(args: Any) -> int:
    """Handle 'presto data process' command."""
    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    verbose = not args.quiet

    if not datadir.exists():
        print(f"Data directory not found: {datadir}", file=sys.stderr)
        print("Use 'presto data download' to download datasets first.", file=sys.stderr)
        return 1

    # Check what's available
    manifest = list_local_datasets(datadir)
    if not manifest.downloads:
        print("No datasets downloaded yet.", file=sys.stderr)
        print("Use 'presto data download' to download datasets first.", file=sys.stderr)
        return 1

    outdir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to process
    datasets_to_process = args.dataset if args.dataset else list(manifest.downloads.keys())

    if verbose:
        print(f"Processing {len(datasets_to_process)} dataset(s)...")
        print(f"  Species filter: {args.species}")
        print(f"  MHC class filter: {args.mhc_class}")
        print(f"  Peptide length: {args.min_peptide_length}-{args.max_peptide_length}")
        print()

    processed_counts = {}

    for name in datasets_to_process:
        if name not in manifest.downloads:
            if verbose:
                print(f"  {name}: Not downloaded, skipping")
            continue

        state = manifest.downloads[name]
        if state.status != "completed":
            if verbose:
                print(f"  {name}: Download incomplete, skipping")
            continue

        local_path = Path(state.processed_path or state.local_path)
        if not local_path.exists():
            if verbose:
                print(f"  {name}: File not found at {local_path}, skipping")
            continue

        if verbose:
            print(f"  Processing {name}...")

        info = DATASETS.get(name)
        if not info:
            continue

        # Process based on source/category
        count = _process_dataset(
            name=name,
            info=info,
            input_path=local_path,
            output_dir=outdir,
            species=args.species,
            mhc_class=args.mhc_class,
            min_pep_len=args.min_peptide_length,
            max_pep_len=args.max_peptide_length,
            deduplicate=args.deduplicate,
            verbose=verbose,
        )
        processed_counts[name] = count

    if verbose:
        print()
        print("Processing complete:")
        for name, count in processed_counts.items():
            print(f"  {name}: {count} records")
        print(f"\nOutput directory: {outdir}")

    # Save processing manifest
    process_manifest = {
        "source_dir": str(datadir),
        "output_dir": str(outdir),
        "filters": {
            "species": args.species,
            "mhc_class": args.mhc_class,
            "min_peptide_length": args.min_peptide_length,
            "max_peptide_length": args.max_peptide_length,
            "deduplicate": args.deduplicate,
        },
        "datasets": processed_counts,
    }
    with open(outdir / "process_manifest.json", 'w') as f:
        json.dump(process_manifest, f, indent=2)

    return 0


def _process_dataset(
    name: str,
    info: DatasetInfo,
    input_path: Path,
    output_dir: Path,
    species: str,
    mhc_class: str,
    min_pep_len: int,
    max_pep_len: int,
    deduplicate: bool,
    verbose: bool,
) -> int:
    """Process a single dataset with filtering.

    Returns the number of records written.
    """
    import csv

    # Determine input file (may be in subdirectory for zips)
    if input_path.is_dir():
        # Find CSV/TSV files in directory
        csv_files = list(input_path.glob("*.csv")) + list(input_path.glob("*.tsv")) + list(input_path.glob("*.txt"))
        if not csv_files:
            return 0
        # Use the largest file (likely the main data file)
        input_file = max(csv_files, key=lambda p: p.stat().st_size)
    else:
        input_file = input_path

    # Determine output file
    output_file = output_dir / f"{name}_processed.tsv"

    # Read and filter
    records = []
    seen = set() if deduplicate else None

    try:
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            # Detect delimiter
            first_line = f.readline()
            f.seek(0)
            delimiter = '\t' if '\t' in first_line else ','

            reader = csv.DictReader(f, delimiter=delimiter)

            for row in reader:
                # Extract peptide (try multiple column names)
                peptide = None
                for col in ['peptide', 'Epitope', 'epitope', 'Description', 'description',
                            'Linear Sequence', 'linear sequence', 'antigen.epitope', 'cdr3']:
                    if col in row and row[col]:
                        peptide = row[col].strip()
                        break

                if not peptide:
                    continue

                # Filter by peptide length
                if not (min_pep_len <= len(peptide) <= max_pep_len):
                    continue

                # Filter by species
                if species != "all":
                    row_species = None
                    for col in ['species', 'Species', 'host', 'Host', 'Organism', 'organism']:
                        if col in row and row[col]:
                            row_species = row[col].lower()
                            break
                    if row_species and species.lower() not in row_species:
                        continue

                # Filter by MHC class
                if mhc_class != "all":
                    row_class = None
                    for col in ['mhc_class', 'MHC Class', 'mhc.class', 'Class']:
                        if col in row and row[col]:
                            row_class = row[col].strip()
                            break
                    if row_class and row_class != mhc_class:
                        # Also check for "class I" vs "I" format
                        if mhc_class not in row_class:
                            continue

                # Deduplicate if requested
                if deduplicate:
                    # Create key from peptide + allele
                    allele = None
                    for col in ['allele', 'Allele', 'mhc_allele', 'MHC Allele', 'mhc.a']:
                        if col in row and row[col]:
                            allele = row[col].strip()
                            break
                    key = (peptide, allele or "")
                    if key in seen:
                        continue
                    seen.add(key)

                records.append(row)

    except Exception as e:
        if verbose:
            print(f"    Error reading {input_file}: {e}")
        return 0

    # Write filtered output
    if records:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys(), delimiter='\t')
            writer.writeheader()
            writer.writerows(records)

    return len(records)


def cmd_data_dedup(args: Any) -> int:
    """Handle 'presto data dedup' command."""
    input_path = Path(args.input)
    verbose = not args.quiet

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        if stem.endswith('.csv'):
            stem = stem[:-4]
        output_path = input_path.parent / f"{stem}_deduped.tsv"

    if verbose:
        print(f"Deduplicating {args.type} data...")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Aggregation: {args.aggregate}")
        print()

    try:
        if args.type == "binding":
            stats = deduplicate_binding_file(input_path, output_path, verbose=verbose)
        else:  # tcell
            stats = deduplicate_tcell_file(input_path, output_path, verbose=verbose)

        if verbose:
            print()
            print("Deduplication complete:")
            print(f"  Input records: {stats['total_input']}")
            print(f"  Output records: {stats['total_output']}")
            print(f"  Duplicates removed: {stats['duplicates_removed']}")
            if stats['by_same_reference'] > 0:
                print(f"    - From same reference: {stats['by_same_reference']}")
            if stats['by_different_reference'] > 0:
                print(f"    - From different references: {stats['by_different_reference']}")

        return 0

    except Exception as e:
        print(f"Error during deduplication: {e}", file=sys.stderr)
        return 1


def cmd_data_merge(args: Any) -> int:
    """Handle 'presto data merge' command for cross-source deduplication."""
    datadir = Path(args.datadir)
    verbose = not args.quiet

    if not datadir.exists():
        print(f"Data directory not found: {datadir}", file=sys.stderr)
        print("Use 'presto data download' to download datasets first.", file=sys.stderr)
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = datadir / "merged_deduped.tsv"
    assay_outdir = None
    if args.per_assay_csv:
        assay_outdir = (
            Path(args.assay_outdir)
            if args.assay_outdir
            else datadir / "merged_assays"
        )

    # Parse record types
    record_types = args.types if args.types else None

    try:
        records, stats = deduplicate_all(
            data_dir=datadir,
            output_path=output_path,
            assay_output_dir=assay_outdir,
            record_types=record_types,
            verbose=verbose,
        )

        if args.json:
            import json
            print(json.dumps(stats, indent=2))
        elif verbose and assay_outdir is not None:
            print(f"Per-assay CSVs written to: {assay_outdir}")

        return 0

    except Exception as e:
        print(f"Error during merge: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def _sniff_delimiter(sample: str) -> Optional[str]:
    if "\t" in sample:
        return "\t"
    if "," in sample:
        return ","
    return None


def _read_alleles_from_file(path: Path, column: Optional[str]) -> List[str]:
    text = path.read_text(encoding="utf-8").splitlines()
    if not text:
        return []

    delimiter = _sniff_delimiter(text[0])
    if delimiter:
        reader = csv.DictReader(text, delimiter=delimiter)
        column_name = column or "allele"
        if not reader.fieldnames or column_name not in reader.fieldnames:
            raise MHCIndexError(
                f"Column '{column_name}' not found in {path}. "
                f"Available columns: {reader.fieldnames}"
            )
        return [
            row[column_name].strip()
            for row in reader
            if row.get(column_name) and row[column_name].strip()
        ]

    values = [line.strip() for line in text if line.strip()]
    column_name = (column or "allele").strip().lower()
    if values and values[0].strip().lower() == column_name:
        return values[1:]
    return values


def _write_resolve_output(
    results: List[Dict[str, object]],
    output_path: Optional[Path],
    fmt: str,
) -> None:
    if fmt == "json":
        payload = json.dumps(results, indent=2)
        if output_path:
            output_path.write_text(payload)
        else:
            print(payload)
        return

    # CSV output
    ordered_fields = [
        "input",
        "normalized",
        "resolved",
        "found",
        "gene",
        "mhc_class",
        "species",
        "source",
        "seq_len",
        "sequence",
        "error",
    ]
    present = {k for row in results for k in row.keys()}
    fieldnames = [k for k in ordered_fields if k in present]
    for k in sorted(present):
        if k not in fieldnames:
            fieldnames.append(k)

    output_stream = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_stream = output_path.open("w", newline="", encoding="utf-8")
    try:
        writer = csv.DictWriter(output_stream or sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    finally:
        if output_stream:
            output_stream.close()


def cmd_data_mhc_index_build(args: Any) -> int:
    """Handle 'presto data mhc-index build' command."""
    try:
        stats = build_mhc_index(
            imgt_fasta=args.imgt_fasta,
            ipd_mhc_dir=args.ipd_mhc_dir,
            out_csv=args.out_csv,
            out_fasta=args.out_fasta,
        )
    except MHCIndexError as exc:
        print(f"Error building MHC index: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("MHC index built:")
        print(f"  total_records: {stats['total']}")
        print(f"  parsed:        {stats['parsed']}")
        print(f"  skipped:       {stats['skipped']}")
        print(f"  duplicates:    {stats['duplicates']}")
        print(f"  replaced:      {stats['replaced']}")
        print(f"  out_csv:       {args.out_csv}")
        if args.out_fasta:
            print(f"  out_fasta:     {args.out_fasta}")
    return 0


def cmd_data_mhc_index_augment(args: Any) -> int:
    """Handle 'presto data mhc-index augment' command."""
    try:
        stats = augment_mhc_index(
            index_csv=args.index_csv,
            output_csv=args.out_csv,
        )
    except MHCIndexError as exc:
        print(f"Error augmenting MHC index: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("MHC index augmented:")
        print(f"  index_csv:        {args.index_csv}")
        print(f"  out_csv:          {args.out_csv}")
        print(f"  total_records:    {stats['total_records']}")
        print(f"  functional_true:  {stats['functional_true']}")
        print(f"  functional_false: {stats['functional_false']}")
        groove_counts = stats.get("by_groove_status", {})
        if isinstance(groove_counts, dict) and groove_counts:
            print("  groove_status:")
            for key, value in groove_counts.items():
                print(f"    {key}: {value}")
    return 0


def _path_has_fasta_payload(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix in {".fa", ".fasta", ".faa", ".gz", ".zip"}
    return any(
        child.suffix in {".fa", ".fasta", ".faa", ".gz", ".zip"}
        for child in path.rglob("*")
        if child.is_file()
    )


def _count_index_prefix(index_csv: str, prefix: str) -> int:
    target = prefix.upper()
    count = 0
    path = Path(index_csv)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            token = (row.get("normalized") or "").strip().upper()
            if token.startswith(target):
                count += 1
    return count


def _resolve_index_input_paths(args: Any, datadir: Path) -> tuple[Optional[str], Optional[str]]:
    imgt_fasta = args.imgt_fasta
    ipd_mhc_dir = args.ipd_mhc_dir

    if not imgt_fasta:
        path = get_dataset_path("imgt_hla", datadir)
        if path:
            imgt_fasta = str(path)
        else:
            fallback = datadir / "imgt" / "hla_prot.fasta"
            if fallback.exists():
                imgt_fasta = str(fallback)

    if not ipd_mhc_dir:
        fallback_dir = datadir / "ipd_mhc"
        ipd_dataset_paths: List[Path] = []
        for dataset_name in sorted(name for name in DATASETS if name.startswith("ipd_mhc")):
            path = get_dataset_path(dataset_name, datadir)
            if path:
                ipd_dataset_paths.append(path)
        if _path_has_fasta_payload(fallback_dir):
            # Prefer directory root so any additional FASTA overlays (e.g., mouse) are included.
            ipd_mhc_dir = str(fallback_dir)
        elif ipd_dataset_paths:
            ipd_mhc_dir = str(ipd_dataset_paths[0])
        else:
            fallback_file = datadir / "ipd_mhc" / "ipd_mhc_prot.fasta"
            if _path_has_fasta_payload(fallback_file):
                ipd_mhc_dir = str(fallback_file)
            elif fallback_dir.exists():
                ipd_mhc_dir = str(fallback_dir)

    if args.download_missing:
        if not imgt_fasta:
            state = download_dataset(
                "imgt_hla",
                datadir,
                force=False,
                agree_terms=False,
                verbose=not args.quiet,
            )
            if state.status == "completed" and state.local_path:
                imgt_fasta = state.local_path
        if not ipd_mhc_dir:
            state = download_dataset(
                "ipd_mhc_nhp",
                datadir,
                force=False,
                agree_terms=False,
                verbose=not args.quiet,
            )
            if state.status == "completed" and state.local_path:
                fallback_dir = datadir / "ipd_mhc"
                if _path_has_fasta_payload(fallback_dir):
                    ipd_mhc_dir = str(fallback_dir)
                else:
                    ipd_mhc_dir = state.local_path

    return imgt_fasta, ipd_mhc_dir


def cmd_data_mhc_index_refresh(args: Any) -> int:
    """Handle 'presto data mhc-index refresh' command."""
    datadir = Path(args.datadir)
    out_csv = args.out_csv or str(datadir / "mhc_index.csv")
    out_fasta = args.out_fasta

    imgt_fasta, ipd_mhc_dir = _resolve_index_input_paths(args, datadir)
    if not imgt_fasta and not ipd_mhc_dir:
        print(
            "Unable to locate IMGT or IPD-MHC inputs. "
            "Provide --imgt-fasta/--ipd-mhc-dir or use --download-missing.",
            file=sys.stderr,
        )
        return 1

    try:
        stats = build_mhc_index(
            imgt_fasta=imgt_fasta,
            ipd_mhc_dir=ipd_mhc_dir,
            out_csv=out_csv,
            out_fasta=out_fasta,
        )
    except MHCIndexError as exc:
        print(f"Error refreshing MHC index: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("MHC index refreshed:")
        print(f"  imgt_fasta: {imgt_fasta or 'N/A'}")
        print(f"  ipd_mhc:    {ipd_mhc_dir or 'N/A'}")
        print(f"  out_csv:    {out_csv}")
        if out_fasta:
            print(f"  out_fasta:  {out_fasta}")
        print(f"  parsed:     {stats['parsed']}")
        print(f"  skipped:    {stats['skipped']}")
        h2_count = _count_index_prefix(out_csv, "H2-")
        print(f"  mouse_h2:   {h2_count}")
        if h2_count == 0:
            print(
                "  WARNING: no H2-* alleles in index; add mouse FASTA files under "
                f"{datadir / 'ipd_mhc'} to include murine MHC sequences."
            )

    return 0


def cmd_data_mhc_index_mouse_overlay(args: Any) -> int:
    """Build a mouse MHC overlay from IMGT gene names + UniProt proteins."""
    datadir = Path(args.datadir)
    out_csv = args.out_csv or str(datadir / "ipd_mhc" / "mouse_uniprot_overlay.csv")
    out_fasta = args.out_fasta or str(datadir / "ipd_mhc" / "mouse_uniprot_overlay.fasta")
    imgt_url = args.imgt_url or IMGT_MOUSE_MHC_NOMENCLATURE_URL

    try:
        stats = build_mouse_mhc_overlay(
            out_csv=out_csv,
            out_fasta=out_fasta,
            imgt_url=imgt_url,
            reviewed_only=not bool(args.include_unreviewed),
            max_genes=int(args.max_genes or 0),
        )
    except Exception as exc:
        print(f"Error building mouse MHC overlay: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("Mouse MHC overlay built:")
        print(f"  out_csv:         {out_csv}")
        print(f"  out_fasta:       {out_fasta}")
        print(f"  imgt_url:        {imgt_url}")
        print(f"  imgt_genes:      {stats['imgt_genes']}")
        print(f"  catalog_rows:    {stats['catalog_rows']}")
        print(f"  selected_alleles:{stats['selected_alleles']}")
        print(
            "  NOTE: refresh the MHC index after this "
            "(presto data mhc-index refresh) to include overlay sequences."
        )
    return 0


def cmd_data_mhc_index_report(args: Any) -> int:
    """Handle 'presto data mhc-index report' command."""
    try:
        report = summarize_mhc_index(args.index_csv)
    except MHCIndexError as exc:
        print(f"Error summarizing MHC index: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else None
    if args.format == "json":
        payload = json.dumps(report, indent=2)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload)
        else:
            print(payload)
        return 0

    # CSV output for downstream scripting
    rows = [("metric", "key", "count")]
    rows.append(("total_records", "", report["total_records"]))
    for metric in ("by_source", "by_species", "by_mhc_class", "by_gene"):
        counts = report.get(metric, {})
        for key, value in counts.items():
            rows.append((metric, key, value))

    out = output_path.open("w", newline="", encoding="utf-8") if output_path else sys.stdout
    try:
        writer = csv.writer(out)
        writer.writerows(rows)
    finally:
        if output_path:
            out.close()
    return 0


def cmd_data_mhc_index_validate(args: Any) -> int:
    """Handle 'presto data mhc-index validate' command."""
    try:
        report = validate_mhc_index(args.index_csv)
    except MHCIndexError as exc:
        print(f"Error validating MHC index: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else None
    if args.format == "json":
        payload = json.dumps(report, indent=2)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload)
        else:
            print(payload)
    else:
        rows = [("metric", "value")]
        rows.append(("valid", report["valid"]))
        rows.append(("total_rows", report["total_rows"]))
        rows.append(("error_count", report["error_count"]))
        rows.append(("warning_count", report["warning_count"]))
        rows.append(("errors", json.dumps(report.get("errors", []))))
        rows.append(("warnings", json.dumps(report.get("warnings", []))))
        out = output_path.open("w", newline="", encoding="utf-8") if output_path else sys.stdout
        try:
            writer = csv.writer(out)
            writer.writerows(rows)
        finally:
            if output_path:
                out.close()

    return 0 if report.get("valid", False) else 1


def cmd_data_mhc_index_resolve(args: Any) -> int:
    """Handle 'presto data mhc-index resolve' command."""
    alleles: List[str] = []
    if args.alleles:
        alleles.extend([a.strip() for a in args.alleles.split(",") if a.strip()])
    if args.allele_file:
        try:
            alleles.extend(
                _read_alleles_from_file(Path(args.allele_file), args.column)
            )
        except MHCIndexError as exc:
            print(f"Error reading alleles: {exc}", file=sys.stderr)
            return 1

    if not alleles:
        print("No alleles provided. Use --alleles or --allele-file.", file=sys.stderr)
        return 1

    try:
        results = resolve_alleles(
            index_csv=args.index_csv,
            alleles=alleles,
            include_sequence=not args.no_seq,
        )
    except MHCIndexError as exc:
        print(f"Error resolving alleles: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else None
    _write_resolve_output(results, output_path, args.format)
    return 0
