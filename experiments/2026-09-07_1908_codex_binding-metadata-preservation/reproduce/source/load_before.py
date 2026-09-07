from __future__ import annotations

def load_records_from_merged_tsv(
    merged_tsv: Path,
    *,
    max_binding: Optional[int],
    max_kinetics: Optional[int],
    max_stability: Optional[int],
    max_processing: Optional[int],
    max_elution: Optional[int],
    max_tcell: Optional[int],
    max_vdjdb: Optional[int],
    cap_sampling: str = "reservoir",
    sampling_seed: int = 42,
) -> Tuple[
    List[BindingRecord],
    List[KineticsRecord],
    List[StabilityRecord],
    List[ProcessingRecord],
    List[ElutionRecord],
    List[TCellRecord],
    List[TcrEvidenceRecord],
    Dict[str, Any],
]:
    """Load canonical training records directly from merged TSV output."""
    if not merged_tsv.exists():
        raise FileNotFoundError(f"Merged TSV not found: {merged_tsv}")

    binding_records: List[BindingRecord] = []
    kinetics_records: List[KineticsRecord] = []
    stability_records: List[StabilityRecord] = []
    processing_records: List[ProcessingRecord] = []
    elution_records: List[ElutionRecord] = []
    tcell_records: List[TCellRecord] = []
    vdjdb_records: List[TcrEvidenceRecord] = []
    by_assay: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    rows_dropped_invalid_peptide = 0
    rows_sanitized_optional_sequences = 0

    binding_limit = _normalize_limit(max_binding)
    kinetics_limit = _normalize_limit(max_kinetics)
    stability_limit = _normalize_limit(max_stability)
    processing_limit = _normalize_limit(max_processing)
    elution_limit = _normalize_limit(max_elution)
    tcell_limit = _normalize_limit(max_tcell)
    vdjdb_limit = _normalize_limit(max_vdjdb)
    sampling_mode = str(cap_sampling or "reservoir").strip().lower()
    if sampling_mode not in {"head", "reservoir"}:
        raise ValueError(
            f"Unsupported cap sampling mode: {cap_sampling!r}. Expected one of: head, reservoir."
        )
    sampling_rng = random.Random(int(sampling_seed))
    binding_seen = 0
    kinetics_seen = 0
    stability_seen = 0
    processing_seen = 0
    elution_seen = 0
    tcell_seen = 0
    vdjdb_seen = 0

    with merged_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            peptide = _normalize_required_aa_sequence(row.get("peptide"))
            if not peptide:
                rows_dropped_invalid_peptide += 1
                continue

            mhc_allele = (row.get("mhc_allele") or "").strip()
            mhc_allele_set = parse_allele_set_field(row.get("mhc_allele_set"))
            mhc_class = normalize_mhc_class(
                row.get("mhc_class"),
                default=infer_mhc_class_optional(mhc_allele),
            )
            source = (row.get("source") or "").strip() or "unknown"
            species = (row.get("species") or "").strip() or infer_species(mhc_allele) or None
            antigen_species_col = (row.get("antigen_species") or "").strip() or None
            value = _parse_float_or_none(row.get("value"))
            qualifier = _parse_int_or_default(row.get("qualifier"), default=0)
            value_type = (row.get("value_type") or "").strip()
            response = (row.get("response") or "").strip()
            assay_type_col = (row.get("assay_type") or "").strip()
            assay_method_col = (row.get("assay_method") or "").strip()
            apc_name_col = (row.get("apc_name") or "").strip()
            effector_culture_col = (row.get("effector_culture_condition") or "").strip()
            apc_culture_col = (row.get("apc_culture_condition") or "").strip()
            in_vitro_process_col = (row.get("in_vitro_process_type") or "").strip()
            in_vitro_responder_col = (row.get("in_vitro_responder_cell") or "").strip()
            in_vitro_stimulator_col = (row.get("in_vitro_stimulator_cell") or "").strip()
            record_type = (row.get("record_type") or "").strip()
            cdr3_alpha, alpha_sanitized = _normalize_optional_aa_sequence(row.get("cdr3_alpha"))
            cdr3_beta, beta_sanitized = _normalize_optional_aa_sequence(row.get("cdr3_beta"))
            rows_sanitized_optional_sequences += int(alpha_sanitized) + int(beta_sanitized)
            trav = (row.get("trav") or "").strip()
            trbv = (row.get("trbv") or "").strip()

            unified = UnifiedRecord(
                peptide=peptide,
                mhc_allele=mhc_allele,
                mhc_class=mhc_class,
                source=source,
                record_type=record_type,
                value=value,
                value_type=value_type,
                qualifier=qualifier,
                response=response,
                assay_type=assay_type_col or None,
                assay_method=assay_method_col or None,
                apc_name=apc_name_col or None,
                effector_culture_condition=effector_culture_col or None,
                apc_culture_condition=apc_culture_col or None,
                in_vitro_process_type=in_vitro_process_col or None,
                in_vitro_responder_cell=in_vitro_responder_col or None,
                in_vitro_stimulator_cell=in_vitro_stimulator_col or None,
                cdr3_alpha=cdr3_alpha or None,
                cdr3_beta=cdr3_beta or None,
                trav=trav or None,
                trbv=trbv or None,
                species=species,
                antigen_species=antigen_species_col,
            )
            assay_type = classify_assay_type(unified)
            by_assay[assay_type] = by_assay.get(assay_type, 0) + 1
            by_source[source] = by_source.get(source, 0) + 1

            if assay_type == "binding_affinity":
                if value is None:
                    continue
                binding_seen = _append_with_cap_sampling(
                    binding_records,
                    BindingRecord(
                        peptide=peptide,
                        mhc_allele=mhc_allele,
                        value=value,
                        qualifier=qualifier,
                        measurement_type=value_type or "IC50",
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                        alleles=mhc_allele_set or None,
                    ),
                    binding_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=binding_seen,
                )
                continue

            if assay_type == "binding_kon":
                if value is None:
                    continue
                kinetics_seen = _append_with_cap_sampling(
                    kinetics_records,
                    KineticsRecord(
                        peptide=peptide,
                        mhc_allele=mhc_allele,
                        kon=value,
                        koff=None,
                        kon_qualifier=qualifier,
                        assay_type=value_type or "kon",
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                        alleles=mhc_allele_set or None,
                    ),
                    kinetics_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=kinetics_seen,
                )
                continue

            if assay_type == "binding_koff":
                if value is None:
                    continue
                kinetics_seen = _append_with_cap_sampling(
                    kinetics_records,
                    KineticsRecord(
                        peptide=peptide,
                        mhc_allele=mhc_allele,
                        kon=None,
                        koff=value,
                        koff_qualifier=qualifier,
                        assay_type=value_type or "koff",
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                        alleles=mhc_allele_set or None,
                    ),
                    kinetics_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=kinetics_seen,
                )
                continue

            if assay_type == "binding_t_half":
                if value is None:
                    continue
                stability_seen = _append_with_cap_sampling(
                    stability_records,
                    StabilityRecord(
                        peptide=peptide,
                        mhc_allele=mhc_allele,
                        t_half=value,
                        tm=None,
                        t_half_qualifier=qualifier,
                        assay_type=value_type or "t_half",
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                        alleles=mhc_allele_set or None,
                    ),
                    stability_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=stability_seen,
                )
                continue

            if assay_type == "binding_tm":
                if value is None:
                    continue
                stability_seen = _append_with_cap_sampling(
                    stability_records,
                    StabilityRecord(
                        peptide=peptide,
                        mhc_allele=mhc_allele,
                        t_half=None,
                        tm=value,
                        tm_qualifier=qualifier,
                        assay_type=value_type or "Tm",
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                        alleles=mhc_allele_set or None,
                    ),
                    stability_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=stability_seen,
                )
                continue

            if assay_type.startswith("elution_ms"):
                detected = _parse_binary_response(response)
                if detected is None:
                    detected = 0.0 if source.startswith("synthetic_negative") else 1.0
                alleles = list(mhc_allele_set)
                if not alleles:
                    alleles = _split_allele_list(mhc_allele)
                if not alleles and mhc_allele:
                    alleles = [mhc_allele]
                if not alleles:
                    continue
                elution_seen = _append_with_cap_sampling(
                    elution_records,
                    ElutionRecord(
                        peptide=peptide,
                        alleles=alleles,
                        detected=bool(detected > 0.5),
                        source_alleles=tuple(alleles),
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                    ),
                    elution_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=elution_seen,
                )
                continue

            if assay_type == "tcell_response":
                response_value = _parse_binary_response(response)
                if response_value is None:
                    continue
                tcell_seen = _append_with_cap_sampling(
                    tcell_records,
                    TCellRecord(
                        peptide=peptide,
                        mhc_allele=mhc_allele,
                        response=response_value,
                        alleles=mhc_allele_set or None,
                        assay_type=assay_type_col or value_type or None,
                        assay_method=assay_method_col or None,
                        apc_name=apc_name_col or None,
                        effector_culture_condition=effector_culture_col or None,
                        apc_culture_condition=apc_culture_col or None,
                        in_vitro_process_type=in_vitro_process_col or None,
                        in_vitro_responder_cell=in_vitro_responder_col or None,
                        in_vitro_stimulator_cell=in_vitro_stimulator_col or None,
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                    ),
                    tcell_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=tcell_seen,
                )
                continue

            if assay_type in {"tcr_pmhc", "tcr_evidence"}:
                if not mhc_allele:
                    continue
                vdjdb_seen = _append_with_cap_sampling(
                    vdjdb_records,
                    TcrEvidenceRecord(
                        peptide=peptide,
                        mhc_a=mhc_allele,
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                        evidence_label=1.0,
                        method_bins=(),
                    ),
                    vdjdb_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=vdjdb_seen,
                )
                continue

            if assay_type == "processing" or record_type == "processing":
                label = _parse_binary_response(response)
                if label is None:
                    label = 1.0
                processing_seen = _append_with_cap_sampling(
                    processing_records,
                    ProcessingRecord(
                        peptide=peptide,
                        label=label,
                        processing_type=value_type or "processing",
                        mhc_allele=mhc_allele or None,
                        mhc_class=mhc_class,
                        species=species,
                        antigen_species=antigen_species_col,
                        source=source,
                    ),
                    processing_limit,
                    sampling=sampling_mode,
                    rng=sampling_rng,
                    seen=processing_seen,
                )

    counts_before_cap = {
        "binding": binding_seen,
        "kinetics": kinetics_seen,
        "stability": stability_seen,
        "processing": processing_seen,
        "elution": elution_seen,
        "tcell": tcell_seen,
        "tcr_evidence": vdjdb_seen,
    }
    records_loaded = {
        "binding": len(binding_records),
        "kinetics": len(kinetics_records),
        "stability": len(stability_records),
        "processing": len(processing_records),
        "elution": len(elution_records),
        "tcell": len(tcell_records),
        "tcr_evidence": len(vdjdb_records),
    }
    rows_scanned = sum(by_assay.values())
    stats = {
        "rows_scanned": rows_scanned,
        "rows_by_assay": dict(sorted(by_assay.items(), key=lambda item: (-item[1], item[0]))),
        "rows_by_source": dict(sorted(by_source.items(), key=lambda item: (-item[1], item[0]))),
        "rows_dropped_invalid_peptide": rows_dropped_invalid_peptide,
        "rows_sanitized_optional_sequences": rows_sanitized_optional_sequences,
        "skipped_invalid_peptide": rows_dropped_invalid_peptide,
        "skipped_unroutable_or_missing_label": max(
            0, rows_scanned - sum(counts_before_cap.values())
        ),
        "cap_sampling": sampling_mode,
        "counts_before_cap": counts_before_cap,
        "records_loaded": records_loaded,
        "rows_dropped_by_cap": {
            name: max(0, count - records_loaded[name]) for name, count in counts_before_cap.items()
        },
    }
    return (
        binding_records,
        kinetics_records,
        stability_records,
        processing_records,
        elution_records,
        tcell_records,
        vdjdb_records,
        stats,
    )
