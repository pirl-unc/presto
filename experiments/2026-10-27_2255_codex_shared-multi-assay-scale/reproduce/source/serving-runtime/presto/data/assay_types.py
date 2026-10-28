"""Assay semantics for merged source records, independent of label missingness.

Concentration labels use the merged format's existing nM contract. Unrecognized
measurements cannot acquire a unit or a training target merely by being numeric.
"""

import re

_MEASUREMENT_BUCKETS = {
    label: bucket
    for bucket, labels in {
        "binding_affinity": (
            "ic50",
            "ec50",
            "kd",
            "kd (~ic50)",
            "kd (~ec50)",
            "half maximal inhibitory concentration (ic50)",
            "half maximal effective concentration (ec50)",
            "dissociation constant kd",
            "dissociation constant kd (~ic50)",
            "dissociation constant kd (~ec50)",
            "dissociation constant",
        ),
        "binding_kon": ("kon", "k_on", "on rate", "on-rate", "association rate"),
        "binding_koff": ("koff", "k_off", "off rate", "off-rate", "dissociation rate"),
        "binding_t_half": (
            "t_half",
            "half life",
            "half-life",
            "t1/2",
            "dissociation half life",
            "dissociation half-life",
        ),
        "binding_tm": (
            "tm",
            "melting temperature",
            "melting point",
            "50% dissociation temperature",
            "dissociation temperature",
        ),
        "binding_qualitative": ("qualitative binding", "mhc binding"),
        "binding_structure": ("3d structure", "x-ray crystallography", "electron microscopy"),
        "binding_association_constant": ("ka", "association constant ka", "association constant"),
        "processing": ("processing", "cleavage", "tap transport", "tap assay", "erap"),
    }.items()
    for label in labels
}
_MS_METHOD = re.compile(r"\b(?:mass[ -]spectrometry|lc[ /-]?ms(?:/ms)?|ms/ms)\b")
_DIA = re.compile(r"\b(?:dia|data[ -]independent(?: acquisition)?)\b")
_DDA = re.compile(r"\b(?:dda|data[ -]dependent(?: acquisition)?)\b")
_TARGETED = re.compile(
    r"\b(?:prm|srm|mrm|parallel reaction monitoring|selected reaction monitoring|"
    r"multiple reaction monitoring)\b"
)


def _normalized(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def elution_assay_bucket(method: str) -> str:
    """Select an acquisition subtype after elution/MS evidence is established."""
    method = _normalized(method)
    if _DIA.search(method):
        return "elution_ms_dia"
    if _DDA.search(method):
        return "elution_ms_dda"
    if _TARGETED.search(method) or re.search(r"\btargeted\b", method):
        return "elution_ms_targeted"
    return "elution_ms"


def binding_assay_bucket(measurement: str, method: str) -> str:
    """Classify an explicit measurement label; retain unsupported source families.

    Qualitative binding, structure and equilibrium association constants are
    recognized observations with no matching canonical objective. They remain
    separate from both an unknown measurement and an absent supported label.
    """
    measurement, method = _normalized(measurement), _normalized(method)
    if measurement in _MEASUREMENT_BUCKETS:
        return _MEASUREMENT_BUCKETS[measurement]

    presentation = measurement == "ligand presentation"
    # A declared measurement has priority over the method used to measure it.
    # Only an absent measurement (or the method itself used as a fallback label)
    # permits classification from the method alone.
    if not presentation and measurement not in {"", method}:
        return "binding_unknown"
    if _MS_METHOD.search(method) or (
        presentation and (_DIA.search(method) or _DDA.search(method) or _TARGETED.search(method))
    ):
        return elution_assay_bucket(method)
    if presentation:
        return "presentation_non_ms" if method else "presentation_unknown_method"
    return "binding_unknown"
