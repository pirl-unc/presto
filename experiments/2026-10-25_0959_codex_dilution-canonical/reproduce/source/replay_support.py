"""Independent source readers and rational arithmetic; no Hitlist helpers."""

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pandas as pd
import pyarrow.parquet as pq

MISSING = {"", "NA", "NaN", "nan", "null"}
PRIMARY = {
    "assets": "source_asset_id",
    "source_records": "source_record_id",
    "assertions": "claim_id",
    "nominal_standards": "standard_id",
    "nominal_levels": "nominal_level_id",
    "acquisitions": "acquisition_id",
    "library_precursors": "library_precursor_id",
    "library_transitions": "transition_id",
    "scored_features": "identification_id",
    "aligned_quantities": "measurement_id",
    "feature_links": "claim_id",
    "comparisons": "claim_id",
    "standard_memberships": "claim_id",
    "material_contexts": "context_id",
    "material_links": "claim_id",
}
MEASURES = {
    "intensity": ("quantitative_raw", "quantitative_value"),
    "RT": ("dilution_rt_raw", "dilution_rt"),
    "leftWidth": ("dilution_left_width_raw", "dilution_left_width"),
    "rightWidth": ("dilution_right_width_raw", "dilution_right_width"),
    "peak_group_rank": ("dilution_peak_rank_raw", "dilution_peak_rank"),
    "m_score": ("dilution_m_score_raw", "dilution_m_score"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def numeric(literal):
    if literal in MISSING:
        return None, "missing"
    try:
        rational = Fraction(literal)
    except (ValueError, ZeroDivisionError):
        try:
            return (None, "nonfinite") if not math.isfinite(float(literal)) else (None, "invalid")
        except ValueError:
            return None, "invalid"
    try:
        value = float(rational)
    except OverflowError:
        return None, "outside_float_range"
    if not math.isfinite(value) or (value == 0 and rational != 0):
        return None, "outside_float_range"
    return value, "positive" if rational > 0 else "negative" if rational < 0 else "zero"


def relation(a, b, atol, rtol):
    if a in MISSING or b in MISSING:
        return (
            "both_missing"
            if a in MISSING and b in MISSING
            else "aligned_missing"
            if a in MISSING
            else "scored_missing"
        ), None
    x, y = Fraction(a), Fraction(b)
    return (
        "same_literal"
        if a == b
        else "decimal_equal"
        if x == y
        else "within_numeric_tolerance"
        if abs(x - y) <= Fraction(atol) + Fraction(rtol) * max(abs(x), abs(y))
        else "different"
    ), x - y


def measures(row, scored):
    result = {"measurement_unit": "source_signal_arbitrary_units"}
    for field, (raw, parsed) in MEASURES.items():
        literal = row["Intensity" if scored and field == "intensity" else field]
        value, state = numeric(literal)
        result[raw], result[parsed] = literal, value
        if field == "intensity":
            result["value_status"] = state
    return result


def literal_rows(path):
    index = 0
    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, chunksize=2048
    ):
        header = list(chunk.columns)
        for cells in chunk.itertuples(index=False, name=None):
            index += 1
            yield index, header, list(cells)


def metadata(config, contract):
    assets = {k: Path(v) for k, v in config["assets"].items()}
    records = {}
    project = json.loads(assets[contract["project"]].read_text())
    for key, value in project.items():
        records[contract["project"] + "#/" + key] = (
            contract["project"],
            ["key", "value_json"],
            [key, json.dumps(value, ensure_ascii=True)],
        )
    inventory = json.loads(assets[contract["inventory"]].read_text())
    for index, row in enumerate(inventory):
        records[contract["inventory"] + "#/" + str(index)] = (
            contract["inventory"],
            list(row),
            [json.dumps(v, ensure_ascii=True) for v in row.values()],
        )
    article = ET.parse(assets[contract["article"]]).getroot()
    level_cells = []
    for index, row in enumerate(article.findall(".//table-wrap[@id='T1']//tbody/tr"), 1):
        cells = ["".join(cell.itertext()) for cell in row.findall("td")]
        locator = f"{contract['article']}#table-wrap:T1/tbody/tr:{index}"
        records[locator] = (
            contract["article"],
            ["dilution_factor", "sample", "concentration"],
            cells,
        )
        level_cells.append((locator, cells))
    paragraphs = []
    for index, paragraph in enumerate(article.findall(".//body//p"), 1):
        text = "".join(paragraph.itertext())
        locator = f"{contract['article']}#body/p:{index}"
        records[locator] = contract["article"], ["text"], [text]
        paragraphs.append((locator, text))
    commands = []
    for asset in contract["assets"]:
        if asset["role"] != "pipeline_script":
            continue
        for index, text in enumerate(
            assets[asset["filename"]].read_text().splitlines(keepends=True), 1
        ):
            locator = f"{asset['filename']}:line:{index}"
            records[locator] = asset["filename"], ["source_code_line"], [text]
            if any(
                token in text
                for token in (
                    "pyprophet export",
                    "max_rs_peakgroup_qvalue",
                    "max_global_peptide_qvalue",
                    "max_global_protein_qvalue",
                    "alignTargetedRuns",
                    "mv DIAlignR.tsv",
                )
            ):
                commands.append((locator, text))
    return records, project, inventory, level_cells, paragraphs, commands


class Replay:
    def __init__(self, config, contract):
        self.config, self.contract = config, contract
        self.typed = Path(config["package"]) / "dilution_observer/typed"
        self.counts = Counter()
        self.visited = set()
        self.common = {
            k: contract[k] for k in ("source_dataset", "source_dataset_precision", "source_panel")
        }

    def rows(self, name):
        assert name not in self.visited, "Verify each relation completely once"
        self.visited.add(name)
        keys = set()
        for batch in pq.ParquetFile(self.typed / (name + ".parquet")).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                key = row[PRIMARY[name]]
                assert re.fullmatch(r"pxd024809:[a-z_]+:[0-9a-f]{64}", key), (name, key)
                assert key not in keys, (name, key)
                keys.add(key)
                assert all(row[k] == v for k, v in self.common.items())
                self.counts[name] += 1
                yield row

    def check(self, name, row, expected):
        expected = {**self.common, PRIMARY[name]: row[PRIMARY[name]], **expected}
        assert set(expected) <= set(row), (name, set(expected) - set(row))
        assert row == {key: expected.get(key) for key in row}, (name, row, expected)


def exhausted(iterator):
    sentinel = object()
    assert next(iterator, sentinel) is sentinel, "Unexpected extra canonical row"
