"""Verify selected original member headers/CRCs and independent zipfile replay."""

import argparse
import hashlib
import io
import json
import resource
import struct
import sys
import time
import zipfile
import zlib
from pathlib import Path, PurePosixPath

started = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "reproduce/launch.json").read_text())
raw = Path(config["raw_directory"])
directory = json.loads((ROOT / "results/zip-directory.json").read_text())
parser = argparse.ArgumentParser()
parser.add_argument("--evidence", action="store_true")
args = parser.parse_args()
range_names = ["Hi20S_txt.zip.tail", "Hi20S_txt.parameters-peptides.range"]
selected = directory["selected"]
if args.evidence:
    range_names = ["Hi20S_txt.zip.tail", "Hi20S_txt.evidence-modified.range"]
    selected = [e for e in directory["members"] if PurePosixPath(e["name"]).name in
                ("evidence.txt", "modificationSpecificPeptides.txt")]
    assert len(selected) == 2
ranges = []
for name in range_names:
    receipt = json.loads((raw / (name + ".receipt.json")).read_text())
    assert receipt["access_status"] == "retrieved_and_length_verified"
    data = (raw / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == receipt["sha256"]
    start, end = map(int, receipt["requested_range"].split(":"))
    assert len(data) == end - start + 1
    ranges.append((start, end + 1, data, name))


def original(offset, length):
    matches = [(start, data, name) for start, end, data, name in ranges
               if start <= offset and offset + length <= end]
    assert len(matches) == 1, (offset, length)
    start, data, name = matches[0]
    return data[offset - start:offset - start + length], name


locals_, centrals, payloads, records = [], [], {}, []
position = 0
for entry in selected:
    assert entry["uncompressed_bytes"] < config["limits"]["asset_bytes"]
    header, _ = original(entry["local_header_offset"], 30)
    signature, needed, flags, method, mtime, mdate, crc, compressed, unpacked, n_name, n_extra = struct.unpack("<4s5H3I2H", header)
    assert signature == b"PK\x03\x04" and flags == entry["flags"] == 0
    assert (method, crc, compressed, unpacked) == tuple(entry[k] for k in ("method", "crc32", "compressed_bytes", "uncompressed_bytes"))
    block, source_range = original(entry["local_header_offset"], 30+n_name+n_extra+compressed)
    assert block[30:30+n_name].decode("cp437") == entry["name"]
    decoder = zlib.decompressobj(-15)
    data = decoder.decompress(block[30+n_name+n_extra:], unpacked + 1)
    assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
    assert len(data) == unpacked and zlib.crc32(data) == crc
    central, _ = original(entry["directory_offset"], entry["directory_record_bytes"])
    central = bytearray(central)
    struct.pack_into("<I", central, 42, position)
    centrals.append(bytes(central))
    locals_.append(block)
    position += len(block)
    payloads[entry["name"]] = data
    records.append({**entry, "retained_range": source_range, "sha256": hashlib.sha256(data).hexdigest()})

local_bytes = b"".join(locals_)
central_bytes = b"".join(centrals)
end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, len(records), len(records), len(central_bytes), len(local_bytes), 0)
with zipfile.ZipFile(io.BytesIO(local_bytes + central_bytes + end)) as independent:
    assert independent.testzip() is None
    assert independent.namelist() == list(payloads)
    for name, data in payloads.items():
        assert independent.read(name) == data

destination = raw / "processed-members"
destination.mkdir(exist_ok=True)
for name, data in payloads.items():
    with (destination / PurePosixPath(name).name).open("xb") as output:
        output.write(data)
result = dict(passed=True, n_members=len(records), members=records,
              argv=sys.argv, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              original_central_and_local_headers_verified=True,
              independent_zipfile_bytes_and_crcs_equal=True,
              full_original_archive_checksum_verified=False,
              seconds=time.monotonic()-started,
              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
result_name = "evidence-member-verification.json" if args.evidence else "member-verification.json"
with (ROOT / "results" / result_name).open("x") as output:
    output.write(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
