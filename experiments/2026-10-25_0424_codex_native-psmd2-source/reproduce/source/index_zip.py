"""Parse only the retained original ZIP directory, without fetching the archive."""

import json
import struct
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())
RAW = Path(CONFIG["raw_directory"])
body = (RAW / "Hi20S_txt.zip.tail").read_bytes()
receipt = json.loads((RAW / "Hi20S_txt.zip.tail.receipt.json").read_text())
assert receipt["access_status"] == "retrieved_and_length_verified"
tail_start = 167324239
position = body.rfind(b"PK\x05\x06")
assert position >= 0
signature, disk, directory_disk, disk_entries, n_entries, size, offset, comment_length = struct.unpack_from("<4s4H2IH", body, position)
assert disk == directory_disk == 0 and disk_entries == n_entries
assert position + 22 + comment_length == len(body)
assert 0 < n_entries < 65535 and tail_start <= offset
assert offset + size == tail_start + position
cursor = offset - tail_start
entries = []
for index in range(n_entries):
    values = struct.unpack_from("<4s6H3I5H2I", body, cursor)
    assert values[0] == b"PK\x01\x02"
    _, made, needed, flags, method, mtime, mdate, crc, compressed, unpacked, n_name, n_extra, n_comment, member_disk, internal, external, local_offset = values
    name = body[cursor + 46:cursor + 46 + n_name].decode("utf-8" if flags & 0x800 else "cp437")
    assert not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
    assert member_disk == 0 and not flags & 1 and method in (0, 8)
    assert local_offset < offset and compressed < 0xFFFFFFFF and unpacked < 0xFFFFFFFF
    entries.append(dict(name=name, flags=flags, method=method, crc32=crc,
                        compressed_bytes=compressed, uncompressed_bytes=unpacked,
                        local_header_offset=local_offset, directory_offset=tail_start + cursor,
                        directory_record_bytes=46+n_name+n_extra+n_comment))
    cursor += 46 + n_name + n_extra + n_comment
assert cursor == position and len({e["name"] for e in entries}) == len(entries)
selected = [e for e in entries if PurePosixPath(e["name"]).name in ("peptides.txt", "summary.txt", "parameters.txt")]
assert any(PurePosixPath(e["name"]).name == "peptides.txt" for e in selected)
for entry in selected:
    assert entry["compressed_bytes"] < CONFIG["limits"]["asset_bytes"]
    assert entry["uncompressed_bytes"] < CONFIG["limits"]["asset_bytes"]
result = dict(archive_size_bytes=167389796, original_directory_offset=offset,
              original_directory_bytes=size, n_members=len(entries), members=entries,
              selected=selected, full_archive_checksum_verified=False)
with (ROOT / "results/zip-directory.json").open("x") as output:
    output.write(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ("n_members", "original_directory_bytes", "selected")}, indent=2))
