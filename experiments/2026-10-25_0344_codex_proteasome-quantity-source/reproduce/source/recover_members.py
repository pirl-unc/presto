"""Recover complete, checksum-verified XLSX records from retained partial ZIP bytes."""

import hashlib
import io
import json
import shutil
import struct
import time
import zipfile
import zlib
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
CONFIG = json.loads((EXP / 'reproduce/launch.json').read_text())
RAW = Path(CONFIG['raw_directory'])
SOURCE = RAW / 'sahu-supplementary.zip'
DEST = RAW / 'complete-members'
assert not DEST.exists()
assert shutil.disk_usage(RAW).free >= 1024**3 + 32 * 1024**2
DEST.mkdir()
start = time.monotonic()
content = SOURCE.read_bytes()
cursor = 0
records = []
reason = None
while cursor + 30 <= len(content):
    begin = cursor
    fields = struct.unpack_from('<IHHHHHIIIHH', content, cursor)
    signature, version, flags, method, mtime, mdate, crc, csize, usize, nlen, elen = fields
    if signature != 0x04034B50:
        reason = f'next_record_not_local_header_at_{cursor}'
        break
    assert method == 8 and flags & 8, 'Only qualified deflate/data-descriptor records'
    assert not flags & 1, 'Encrypted member is not supported'
    name = content[cursor + 30:cursor + 30 + nlen].decode('utf-8')
    cursor += 30 + nlen + elen
    data_start = cursor
    decoder = zlib.decompressobj(-15)
    chunks = []
    total = 0
    while cursor < len(content) and not decoder.eof:
        chunk = content[cursor:cursor + 65536]
        cursor += len(chunk)
        out = decoder.decompress(chunk, 32 * 1024**2 - total + 1)
        total += len(out)
        assert total <= 32 * 1024**2, 'Member expansion cap'
        assert not decoder.unconsumed_tail, 'Unexpected decompression limit'
        chunks.append(out)
    if not decoder.eof:
        reason = f'truncated_member:{name}'
        break
    cursor -= len(decoder.unused_data)
    compressed_length = cursor - data_start
    descriptor_start = cursor
    if cursor + 4 <= len(content) and content[cursor:cursor + 4] == b'PK\x07\x08':
        cursor += 4
    if cursor + 12 > len(content):
        reason = f'truncated_descriptor:{name}'
        break
    expected_crc, expected_compressed, expected_size = struct.unpack_from('<III', content, cursor)
    cursor += 12
    value = b''.join(chunks)
    assert len(value) == expected_size
    assert compressed_length == expected_compressed
    assert zlib.crc32(value) & 0xffffffff == expected_crc
    record = {'name': name, 'local_header_offset': begin, 'data_offset': data_start,
              'descriptor_offset': descriptor_start, 'end_offset': cursor,
              'compressed_bytes': compressed_length, 'uncompressed_bytes': len(value),
              'crc32': f'{expected_crc:08x}', 'sha256': hashlib.sha256(value).hexdigest(),
              'recovered': False}
    if name.endswith('.xlsx'):
        assert Path(name).name == name and len(value) <= CONFIG['limits']['asset_bytes']
        with zipfile.ZipFile(io.BytesIO(value)) as workbook:
            assert workbook.testzip() is None
            assert 'xl/workbook.xml' in workbook.namelist()
        assert sum(p.stat().st_size for p in RAW.rglob('*') if p.is_file()) + len(value) < CONFIG['limits']['total_raw_bytes']
        path = DEST / name
        with path.open('xb') as stream:
            stream.write(value)
        record.update(recovered=True, path=str(path), workbook_crc_verified=True)
    records.append(record)
    print(json.dumps({k: record[k] for k in ('name', 'uncompressed_bytes', 'recovered')}), flush=True)
    assert time.monotonic() - start < 1200
    assert shutil.disk_usage(RAW).free >= 1024**3 + 32 * 1024**2
result = {'partial_archive_sha256': hashlib.sha256(content).hexdigest(), 'records': records,
          'terminal_reason': reason, 'verified_through_offset': records[-1]['end_offset'] if records else 0,
          'partial_archive_bytes': len(content), 'outer_central_directory_verified': False,
          'seconds': time.monotonic() - start}
(EXP / 'results/member-recovery.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'records'}))
