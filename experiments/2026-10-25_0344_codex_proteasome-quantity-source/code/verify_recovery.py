"""Independent ZIP-library replay and explicit source-date representation audit."""

import datetime
import hashlib
import io
import json
import struct
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
RAW = Path(json.loads((EXP / 'reproduce/launch.json').read_text())['raw_directory'])
start = time.monotonic()
recovery = json.loads((EXP / 'results/member-recovery.json').read_text())
source = (RAW / 'sahu-supplementary.zip').read_bytes()
assert hashlib.sha256(source).hexdigest() == recovery['partial_archive_sha256']
directory = []
for r in recovery['records']:
    fields = struct.unpack_from('<IHHHHHIIIHH', source, r['local_header_offset'])
    signature, version, flags, method, mtime, mdate, crc, cs, us, nlen, elen = fields
    assert signature == 0x04034B50
    name_start = r['local_header_offset'] + 30
    name = source[name_start:name_start + nlen]
    assert name.decode('utf-8') == r['name']
    assert name_start + nlen + elen == r['data_offset']
    pos = r['descriptor_offset']
    if source[pos:pos + 4] == b'PK\x07\x08':
        pos += 4
    crc, cs, us = struct.unpack_from('<III', source, pos)
    assert pos + 12 == r['end_offset'] and r['data_offset'] + cs == r['descriptor_offset']
    assert cs == r['compressed_bytes'] and us == r['uncompressed_bytes']
    directory.append(struct.pack('<I6H3I5H2I', 0x02014B50, 20, version, flags, method,
                                 mtime, mdate, crc, cs, us, nlen, 0, 0, 0, 0, 0,
                                 r['local_header_offset']) + name)
central = b''.join(directory)
prefix = source[:recovery['verified_through_offset']]
end = struct.pack('<I4H2IH', 0x06054B50, 0, 0, len(directory), len(directory),
                  len(central), len(prefix), 0)
with zipfile.ZipFile(io.BytesIO(prefix + central + end)) as verified:
    assert verified.testzip() is None
    assert len(verified.infolist()) == len(recovery['records'])
    for r in recovery['records']:
        payload = verified.read(r['name'])
        assert hashlib.sha256(payload).hexdigest() == r['sha256']
        if r['recovered']:
            assert Path(r['path']).read_bytes() == payload
            with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                assert inner.testzip() is None

inventory = json.loads((EXP / 'results/workbook-inventory.json').read_text())
assert not inventory['mismatches']
assert len(inventory['date_cells']) == 13
ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
dates = []
for filename, sheet, coord, rendered, item in inventory['date_cells']:
    assert filename == '41467_2021_26427_MOESM10_ESM.xlsx' and sheet == 'peptides'
    assert coord.startswith('H') and item['style'] == '2' and item['formula'] is None
    with zipfile.ZipFile(RAW / 'complete-members' / filename) as wb:
        props = ET.fromstring(wb.read('xl/workbook.xml')).find('s:workbookPr', ns)
        assert props is None or props.attrib.get('date1904', '0') in ('0', 'false')
        styles = ET.fromstring(wb.read('xl/styles.xml')).find('s:cellXfs', ns)
        assert styles[2].attrib['numFmtId'] == '17'
    serial = int(item['raw_value'])
    assert serial > 60 and serial == item['value']
    manual = datetime.datetime(1899, 12, 30) + datetime.timedelta(days=serial)
    assert str(manual) == rendered
    dates.append({'file': filename, 'sheet': sheet, 'cell': coord,
                  'raw_excel_serial': serial, 'rendered_date': rendered,
                  'gene_name_reconstructed': False})
count = sum(s['nonempty_cells'] for f in inventory['files'] for s in f['sheets'])
result = {'passed': True, 'complete_outer_members_checked': len(directory),
          'complete_workbooks_checked': sum(r['recovered'] for r in recovery['records']),
          'literal_populated_cells_verified': count, 'date_shaped_gene_annotations': dates,
          'original_failed_inventory_preserved': True, 'source_values_rewritten': False,
          'outer_central_directory_verified': False,
          'scope': 'Complete-member original data descriptors plus independently rebuilt index and standard ZIP CRC; all recovered workbook cells with explicit date representation check.',
          'seconds': time.monotonic() - start}
dest = EXP / 'results/independent-recovery-verification.json'
with dest.open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({k: v for k, v in result.items() if k != 'date_shaped_gene_annotations'}))
