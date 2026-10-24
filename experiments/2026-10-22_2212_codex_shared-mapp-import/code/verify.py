"""Independently replay original TSV rows, XML cells and quantity derivations."""
from collections import defaultdict, Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import xml.etree.ElementTree as ET
import zipfile

import pyarrow.parquet as pq

FAMILY=Path(__file__).resolve().parents[1]
ROOT=Path('/private/tmp/hitlist-v2-mapp-external-first')
SOURCE=ROOT/'external/pxd009092'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def rows(name):
    yield from pq.read_table(SOURCE/'typed'/f'{name}.parquet').to_pylist()


def tsv(name):
    with (SOURCE/'assets'/name).open(newline='') as stream:
        yield from csv.DictReader(stream,delimiter='\t')


def xml_cells(path):
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=ET.fromstring(z.read('xl/sharedStrings.xml'))
            strings=[''.join(x.itertext()) for x in root.findall('s:si',ns)]
        names=[x.attrib['name'] for x in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',ns)]
        rels={x.attrib['Id']:x.attrib['Target'] for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        sheets=ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',ns)
        for name,sheet in zip(names,sheets):
            target=rels[sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
            target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
            root=ET.fromstring(z.read(target))
            for cell in root.findall('.//s:sheetData/s:row/s:c',ns):
                value=cell.find('s:v',ns); kind=cell.attrib.get('t','n')
                if cell.find('s:f',ns) is not None:raise AssertionError('Unexpected source formula')
                if kind=='inlineStr':
                    v=''.join(cell.find('s:is',ns).itertext())
                elif value is None or value.text is None:continue
                elif kind=='s':v=strings[int(value.text)]
                elif kind=='n':v=float(value.text)
                else:v=value.text
                yield name,cell.attrib['r'],v


def main():
    tick=time.monotonic()
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert set(manifest['files'])=={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p!=ROOT/'manifest.json'}
    for name,expected in manifest['files'].items():
        p=ROOT/name
        assert p.stat().st_size==expected['size_bytes'] and sha(p)==expected['sha256'],name
    raw=defaultdict(dict)
    for r in rows('source_records'):
        assert r['row_1based'] not in raw[r['asset_path']]
        raw[r['asset_path']][r['row_1based']]=json.loads(r['fields_json'])
    n_raw=0
    for asset,got in raw.items():
        expected={i:r for i,r in enumerate(tsv(asset),2)}
        assert got==expected,asset
        n_raw+=len(expected)
    peptides={r['id']:r for r in tsv('txt/peptides.txt')}
    sums=defaultdict(float); evidence=defaultdict(list)
    for r in tsv('txt/evidence.txt'):
        key=r['Peptide ID'],r['Experiment']
        sums[key]+=float(r['Intensity'] or 0)
        evidence[key].append(r)
    source_evidence={r['id']:r for records in evidence.values() for r in records}
    missing_confidence=0
    for r in rows('evidence'):
        original=source_evidence[r['evidence_id'].rsplit(':',1)[1]]
        assert r['pep_raw']==original['PEP']
        if original['PEP']=='NaN':
            assert r['pep'] is None and r['pep_status']=='unreported_source_nan'
            missing_confidence+=1
        else: assert r['pep']==float(original['PEP'])
        assert r['intensity_raw']==original['Intensity']
        assert r['intensity']==(float(original['Intensity']) if original['Intensity'] else None)
        assert r['sequence_raw']==original['Sequence']
        assert r['type_raw']==original['Type']
    runs={r['Experiment']:r for r in tsv('txt/summary.txt') if r['Experiment']}
    beads=[e for e in runs if e.rsplit('_',2)[1]=='BEADS']
    typed={r['measurement_id']:r for r in rows('quantity_measurements')}
    assert len(typed)==362882
    counts=Counter();max_filter_error=0
    for pid,p in peptides.items():
        background=statistics.median(sums[pid,e] for e in beads)
        key=f'pxd009092:peptide:{pid}'
        assert typed[key+':beads_median']['value']==background
        for experiment in runs:
            m=typed[key+':run:'+experiment+':evidence_sum']; source=evidence[pid,experiment]
            n_numeric=sum(r['Intensity']!='' for r in source)
            status='no_reported_evidence' if not source else ('all_evidence_intensities_unreported' if n_numeric==0 else ('partial_evidence_intensity_sum' if n_numeric<len(source) else 'reported_evidence_sum'))
            assert m['value']==sums[pid,experiment] and m['value_status']==status
            counts[status]+=1
        for column in p:
            if not column.startswith('Intensity '):continue
            experiment=column[len('Intensity '):].replace('-MAPPbeadFilt','')
            raw_sum=sums[pid,experiment]
            expected=raw_sum if raw_sum>=2*background else 0
            value=float(p[column]);m=typed[key+':run:'+experiment+':bead_filtered']
            status='filtered_zero' if value==0 and raw_sum>0 else ('reported_zero_no_positive_evidence' if value==0 else 'reported_positive')
            assert m['value']==value and m['recomputed_value']==expected and m['value_status']==status
            assert math.isclose(value,expected,rel_tol=1e-4,abs_tol=1)
            counts[status]+=1;max_filter_error=max(max_filter_error,abs(value-expected))
    # Independently reconstruct every derivation edge from original row identities.
    expected_edges=set()
    for source in evidence.values():
        for r in source:
            key=f"pxd009092:peptide:{r['Peptide ID']}:run:{r['Experiment']}:evidence_sum"
            expected_edges.add((key,'pxd009092:evidence:'+r['id'],'source_evidence_intensity'))
    for pid,p in peptides.items():
        key=f'pxd009092:peptide:{pid}'
        for experiment in beads:
            expected_edges.add((key+':beads_median',key+':run:'+experiment+':evidence_sum','background_run'))
        for column in p:
            if column.startswith('Intensity '):
                e=column[len('Intensity '):].replace('-MAPPbeadFilt','');result=key+':run:'+e
                expected_edges.add((result+':bead_filtered',result+':evidence_sum','unfiltered_run_sum'))
                expected_edges.add((result+':bead_filtered',key+':beads_median','background_threshold'))
    source_cells={}
    for asset in ['publisher/suppdata2.xlsx','publisher/suppdata3.xlsx']:
        for sheet,cell,value in xml_cells(SOURCE/'assets'/asset):source_cells[f'{asset}:{sheet}:{cell}']=value
    parsed_cells=list(rows('publisher_cells'));assert len(parsed_cells)==len(source_cells)
    for r in parsed_cells:
        value=source_cells[r['cell_id']]
        if isinstance(value,float):assert value==r['value_number']
        else:assert value==r['value_text']
    numeric=Counter()
    for r in rows('publisher_quantities'):
        value=source_cells.get(r['source_cell']);assert value==r['value']
        if r['condition'] is None:
            assert r['recomputed_value'] is None
            assert r['representation']=='imputed_header_conflict'
        else:
            pid=r['peptide_row_id'].rsplit(':',1)[1]
            condition=r['condition'];experiments=[e for e in runs if e.rsplit('_',2)[1]==condition]
            for e in experiments:
                expected_edges.add((r['measurement_id'],r['peptide_row_id']+':run:'+e+':bead_filtered','median_run'))
            if value is not None:
                vals=[typed[r['peptide_row_id']+':run:'+e+':bead_filtered']['value'] for e in experiments]
                assert math.isclose(value,math.log10(statistics.median(vals)),rel_tol=1e-10,abs_tol=1e-5)
                numeric[condition]+=1
    edges=[(r['derived_measurement_id'],r['input_measurement_id'],r['input_role']) for r in rows('derivation_inputs')]
    assert len(edges)==len(set(edges)) and set(edges)==expected_edges
    result={'passed':True,'seconds':time.monotonic()-tick,'n_package_files':len(manifest['files']),
        'n_original_rows_replayed':n_raw,'n_quantities_replayed':len(typed),'n_missing_evidence_PEP':missing_confidence,
        'quantity_status_counts':dict(counts),'n_publisher_cells_replayed':len(source_cells),
        'publisher_measured_numeric':dict(numeric),'n_derivation_edges_replayed':len(edges),
        'maximum_absolute_filter_replay_error':max_filter_error,
        'interpretation':'source ingestion verification; reference joins, eligibility and biological fitting remain pending'}
    (FAMILY/'results/independent-verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
