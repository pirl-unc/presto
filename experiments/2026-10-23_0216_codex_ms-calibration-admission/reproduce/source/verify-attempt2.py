"""Independent XML values, panel identities, source amounts and curve-grain audit."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,math,re,time,xml.etree.ElementTree as E,zipfile
R=Path(__file__).resolve().parents[1];A=Path('/private/tmp/hitlist-pxd004886-assets')
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def shared_formula_shift(text, origin, dest):
    def index(label):
        n=0
        for c in label:n=n*26+ord(c)-64
        return n
    def label(n):
        chars=[]
        while n:n,d=divmod(n-1,26);chars.append(chr(65+d))
        return ''.join(reversed(chars))
    ao=re.fullmatch(r'([A-Z]+)([0-9]+)',origin);ad=re.fullmatch(r'([A-Z]+)([0-9]+)',dest)
    dc=index(ad[1])-index(ao[1]);dr=int(ad[2])-int(ao[2])
    def shift(match):
        ca,c,ra,row=match.groups()
        return ca+(c if ca else label(index(c)+dc))+ra+str(int(row) if ra else int(row)+dr)
    pieces=re.split(r'("(?:[^"]|"")*")',text)
    for i in range(0,len(pieces),2):pieces[i]=re.sub(r'(?<![A-Za-z0-9_])(\$?)([A-Z]{1,3})(\$?)([0-9]+)(?![A-Za-z0-9_])',shift,pieces[i])
    return ''.join(pieces)

def read_xml(path):
    result={};n=0
    with zipfile.ZipFile(path) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            for si in E.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS):shared.append(''.join(t.text or '' for t in si.iter() if t.tag=='{'+NS['s']+'}t'))
        rel={x.attrib['Id']:x.attrib['Target'] for x in E.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        wb=E.fromstring(z.read('xl/workbook.xml'))
        for s in wb.findall('s:sheets/s:sheet',NS):
            target=rel[s.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']];target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
            cells={}
            nodes=E.fromstring(z.read(target)).findall('.//s:sheetData/s:row/s:c',NS)
            shared_bases={}
            for node in nodes:
                f=node.find('s:f',NS)
                if f is not None and f.attrib.get('t')=='shared' and f.text is not None:shared_bases[f.attrib['si']]=(node.attrib['r'],f.text)
            for c in nodes:
                v=c.find('s:v',NS);form=c.find('s:f',NS);inline=c.find('s:is',NS)
                if v is None and form is None and inline is None:continue
                typ=c.attrib.get('t','n');val=None
                if typ=='s':val=shared[int(v.text)]
                elif typ=='inlineStr':val=''.join(t.text or '' for t in inline.iter() if t.tag=='{'+NS['s']+'}t')
                elif v is not None and v.text is not None:
                    val=float(v.text) if typ=='n' else bool(int(v.text)) if typ=='b' else v.text
                formula=form.text if form is not None else None
                if form is not None and form.attrib.get('t')=='shared' and formula is None:
                    origin,base=shared_bases[form.attrib['si']];formula=shared_formula_shift(base,origin,c.attrib['r'])
                cells[c.attrib['r']]={'value':val,'formula':formula,'xml_type':typ}
            result[s.attrib['name']]=cells;n+=len(cells)
    return result,n

def canonical(label,panel=False):
    # Retain the original reporter/form; this only audits potential panel membership.
    part=label if panel else label.split('.',1)[1]
    part=re.sub(r'\.sum[0-9]+$','',part)
    return re.sub(r'\[[^]]+\]','',part)
def numeric(v):return isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v)
def colno(s):
    n=0
    for c in s:n=n*26+ord(c)-64
    return n

def main():
    start=time.monotonic();books={};counts={};checked=0
    inv=json.loads((R/'results/workbook-inventory-complete.json').read_text())
    for key in ['1','2','4','8']:
        sheets,n=read_xml(A/f'supplementary-data-{key}.xlsx');counts[key]=n
        records=[json.loads(x) for x in gzip.decompress((R/'results'/f'supplementary-data-{key}-cells.jsonl.gz').read_bytes()).splitlines()]
        assert len(records)==n==inv[key]['n_source_cells']
        for c in records:
            x=sheets[c['sheet']][c['coordinate']];v=c['value']
            assert x['value']==v,(key,c['coordinate'],x['value'],v)
            if c['data_type']=='f':assert c['formula_or_value']=='='+x['formula']
            checked+=1
        assert len(sheets)==1
        books[key]=next(iter(sheets.values()))
    def v(k,c):return books[k].get(c,{}).get('value')
    panel={}
    for row in range(5,35):
        label=v('2',f'A{row}');seq=canonical(label,True)
        assert re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+',seq)
        assert seq not in panel
        panel[seq]={'label':label,'group':v('2',f'C{row}'),'source_cell':f'A{row}'}
    assert collections.Counter(x['group'] for x in panel.values())==dict.fromkeys('ABCDE',6)
    amounts={};max_dilution_error=0.
    for row in range(6,11):
        group=v('1',f'A{row}');vals=[]
        for col in 'BCDEF':
            sample=v('1',f'{col}5');amount=v('1',f'{col}{row}');amounts[(group,sample)]={'amount_fmol':amount,'cell':f'{col}{row}'};vals.append(amount)
        for a,b in zip(vals,vals[1:]):max_dilution_error=max(max_dilution_error,abs(a/b-3))
    assert max_dilution_error<1e-12
    # Original report identities: never collapse chemically different channels.
    four=books['4'];raw4={};norm4={};mode=None
    for row in range(1,75):
        label=v('4',f'A{row}')
        if isinstance(label,str) and label.strip().lower() in ['un-normalized','normalized']:mode=label.strip().lower()
        if isinstance(label,str) and '.' in label and mode:
            seq=canonical(label);assert seq in panel,(row,label)
            record={'row':row,'sequence':seq,'source_label':label,'group':panel[seq]['group'],'nominal_S4_fmol':amounts[(panel[seq]['group'],'S4')]['amount_fmol'],'reported_amount':v('4',f'B{row}')}
            (raw4 if mode=='un-normalized' else norm4)[label]=record
    assert set(raw4)==set(norm4)
    raw_slots=[];norm_slots=[]
    for coord,c in four.items():
        match=re.fullmatch(r'([A-Z]+)([0-9]+)',coord);col,row=colno(match[1]),int(match[2])
        if 6<=col<=104:
            for report,slots in [(raw4,raw_slots),(norm4,norm_slots)]:
                if any(x['row']==row for x in report.values()):slots.append((coord,col,c['value']))
    def stats(slots,report):
        return {'expected_slots':len(report)*99,'populated_source_cells':len(slots),'positive_values':sum(numeric(x[2]) and x[2]>0 for x in slots),'explicit_empty_strings':sum(x[2]=='' for x in slots),'zero_values':sum(numeric(x[2]) and x[2]==0 for x in slots),'blank_or_absent_slots':len(report)*99-sum(numeric(x[2]) for x in slots),'columns_without_positive_response':[i for i in range(6,105) if not any(c==i and numeric(x) and x>0 for _,c,x in slots)]}
    # Inspect source group-specific response grid, retaining its source header separately.
    curve_rows=[];site=None;mode=None;header_row=None
    for row in range(1,872):
        label=v('8',f'A{row}')
        if isinstance(label,str) and re.fullmatch(r'Site [0-9]+',label.strip()):site=int(label.split()[-1])
        if isinstance(label,str) and label.strip().lower() in ['un-normalized','normalized']:
            mode=label.strip().lower();header_row=row+1
        group=v('8',f'B{row}')
        if isinstance(label,str) and '.' in label and isinstance(group,str) and group in set('ABCDE'):
            seq=canonical(label);assert seq in panel and panel[seq]['group']==group,(row,label,group)
            assert site and mode and header_row
            curve_rows.append({'row':row,'site':site,'mode':mode,'header_row':header_row,'source_label':label,'sequence':seq,'group':group})
    assert len(curve_rows)==len(raw4)*22,(len(curve_rows),len(raw4))
    source_header_disagreements=[];points=[]
    for record in curve_rows:
        start_col=3+2*('ABCDE'.index(record['group']))
        for j in range(5):
            col=chr(64+start_col+j);coord=f'{col}{record["row"]}';sample=f'S{j+1}';declared=amounts[(record['group'],sample)]
            header=v('8',f'{col}{record["header_row"]}');expected=declared['amount_fmol'];response=v('8',coord)
            assert numeric(header)
            rel=abs(header-expected)/expected
            point={**record,'cell':coord,'sample_by_group_position':sample,'reported_grid_amount':header,'design_amount_fmol':expected,'design_source_cell':declared['cell'],'relative_amount_disagreement':rel,'response':response,'amount_join_status':'group_order_hypothesis_requires_source_reconciliation'}
            points.append(point)
            if rel>0.05:source_header_disagreements.append(point)
    # Distinct source row annotations and amount disagreements are evidence; no training labels.
    (R/'results/source-join-audit.json').write_text(json.dumps({'panel':panel,'sample_amounts':{f'{g}/{s}':x for (g,s),x in amounts.items()},'S4_reporters':list(raw4.values()),'curve_rows':curve_rows,'curve_point_audit':points},indent=2)+'\n')
    report={'passed':True,'independent_xml_value_and_formula_cells':checked,'cells_per_supplement':counts,'n_panel_sequences':len(panel),'group_counts':dict(collections.Counter(x['group'] for x in panel.values())),'max_threefold_ratio_error':max_dilution_error,'S4_reporter_rows_per_representation':len(raw4),'S4_canonical_sequences':len(set(x['sequence'] for x in raw4.values())),'S4_raw':stats(raw_slots,raw4),'S4_normalized':stats(norm_slots,norm4),'curve_reporter_rows':len(curve_rows),'curve_expected_reporter_slots':len(points),'curve_positive_values':sum(numeric(p['response']) and p['response']>0 for p in points),'curve_grid_disagreements_over5percent':len(source_header_disagreements),'maximum_grid_relative_disagreement':max(x['relative_amount_disagreement'] for x in points),'supplement8_header':v('8','A1'),'reporter_multiplicity':{s:[x['source_label'] for x in raw4.values() if x['sequence']==s] for s in panel if sum(x['sequence']==s for x in raw4.values())>1},'wall_seconds':time.monotonic()-start,'admission_status':'not_admitted; reconcile curve amounts and acquisition/selection metadata before shared recipe or fitting'}
    (R/'results/independent-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
