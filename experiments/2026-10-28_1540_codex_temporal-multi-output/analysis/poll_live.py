"""Independent lightweight status polling; never waits on artifact archives."""
from pathlib import Path
import functools,http.server,json,re,threading,time
import modal

F=Path(__file__).resolve().parents[1]
FID=F.name
D=modal.Dict.from_name('presto-cross-class-live')
server=http.server.ThreadingHTTPServer(('127.0.0.1',54502),functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(F)))
threading.Thread(target=server.serve_forever,daemon=True).start()

def read(p,default=None):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return default

while True:
 state={'family':FID,'refreshed_unix':time.time(),'source_files':{},'runs':[],'preparation':None,'poll_error':None}
 acquisition=read(F/'results/acquisition.json',{})
 state['source_files']={k:{q:v[q] for q in ('size_bytes','sha256')} for k,v in acquisition.get('sources',{}).items()}
 dates=read(F/'results/pubmed-2026-dates.json',{})
 state['verified_2026_studies']=len(dates.get('records',[]))
 try:
  p=max((F/'launch_logs').glob('build-attempt*.log'),key=lambda p:p.stat().st_mtime)
  with p.open('rb') as stream:
   stream.seek(max(0,p.stat().st_size-100000));tail=stream.read().decode(errors='replace')
  matches=re.findall(r'Scanning (iedb|cedar):\s*(\d+)%',tail)
  state['source_parse']={'iedb':100 if any(k=='cedar' for k,v in matches) else 0,'cedar':0}
  for name,percent in matches:state['source_parse'][name]=int(percent)
  if 'Published complete dataset:' in tail:state['source_parse']={'iedb':100,'cedar':100}
 except FileNotFoundError:state['source_parse']={}
 try:
  state['preparation']=D.get(FID+':preparation')
  for name in ('wide-multi-s42','wide-multi-s43'):
   value=D.get(FID+':'+name)
   if value:state['runs'].append(value)
 except Exception as exc:state['poll_error']=repr(exc)
 state['admission']=read(F/'results/admission.json',{})
 state['build']=read(F/'results/build.json',{})
 if state['build'].get('status')=='built' and state['preparation'] and state['preparation'].get('status')=='source_index_retry':
  state['preparation']={'status':'admission_starting','stage':'Source build complete; starting canonical admission','counts':{},'updated_unix':state['build']['updated_unix']}
 path=F/'results/live.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,allow_nan=False)+'\n');temp.replace(path)
 time.sleep(20)
