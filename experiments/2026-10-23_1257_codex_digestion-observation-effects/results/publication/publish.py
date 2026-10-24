"""Fresh compare-and-patch body updates with state-preserving readback."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def get(path):return json.loads(subprocess.check_output(["gh","api",path],text=True))
for e in json.loads((ROOT/"updates.json").read_text()):
 path=f"repos/{e['repository']}/{e['kind']}/{e['number']}";before=get(path);body=before["body"] or ""
 assert e["marker"] not in body
 prefix=e["repository"].split("/")[-1]+"-"+str(e["number"])
 (ROOT/(prefix+"-before.json")).write_text(json.dumps(before,indent=2)+"\n")
 for old,new in e["replacements"]:
  assert body.count(old)==1,(prefix,old)
  body=body.replace(old,new)
 body=body.rstrip()+e["section"]
 request=ROOT/(prefix+"-request.json");request.write_text(json.dumps({"body":body},indent=2)+"\n")
 fresh=get(path);assert fresh["body"]==before["body"] and fresh["updated_at"]==before["updated_at"]
 subprocess.run(["gh","api","-X","PATCH",path,"--input",str(request)],check=True,stdout=subprocess.DEVNULL)
 after=get(path);assert after["body"]==body and after["state"]==before["state"]
 if e["kind"]=="pulls":assert after["draft"]==before["draft"]==True
 (ROOT/(prefix+"-readback.json")).write_text(json.dumps(after,indent=2)+"\n")
 print(prefix,"body and state verified",flush=True)
