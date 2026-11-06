import json, sys
sys.path.insert(0, "engine")
from test import eng_lines, digests
old = digests(eng_lines(open(sys.argv[1]).read()), "pie2")
new = digests(eng_lines(open(sys.argv[2]).read()), sys.argv[3] if len(sys.argv) > 3 else "pie2")
print("identical to v1" if old == new else "DIFFERENT: " + json.dumps({k: (old.get(k), new.get(k)) for k in old if old.get(k) != new.get(k)}))
