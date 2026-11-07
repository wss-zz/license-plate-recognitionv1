import json
import store

with store.conn() as c:
    rows = [dict(r) for r in c.execute(
        "SELECT id,plate,entry_time,exit_time,fee,status FROM session ORDER BY id DESC LIMIT 8")]
for r in rows:
    print(r)
with store.conn() as c:
    d = [dict(r) for r in c.execute("SELECT plate,detail FROM bill ORDER BY id DESC LIMIT 3")]
for x in d:
    det = json.loads(x["detail"])
    print(x["plate"], "parked", det["parked_min"], "fee", det["fee"])
    for s in det["segments"]:
        print("   ", s)
