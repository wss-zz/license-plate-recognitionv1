# -*- coding: utf-8 -*-
"""把合并真值（自采 289 + 特殊车牌 76）warp 成 192x64 裁剪图，供识别评测。"""
import json
from pathlib import Path
import cv2, numpy as np

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
OUT = ROOT/"data/crops_all"
items = json.loads((ROOT/"data/quads_all.json").read_text(encoding="utf-8"))
OUT.mkdir(parents=True, exist_ok=True)

def read(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def warp(img, quad, w=192, h=64):
    q=np.array(quad,np.float32); s=q.sum(1); d=np.diff(q,axis=1).ravel()
    r=np.zeros((4,2),np.float32); r[0]=q[np.argmin(s)]; r[2]=q[np.argmax(s)]; r[1]=q[np.argmin(d)]; r[3]=q[np.argmax(d)]
    dst=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(r,dst),(w,h))

man=[]
for k,it in enumerate(items):
    img=read(it["file"])
    if img is None: continue
    crop=warp(img,it["quad"])
    name=f"{k:04d}.jpg"
    cv2.imencode(".jpg",crop)[1].tofile(str(OUT/name))
    t=it["text"]
    color = "green" if len(t)==8 and not any(c in t for c in "使领港澳警学挂") else ("green" if len(t)==8 and ("港" in t or "澳" in t) else "blue")
    man.append({"path":str(OUT/name),"text":t,"color":color,"source":it["source"],"name":it["name"]})
(OUT/"manifest.json").write_text(json.dumps(man,ensure_ascii=False,indent=1),encoding="utf-8")
print("crops",len(man))
from collections import Counter
print(Counter(m["source"] for m in man), Counter(m["color"] for m in man))
