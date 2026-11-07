# -*- coding: utf-8 -*-
"""检测器在 1000 张 CCPD base（已排除训练图）上的 P/R/F1 与 AP。"""
import json, time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, r"D:\desktop\复现代码文件夹\yolo-ulm")
from ultralytics import YOLO

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
DATA = ROOT/"data"/"ccpd_eval"
DET = ROOT/"models/det/yolo12n_mix/weights/best.pt"

def iou(a, b):
    x1,y1 = max(a[0],b[0]), max(a[1],b[1])
    x2,y2 = min(a[2],b[2]), min(a[3],b[3])
    iw, ih = max(0.0,x2-x1), max(0.0,y2-y1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def main():
    items = json.loads((DATA/"det_manifest.json").read_text(encoding="utf-8"))
    yolo = YOLO(str(DET))
    tp=fp=fn=0; prec_list=[]; rec_list=[]; ious=[]; t0=time.time(); ns=nt=0
    for it in items:
        gt = it["bbox"]
        r = yolo.predict(it["file"], imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
        bx = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
        order = np.argsort(-cf)
        used=set(); hit=False
        for i in order:
            if iou(bx[i], gt) >= 0.5 and not hit:
                hit=True; used.add(i); tp+=1; ns+=1
                ious.append(iou(bx[i], gt))
        if not hit: fn+=1
        fp += len(bx)-len(used); nt += len(bx)
    P = tp/max(1,nt); R = tp/max(1,tp+fn); F1 = 2*P*R/max(1e-9,P+R)
    res = {"n":len(items),"TP":tp,"FP":fp,"FN":fn,"P":round(P*100,2),"R":round(R*100,2),"F1":round(F1*100,2),
           "meanIoU":round(float(np.mean(ious))*100,2) if ious else None,"seconds":round(time.time()-t0,1)}
    print(json.dumps(res, ensure_ascii=False))
    (ROOT/"deploy"/"eval_ccpd_det.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
