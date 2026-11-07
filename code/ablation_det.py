# -*- coding: utf-8 -*-
"""检测器消融：在同一批人工四点标注图上比 AP50 / AP95 / mIoU / P-R-F1 / 延迟 / 体积。

GT 取人工四点（`data/quads_manual.json`）的外接矩形，IoU≥0.5 记为命中。
AP 按单类 VOC 全点插值计算；mIoU 取 IoU≥0.5 的匹配框的平均 IoU。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
QUADS = json.loads((ROOT / "data" / "quads_manual.json").read_text(encoding="utf-8"))

DETS = {
    "YOLO12n(mix)":      ROOT / "models/det/yolo12n_mix/weights/best.pt",
    "YOLO12n(hard)":     ROOT / "models/det/yolo12n_hard/weights/best.pt",
    "YOLO12n(small)":    ROOT / "models/det/yolo12n_small/weights/best.pt",
    "YOLO11n":           ROOT / "models/det/yolo11n_hard/weights/best.pt",
    "YOLOv8n":           ROOT / "models/det/yolov8n_hard/weights/best.pt",
    "YOLO-ULM":          ROOT / "models/det/ulm_persp/weights/best.pt",
    "YOLO-ULM(微调)":     ROOT / "models/det/yolo12n_ulm_mix/weights/best.pt",
}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_boxes(item):
    q = np.array(item["quad"], dtype=float)
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def read_img(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def ap_at(dets, gts, thr):
    """单类 AP：dets=[(img_idx, score, box)]，gts={img: [box,...]}"""
    total_gt = sum(len(v) for v in gts.values())
    if not dets or total_gt == 0:
        return 0.0
    dets = sorted(dets, key=lambda t: -t[1])
    used = {k: [False] * len(v) for k, v in gts.items()}
    tp, fp = [], []
    for idx, _, box in dets:
        best, bj = 0.0, -1
        for j, g in enumerate(gts[idx]):
            v = iou(box, g)
            if v > best:
                best, bj = v, j
        if best >= thr and bj >= 0 and not used[idx][bj]:
            used[idx][bj] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    rec = tp_c / total_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]) * 100)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔的模型名，留空则全跑")
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    todo = {k: v for k, v in DETS.items() if (not only or k in only) and Path(v).exists()}
    DETS.clear() if False else None
    paths = [q["path"] if "path" in q else str(Path(q["file"])) for q in QUADS]
    # 规范化路径：quads_manual 里可能只有文件名
    for i, q in enumerate(QUADS):
        p = Path(q.get("path") or q.get("file"))
        if not p.exists():
            cand = Path(r"D:\车牌图片") / p.name
            if cand.exists():
                paths[i] = str(cand)
    gts = {i: [gt_boxes(q)] for i, q in enumerate(QUADS)}
    imgs = [read_img(p) for p in paths]
    ok = 0
    for i, im in enumerate(imgs):
        if im is None:
            imgs[i] = np.zeros((64, 64, 3), np.uint8)
        else:
            ok += 1
    print(f"图像 {len(imgs)} 张，可读 {ok}")

    rows = []
    for name, w in todo.items():
        if not w.exists():
            print(f"[skip] {name}: 权重不存在")
            continue
        m = YOLO(str(w))
        nparam = sum(p.numel() for p in m.model.parameters()) / 1e6
        size = w.stat().st_size / 2 ** 20
        dets = []
        miou_sum, miou_n = 0.0, 0
        t0 = time.time()
        for i, im in enumerate(imgs):
            r = m.predict(im, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
            for b in r.boxes:
                box = [float(v) for v in b.xyxy[0].tolist()]
                sc = float(b.conf[0])
                dets.append((i, sc, box))
                v = iou(box, gts[i][0])
                if v >= 0.5:
                    miou_sum += v; miou_n += 1
        infer_s = time.time() - t0
        ap50 = ap_at(dets, gts, 0.5)
        ap75 = ap_at(dets, gts, 0.75)
        ap95 = ap_at(dets, gts, 0.95)
        # P/R/F1 @0.5
        dets_s = sorted(dets, key=lambda t: -t[1])
        used = {i: False for i in gts}
        tp = fp = 0
        for idx, sc, box in dets_s:
            if iou(box, gts[idx][0]) >= 0.5 and not used[idx]:
                used[idx] = True; tp += 1
            else:
                fp += 1
        fn = len(gts) - tp
        P = tp / max(tp + fp, 1) * 100
        R = tp / max(tp + fn, 1) * 100
        F1 = 2 * P * R / max(P + R, 1e-9)
        # 延迟：单图 640 前向
        im0 = imgs[0]
        for _ in range(3):
            m.predict(im0, imgsz=640, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        N = 20
        for _ in range(N):
            m.predict(im0, imgsz=640, conf=0.25, iou=0.7, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat = (time.time() - t0) / N * 1000
        rows.append(dict(name=name, ap50=ap50, ap75=ap75, ap95=ap95,
                         miou=(miou_sum / miou_n * 100 if miou_n else 0),
                         P=P, R=R, F1=F1, ms=lat, mb=size, mparam=nparam,
                         boxes=len(dets), batch_s=infer_s))
        print(f"{name:16s} AP50 {ap50:5.2f} AP95 {ap95:5.2f} mIoU {rows[-1]['miou']:5.2f} "
              f"P {P:5.2f} R {R:5.2f} F1 {F1:5.2f} {lat:6.1f}ms {size:5.2f}MB {nparam:5.2f}M")

    out = ROOT / "deploy" / "ablation_det.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
