# -*- coding: utf-8 -*-
"""检测器在三套测试集上的评测：自采289 / 合并342(含特殊) / CCPD1000。"""
import argparse, json, time
from pathlib import Path
import cv2, numpy as np, torch

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")


def load_yolo(w):
    from ultralytics import YOLO
    return YOLO(str(w))


def gt_from_quads(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    g = {}
    for r in d:
        f = r.get("file") or r.get("path")
        q = np.array(r["quad"] if "quad" in r else r["points"], np.float32)
        g.setdefault(f, []).append(q)
    return g


def bbox(q):
    x1, y1 = q[:, 0].min(), q[:, 1].min(); x2, y2 = q[:, 0].max(), q[:, 1].max()
    return np.array([x1, y1, x2, y2], np.float32)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1); inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate(model, gt, conf=0.25, iou_thr=0.7):
    tp = fp = fn = 0; ious = []; t0 = time.time(); n = 0
    for f, quads in gt.items():
        img = cv2.imdecode(np.fromfile(f, np.uint8), 1)
        if img is None:
            continue
        n += 1
        r = model.predict(img, conf=conf, iou=iou_thr, imgsz=640, verbose=False)[0]
        pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4), np.float32)
        gb = [bbox(q) for q in quads]
        used = set()
        for g in gb:
            best, bi = 0.0, -1
            for j, p in enumerate(pb):
                if j in used:
                    continue
                v = iou(g, p)
                if v > best:
                    best, bi = v, j
            if best >= 0.5:
                tp += 1; used.add(bi); ious.append(best)
            else:
                fn += 1
        fp += max(0, len(pb) - len(used))
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    F1 = 2 * P * R / max(1e-6, P + R)
    return dict(n_img=n, tp=tp, fp=fp, fn=fn, P=P * 100, R=R * 100, F1=F1 * 100,
                mIoU=float(np.mean(ious) * 100) if ious else 0.0, sec=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="self289,all342,ccpd1000")
    a = ap.parse_args()
    m = load_yolo(a.weights)
    out = {}
    if "self289" in a.sets:
        out["self289"] = evaluate(m, gt_from_quads(ROOT / "data" / "quads_manual.json"))
    if "all342" in a.sets:
        out["all342"] = evaluate(m, gt_from_quads(ROOT / "data" / "quads_all.json"))
        out["special76"] = evaluate(m, gt_from_quads(ROOT / "data" / "quads_special.json"))
    if "ccpd1000" in a.sets:
        dm = json.loads((ROOT / "data" / "ccpd_eval" / "det_manifest.json").read_text(encoding="utf-8"))
        gt = {}
        for r in dm:
            gt.setdefault(r.get("file") or r["path"], []).append(np.array(r["quad"], np.float32))
        out["ccpd1000"] = evaluate(m, gt)
    for k, v in out.items():
        print(f"{a.tag:20s} {k:9s} n={v['n_img']:4d} P={v['P']:6.2f} R={v['R']:6.2f} F1={v['F1']:6.2f} mIoU={v['mIoU']:5.2f}")
    p = ROOT / "deploy" / f"eval_det_{a.tag}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
