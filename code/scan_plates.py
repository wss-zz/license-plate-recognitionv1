# -*- coding: utf-8 -*-
"""Run the project plate detector over every unique collected image.

Pass 1: imgsz=640 over all images.
Pass 2: imgsz=1280 for images where pass 1 found no plate.
Writes data/clip_scene/det_scan.csv with boxes, plate count, largest-plate area ratio,
plus image-type (crop / closeup / scene / unknown) and classical quality metrics
measured on the detected plate crop (fallback: whole image).
"""
import csv
import time
from pathlib import Path

import cv2
import numpy as np

import dataset_index as di

WEIGHTS = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05\models\det\yolo12n_mix\weights\best.pt")
OUT = di.OUT


def imread_u(p):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
    except Exception:
        return None


def quality(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    if max(h, w) < 48:  # too tiny, upscale
        g = cv2.resize(g, (max(64, w * 4), max(32, h * 4)), interpolation=cv2.INTER_CUBIC)
        h, w = g.shape
    g = cv2.resize(g, (256, 64), interpolation=cv2.INTER_AREA)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    return lap, float(g.mean()), float((g >= 250).mean()), float((g < 25).mean())


def hsv_color(bgr, box):
    """Dominant plate colour from the detected plate region (circular hue of saturated pixels)."""
    h, w = bgr.shape[:2]
    c = bgr
    if box is not None and len(box) == 4:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - 0.04 * bw)); x2 = min(w, int(x2 + 0.04 * bw))
        y1 = max(0, int(y1 - 0.10 * bh)); y2 = min(h, int(y2 + 0.10 * bh))
        if x2 - x1 > 10 and y2 - y1 > 4:
            c = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.float32); S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    m = (S >= 70) & (V >= 40)
    if float(m.mean()) >= 0.15:
        ang = H[m] * 2 * np.pi / 180.0
        hue = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) * 90 / np.pi
        hue = hue % 180
        if 96 <= hue <= 145:
            return "蓝牌"
        if 41 <= hue < 96:
            return "绿牌"
        if 10 <= hue < 41:
            return "黄牌"
        return "其他"
    vmed = float(np.median(V))
    return "白牌" if vmed >= 130 else ("黑牌" if vmed < 95 else "其他")


def plate_crop(bgr, box, pad=0.12):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw)); y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw)); y2 = min(h, int(y2 + pad * bh))
    if x2 - x1 < 8 or y2 - y1 < 4:
        return bgr
    return bgr[y1:y2, x1:x2]


def predict(model, pairs, imgsz, conf):
    res = model.predict([a for _, a in pairs], imgsz=imgsz, conf=conf, verbose=False, device=0)
    out = []
    for (b, a), r in zip(pairs, res):
        bx = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
        cf = r.boxes.conf.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0,))
        out.append((b, a, bx, cf))
    return out


def make_row(b, a, bx, cf, pass_no):
    h, w = a.shape[:2]
    ratios = [float((x2 - x1) * (y2 - y1)) / float(w * h) for x1, y1, x2, y2 in bx]
    if not ratios:
        typ, maxr, crop = "unknown", 0.0, a
    else:
        j = int(np.argmax(ratios))
        maxr = ratios[j]
        typ = "crop" if maxr >= 0.45 else ("closeup" if maxr >= 0.12 else "scene")
        crop = plate_crop(a, bx[j])
    lap, lum, over, under = quality(crop)
    top_box = bx[int(np.argmax(ratios))] if ratios else None
    return {
        "idx": b["idx"], "md5": b["md5"], "tag": b["tag"], "sub": b["sub"], "name": b["name"],
        "path": b["path"], "w": w, "h": h, "n_plate": len(ratios),
        "max_ratio": round(maxr, 5), "type": typ, "pass": pass_no,
        "hsv_color": hsv_color(a, top_box),
        "min_conf": round(float(cf.min()), 3) if len(cf) else "",
        "max_conf": round(float(cf.max()), 3) if len(cf) else "",
        "ratios": ";".join(f"{r:.4f}" for r in sorted(ratios, reverse=True)[:5]),
        "box": ";".join(f"{int(round(v))}" for v in bx[int(np.argmax(ratios))]) if ratios else "",
        "lap": round(lap, 1), "lum": round(lum, 1), "over": round(over, 4), "under": round(under, 4),
    }


def main(batch=16):
    from ultralytics import YOLO
    items, _ = di.build_index()
    model = YOLO(str(WEIGHTS))
    rows, t0 = [], time.time()

    buf, pending = [], []
    for it in items:
        buf.append(it)
        if len(buf) >= batch:
            pending.append(buf); buf = []
    if buf:
        pending.append(buf)

    for pi, chunk in enumerate(pending):
        pairs = [(it, imread_u(it["path"])) for it in chunk]
        pairs = [(b, a) for b, a in pairs if a is not None]
        if not pairs:
            continue
        for b, a, bx, cf in predict(model, pairs, 640, 0.25):
            rows.append(make_row(b, a, bx, cf, 1))
        if pi % 8 == 0:
            print(f"[pass1] {len(rows)}/{len(items)} {time.time()-t0:.1f}s", flush=True)

    todo = [r for r in rows if r["n_plate"] == 0]
    print(f"[pass2] {len(todo)} images at imgsz=1280")
    for s in range(0, len(todo), 8):
        chunk = todo[s:s + 8]
        pairs = [(r, imread_u(r["path"])) for r in chunk]
        pairs = [(b, a) for b, a in pairs if a is not None]
        if not pairs:
            continue
        for b, a, bx, cf in predict(model, pairs, 1280, 0.12):
            rows[b["idx"]] = make_row(b, a, bx, cf, 2)

    seen = {r["idx"] for r in rows}
    for it in items:
        if it["idx"] not in seen:
            rows.append({"idx": it["idx"], "md5": it["md5"], "tag": it["tag"], "sub": it["sub"],
                         "name": it["name"], "path": it["path"], "w": 0, "h": 0, "n_plate": 0,
                         "max_ratio": 0.0, "type": "read_fail", "pass": 0, "min_conf": "", "max_conf": "",
                         "ratios": "", "box": "", "hsv_color": "其他", "lap": 0, "lum": 0, "over": 0, "under": 0})
    rows.sort(key=lambda r: r["idx"])
    with open(OUT / "det_scan.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    from collections import Counter
    print(Counter(r["type"] for r in rows), Counter(r["pass"] for r in rows))
    print("plates total:", sum(r["n_plate"] for r in rows),
          "| >=2 plates:", sum(1 for r in rows if r["n_plate"] >= 2))
    for key in ("lap", "lum", "over", "under"):
        v = np.array([r[key] for r in rows], dtype=float)
        print(f"{key}: p5={np.percentile(v,5):.1f} p10={np.percentile(v,10):.1f} med={np.median(v):.1f} "
              f"p75={np.percentile(v,75):.1f}")
    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

