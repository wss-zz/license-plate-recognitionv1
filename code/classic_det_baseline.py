# -*- coding: utf-8 -*-
"""传统车牌定位基线：颜色空间 / 边缘+形态学 / 投影法 / Sobel+二值化。
在自采 289 上按与深度学习检测器相同的 IoU 0.5 判据评测。"""
import json, time
from pathlib import Path
import cv2, numpy as np

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
GT = json.loads((ROOT / "data" / "quads_manual.json").read_text(encoding="utf-8"))

K_CLOSE = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
K_CLOSE2 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))


def filt(cnt, W, H):
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area < 0.0006 * W * H or area > 0.25 * W * H:
        return None
    ar = w / max(1, h)
    if not (1.8 <= ar <= 8.0):
        return None
    return (x, y, x + w, y + h)


def color_locate(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.zeros(h.shape, np.uint8)
    m[((h >= 100) & (h <= 140) & (s > 60) & (v > 40))] = 255
    m[((h >= 35) & (h <= 95) & (s > 60) & (v > 40))] = 255
    m[((h >= 10) & (h <= 35) & (s > 80) & (v > 60))] = 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K_CLOSE)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img.shape[:2]
    boxes = [b for b in (filt(c, W, H) for c in cnts) if b]
    return sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:3]


def edge_locate(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 60, 180)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, K_CLOSE2)
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img.shape[:2]
    boxes = [b for b in (filt(c, W, H) for c in cnts) if b]
    return sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:3]


def sobel_locate(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.convertScaleAbs(cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=3))
    sx = cv2.GaussianBlur(sx, (3, 3), 0)
    _, bw = cv2.threshold(sx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, K_CLOSE2)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img.shape[:2]
    boxes = [b for b in (filt(c, W, H) for c in cnts) if b]
    return sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:3]


def proj_locate(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gx = cv2.GaussianBlur(gx, (5, 5), 0)
    gx = (gx > gx.mean() + 0.8 * gx.std()).astype(np.uint8)
    rows = gx.sum(1).astype(np.float32); cols = gx.sum(0).astype(np.float32)
    out = []
    if rows.max() <= 0:
        return out
    thr = rows.mean() + 1.2 * rows.std()
    inband = rows > thr
    y = 0
    while y < len(inband):
        if inband[y]:
            y0 = y
            while y < len(inband) and inband[y]:
                y += 1
            y1 = y
            band = gx[y0:y1]
            if band.size:
                c = band.sum(0); cthr = c.mean() + 1.0 * c.std()
                xs = np.where(c > cthr)[0]
                if len(xs):
                    x0, x1 = int(xs.min()), int(xs.max())
                    h, w = y1 - y0, x1 - x0
                    if h >= 8 and w >= 2.0 * h and w <= 8.0 * h:
                        out.append((x0, y0, x1, y1))
        else:
            y += 1
    return sorted(out, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:3]


METH = {"颜色空间": color_locate, "边缘+形态学": edge_locate, "Sobel+二值化": sobel_locate, "投影法": proj_locate}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1); inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def main():
    res = {}
    for name, fn in METH.items():
        for topk in (3, 1):
            tp = fp = fnm = 0
            ious = []
            t0 = time.time()
            for r in GT:
                img = cv2.imdecode(np.fromfile(r["file"], np.uint8), 1)
                if img is None:
                    continue
                q = np.array(r["quad"], np.float32)
                gb = (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max())
                pb = fn(img)[:topk]
                best, bi = 0.0, -1
                for j, b in enumerate(pb):
                    v = iou(gb, b)
                    if v > best:
                        best, bi = v, j
                if best >= 0.5:
                    tp += 1
                    ious.append(best)
                    fp += len(pb) - 1
                else:
                    fnm += 1
                    fp += len(pb)
            P = tp / max(1, tp + fp)
            R = tp / max(1, tp + fnm)
            F1 = 2 * P * R / max(1e-6, P + R)
            key = f"{name}(top{topk})"
            res[key] = dict(P=float(P * 100), R=float(R * 100), F1=float(F1 * 100),
                            mIoU=float(np.mean(ious) * 100 if ious else 0.0),
                            sec=float(time.time() - t0), top1=(topk == 1))
            v = res[key]
            print(f"{key:18s} P={v['P']:5.2f} R={v['R']:5.2f} F1={v['F1']:5.2f} mIoU={v['mIoU']:5.2f}")
    (ROOT / "deploy" / "eval_classic_det.json").write_text(
        __import__("json").dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
