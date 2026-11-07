# -*- coding: utf-8 -*-
"""在 289 张人工四点标注图上评测：
   1) 检测 P/R/F1（GT = 四点外接框，IoU 0.5）
   2) 识别整牌准确率 / 字符准确率（GT = 图片文件名整理出的车牌号，经人工四点 warp）
   3) ONNX 检测器与 torch 检测器的一致性
输出 runs_2026-11-05/data/eval_manual.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
sys.path.insert(0, r"D:\desktop\复现代码文件夹\yolo-ulm")
sys.path.insert(0, str(ROOT / "code"))
from ulm.lprnet import LPRNet, PlateCharset          # noqa: E402
from ultralytics import YOLO                          # noqa: E402

DET = ROOT / "models/det/yolo12n_mix/weights/best.pt"
LPR = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-02\models\lprnet_final8\best.pt")
QUADS = ROOT / "data/quads_manual.json"
OUT = ROOT / "data/eval_manual.json"


def read_img(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def bbox_of(quad):
    a = np.array(quad, dtype=np.float32)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def order_quad(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def warp(img, quad, w=192, h=64):
    q = order_quad(quad)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (w, h))


def to_tensor(im):
    v = cv2.resize(im, (94, 24)).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(v.transpose(2, 0, 1))[None]


def decode(lg, cs):
    idx = lg.detach().cpu().argmax(-1).numpy().ravel()
    out, prev = [], -1
    for i in idx:
        if i != 0 and i != prev:
            out.append(int(i))
        prev = i
    return cs.decode(out)


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def main():
    torch.cuda.is_available() and torch.set_grad_enabled(False)
    ck = torch.load(str(LPR), map_location="cpu", weights_only=False)
    cs = PlateCharset(ck["charset"])
    net = LPRNet(len(cs), dropout=0.0)
    net.load_state_dict(ck["model"])
    net.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(dev)
    yolo = YOLO(str(DET))

    items = json.load(open(QUADS, encoding="utf-8"))
    tp = fp = fn = 0
    hit = 0
    cer_num = cer_den = 0
    rows = []
    t0 = time.time()
    for k, it in enumerate(items):
        img = read_img(it["file"])
        if img is None:
            continue
        gt = bbox_of(it["quad"])
        r = yolo.predict(img, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        order = np.argsort(-confs)
        used = set()
        best_hit = -1
        for i in order:
            b = boxes[i]
            if iou(b, gt) >= 0.5 and best_hit < 0:
                best_hit = i
                tp += 1
                used.add(i)
                break
        if best_hit < 0:
            fn += 1
        fp += len(boxes) - len(used)

        # 识别：用人工四点 warp（排除检测误差对识别的影响）
        crop = warp(img, it["quad"])
        pred = decode(net(to_tensor(crop).to(dev)), cs)
        gtxt = it.get("text") or it.get("name") or ""
        hit += int(str(pred) == str(gtxt))
        cer_num += edit_distance(pred, gtxt)
        cer_den += max(len(gtxt), 1)
        rows.append({"file": it["file"], "gt": gtxt, "pred": pred, "ok": bool(pred == gtxt),
                     "det": bool(best_hit >= 0), "score": float(confs[best_hit]) if best_hit >= 0 else 0.0})
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(items)} … {time.time()-t0:.1f}s", flush=True)

    n = len(rows)
    p = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    f1 = 2 * p * rc / max(p + rc, 1e-9)
    res = {
        "n": n,
        "det": {"tp": tp, "fp": fp, "fn": fn, "precision": round(p * 100, 2),
                "recall": round(rc * 100, 2), "f1": round(f1 * 100, 2)},
        "rec": {"plate_acc": round(hit / n * 100, 2), "char_acc": round((1 - cer_num / cer_den) * 100, 2),
                "hits": hit},
        "secs": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
