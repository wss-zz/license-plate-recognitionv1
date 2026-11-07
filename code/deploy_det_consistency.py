# -*- coding: utf-8 -*-
"""检测器 torch 与 ONNX(fp32) 的一致性：
   - 在 289 张人工四点图上分别跑，GT = 四点外接框
   - 输出各自 P/R/F1@IoU0.5、每图平均框数、双方框匹配率
"""
import json, sys, time
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
ONNX = ROOT / "deploy/yolo12n_mix_fp32.onnx"
DET = ROOT / "models/det/yolo12n_mix/weights/best.pt"
QUADS = ROOT / "data/quads_manual.json"


def read_img(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(cand, thr=0.7, max_det=300):
    cand = sorted(cand, key=lambda t: -t[0])
    keep = []
    for sc, b in cand:
        if all(iou(b, k[1]) < thr for k in keep):
            keep.append((sc, b))
            if len(keep) >= max_det:
                break
    return keep


def run_onnx(sess, img, conf=0.25, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh))
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(x)})[0][0]
    cand = []
    for i in np.where(out[4] >= conf)[0]:
        cx, cy, bw, bh = out[0, i], out[1, i], out[2, i], out[3, i]
        b = ((cx - bw / 2 - x0) / r, (cy - bh / 2 - y0) / r,
             (cx + bw / 2 - x0) / r, (cy + bh / 2 - y0) / r)
        cand.append((float(out[4, i]), b))
    return nms(cand)


def prf(pairs):
    tp, fp, fn = pairs
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p * 100, r * 100, (2 * p * r / max(p + r, 1e-9)) * 100


def main():
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])
    yolo = YOLO(str(DET))
    items = json.load(open(QUADS, encoding="utf-8"))
    stat = {"torch": [0, 0, 0], "onnx": [0, 0, 0]}
    nb_t = nb_o = same_cnt = matched = total_boxes = 0
    t_on = t_to = 0.0
    for it in items:
        img = read_img(it["file"])
        q = np.array(it["quad"], np.float32)
        gt = (float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max()))

        t = time.perf_counter()
        r = yolo.predict(img, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
        t_to += time.perf_counter() - t
        tb = [(float(c), tuple(float(v) for v in b)) for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                                                                   r.boxes.conf.cpu().numpy())]
        t = time.perf_counter()
        ob = run_onnx(sess, img)
        t_on += time.perf_counter() - t

        nb_t += len(tb)
        nb_o += len(ob)
        same_cnt += int(len(tb) == len(ob))
        for name, boxes in (("torch", tb), ("onnx", ob)):
            hit = any(iou(b, gt) >= 0.5 for _, b in boxes)
            stat[name][0] += int(hit)
            stat[name][2] += int(not hit)
            stat[name][1] += len(boxes) - int(hit)
        # 双方框互相匹配（IoU>=0.9 视为同一框）
        for _, b in tb:
            if any(iou(b, c) >= 0.9 for _, c in ob):
                matched += 1
        total_boxes += len(tb)

    n = len(items)
    summary = {"n": n, "torch": {}, "onnx": {}, "box_agreement": round(matched / max(total_boxes, 1) * 100, 2),
               "same_count_rate": round(same_cnt / n * 100, 2),
               "boxes_per_img": {"torch": round(nb_t / n, 2), "onnx": round(nb_o / n, 2)},
               "latency_ms": {"torch": round(t_to / n * 1000, 1), "onnx": round(t_on / n * 1000, 1)}}
    for name in ("torch", "onnx"):
        p, rc, f1 = prf(tuple(stat[name]))
        summary[name] = {"precision": round(p, 2), "recall": round(rc, 2), "f1": round(f1, 2), "counts": stat[name]}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    (ROOT / "deploy/det_consistency.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
