# -*- coding: utf-8 -*-
"""部署优化实验 3/3：量化前后精度一致性（同一测试集）。

识别：CCPD 车牌按标注框裁剪（文件名内含 bbox 与车牌号）× 420 张，
     比较 torch fp32 / onnx fp32 / onnx int8 的整牌准确率与字符准确率。
检测：onnx fp32 与 torch fp32 在同一批图上做 IoU>=0.5 一致性核对。
产出 runs_2026-11-05\deploy\accuracy.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\desktop\车辆闸机项目")
R = ROOT / "runs_2026-11-05"
DEP = R / "deploy"
CCPD = Path(r"D:\CCPD\dataset\CCPD2019")
sys.path.insert(0, str(ROOT / "runs_2026-11-02" / "code"))
sys.path.insert(0, r"D:\desktop\复现代码文件夹\yolo-ulm")


def read_img(p):
    import cv2
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


PROVINCES = "皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ"          # 24 个，不含 I/O
ADS = "0123456789" + ALPHA                  # 车牌第 3 位起的字符集


def decode_plate(seg):
    """CCPD 车牌字段是索引编码，如 0_0_22_10_26_29_24 -> 皖A12345。"""
    parts = seg.split("_")
    if len(parts) < 3:
        return None
    try:
        idx = [int(p) for p in parts]
    except ValueError:
        return None
    if not (0 <= idx[0] < len(PROVINCES) and 0 <= idx[1] < len(ALPHA)):
        return None
    if any(not (0 <= i < len(ADS)) for i in idx[2:]):
        return None
    return PROVINCES[idx[0]] + ALPHA[idx[1]] + "".join(ADS[i] for i in idx[2:])


def parse_ccpd(name):
    """CCPD: 025-95_113-154&383_386&473-386&473_..._363&402-0_0_22_27_27_33_16-37-15.jpg"""
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) < 5:
        return None, None
    bbox, verts, plate = parts[2], parts[3], parts[4]
    try:
        x1y1, x2y2 = bbox.split("_")
        x1, y1 = map(int, x1y1.split("&"))
        x2, y2 = map(int, x2y2.split("&"))
        quad = [tuple(map(int, v.split("&"))) for v in verts.split("_")]
    except ValueError:
        return None, None, None
    if len(quad) != 4:
        return None, None, None
    txt = decode_plate(plate)
    if txt is None:
        return None, None, None
    return (x1, y1, x2, y2), quad, txt


def collect(limit=420):
    import random
    random.seed(2026)
    files = []
    for sub in ("ccpd_base", "ccpd_tilt", "ccpd_blur", "ccpd_weather", "ccpd_challenge"):
        d = CCPD / sub
        if not d.exists():
            continue
        fs = [p for p in d.iterdir() if p.suffix.lower() == ".jpg"]
        files.extend(random.sample(fs, min(len(fs), limit // 5)))
    import cv2
    out = []
    for f in files:
        box, quad, plate = parse_ccpd(f.name)
        if box is None:
            continue
        im = read_img(f)
        if im is None:
            continue
        src = np.array(quad, np.float32)
        # 按左上、右上、右下、左下排序后透视变换（与推理层 warp 一致）
        s = src.sum(1)
        d = np.diff(src, axis=1).ravel()
        ordered = np.array([src[np.argmin(s)], src[np.argmin(d)],
                            src[np.argmax(s)], src[np.argmax(d)]], np.float32)
        dst = np.array([[0, 0], [191, 0], [191, 63], [0, 63]], np.float32)
        M = cv2.getPerspectiveTransform(ordered, dst)
        crop = cv2.warpPerspective(im, M, (192, 64))
        if crop.size == 0:
            continue
        out.append((crop, plate))
    return out


def pre(x):
    import cv2
    v = cv2.resize(x, (94, 24))
    return ((v.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1))[None]


def greedy(logits):
    idx = logits.argmax(1)
    out, prev = [], -1
    for i in idx:
        if i != 0 and i != prev:
            out.append(int(i))
        prev = i
    return out


def main():
    from ulm.lprnet import LPRNet, PlateCharset
    import onnxruntime as ort

    ck = torch.load(str(ROOT / "runs_2026-11-02/models/lprnet_final8/last.pt"),
                    map_location="cpu", weights_only=False)
    charset = PlateCharset(ck["charset"])
    model = LPRNet(len(charset), dropout=0.0)
    model.load_state_dict(ck["model"])
    model.eval()

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    s32 = ort.InferenceSession(str(DEP / "lprnet_fp32.onnx"), so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(DEP / "lprnet_int8.onnx"), so, providers=["CPUExecutionProvider"])
    name = s32.get_inputs()[0].name

    items = collect()
    print(f"识别测试集 {len(items)} 张（CCPD，按标注框裁剪）", flush=True)

    hit = {k: 0 for k in ("torch_fp32", "onnx_fp32", "onnx_int8")}
    ch = {k: [0, 0] for k in hit}
    diff = []
    for crop, gt in items:
        x = pre(crop)
        with torch.no_grad():
            t = model(torch.from_numpy(x)).numpy()[0]
        o32 = s32.run(None, {name: x})[0][0]
        o8 = s8.run(None, {name: x})[0][0]
        for k, lg in (("torch_fp32", t), ("onnx_fp32", o32), ("onnx_int8", o8)):
            txt = charset.decode(greedy(lg))
            hit[k] += int(txt == gt)
            L = max(len(txt), len(gt))
            ch[k][0] += sum(1 for i in range(L) if i < len(txt) and i < len(gt) and txt[i] == gt[i])
            ch[k][1] += len(gt)
        a, b = charset.decode(greedy(o32)), charset.decode(greedy(o8))
        if a != b:
            diff.append({"gt": gt, "fp32": a, "int8": b})

    rec = {k: {"plate_acc": round(hit[k] / len(items) * 100, 2),
               "char_acc": round(ch[k][0] / ch[k][1] * 100, 2)} for k in hit}
    print(json.dumps(rec, ensure_ascii=False, indent=2), flush=True)

    # 检测：onnx fp32 vs torch fp32 一致性
    import csv, random
    from ultralytics import YOLO

    rows = list(csv.DictReader(open(R / "data/clip_scene/images.csv", encoding="utf-8-sig")))
    random.seed(11)
    paths = [r["path"] for r in random.sample(rows, 120)]
    ym = YOLO(str(R / "models/det/yolo12n_mix/weights/best.pt"))

    def letterbox(im, s=640):
        import cv2
        h, w = im.shape[:2]
        r = s / max(h, w)
        im2 = cv2.resize(im, (int(round(w * r)), int(round(h * r))))
        c = np.full((s, s, 3), 114, np.uint8)
        y, x = (s - im2.shape[0]) // 2, (s - im2.shape[1]) // 2
        c[y:y + im2.shape[0], x:x + im2.shape[1]] = im2
        return c, r, x, y

    def boxes_onnx(im, conf=0.25):
        c, r, px, py = letterbox(im)
        x = c[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        o = s32det.run(None, {s32det.get_inputs()[0].name: x})[0][0]
        cand = []
        for i in np.where(o[4] >= conf)[0]:
            cx, cy, w, h = o[0, i], o[1, i], o[2, i], o[3, i]
            cand.append((float(o[4, i]), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        # 贪心 NMS（与 Ultralytics 默认 IoU=0.7 对齐）
        cand.sort(key=lambda t: -t[0])
        keep = []
        for sc, b in cand:
            if all(iou(b, k) < 0.7 for k in keep):
                keep.append((b[0] - px, b[1] - py, b[2] - px, b[3] - py))
        return [((b[0]) / r, (b[1]) / r, (b[2]) / r, (b[3]) / r) for b in keep]

    def boxes_torch(im):
        r = ym.predict(im, imgsz=640, conf=0.25, verbose=False)[0]
        return [tuple(map(float, b)) for b in r.boxes.xyxy.cpu().numpy()]

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    s32det = ort.InferenceSession(str(DEP / "yolo12n_mix_fp32.onnx"), so, providers=["CPUExecutionProvider"])
    tot_t = tot_o = tp = fp = fn = same = 0
    for p in paths:
        im = read_img(p)
        if im is None:
            continue
        bt, bo = boxes_torch(im), boxes_onnx(im)
        tot_t += len(bt)
        tot_o += len(bo)
        used, m = set(), 0
        for a in bo:
            best, bi = 0.0, -1
            for j, b in enumerate(bt):
                if j in used:
                    continue
                v = iou(a, b)
                if v > best:
                    best, bi = v, j
            if best >= 0.5:
                used.add(bi)
                m += 1
            else:
                fp += 1
        tp += m
        fn += len(bt) - m
        same += int(len(bt) == len(bo))
    det = {"images": len(paths), "torch_boxes": tot_t, "onnx_boxes": tot_o,
           "recall_iou50": round(tp / max(1, tot_t) * 100, 2),
           "precision_iou50": round(tp / max(1, tp + fp) * 100, 2),
           "same_count_rate": round(same / len(paths) * 100, 2)}
    print(json.dumps(det, ensure_ascii=False), flush=True)

    out = {"recognition": rec, "recognition_n": len(items),
           "recog_diff_examples": diff[:40],
           "detection_onnx_vs_torch": det}
    (DEP / "accuracy.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", DEP / "accuracy.json")


if __name__ == "__main__":
    main()
