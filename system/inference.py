# -*- coding: utf-8 -*-
"""推理封装：检测（yolo12n_mix）→ 颜色 → 抠图 → LPRNet 多视角识别。

复用已有实验代码:
  runs_2026-11-03\\code\\pipeline.py  (detect / classify_color / lpr_views / load_lpr)
  D:\\desktop\\复现代码文件夹\\yolo-ulm   (LPRNet 定义)
"""
import base64
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

ROOT5 = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
ROOT3 = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-03")
ROOT2 = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-02")
for _p in (r"D:\CCPD\corner", r"D:\CCPD\combo", r"D:\desktop\复现代码文件夹\yolo-ulm",
           str(ROOT2 / "code"), str(ROOT3 / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline  # noqa: E402
from pipeline import classify_color, imread, lpr_views, load_lpr  # noqa: E402

DET_WEIGHTS = Path(os.environ.get(
    "GATE_DET", ROOT5 / "models" / "det" / "yolo12n_mix" / "weights" / "best.pt"))
CORNER_W = Path(os.environ.get("GATE_CORNER", r"D:\CCPD\corner\runs\cornernet\best.pt"))
N_VIEWS = int(os.environ.get("GATE_VIEWS", "5"))
# 颜色级联：黑底特殊牌（使/领/港澳）走 73 类模型，其余走原模型
SPEC_W = os.environ.get("GATE_REC_SPEC", str(ROOT5 / "models" / "rec" / "lprnet_special73" / "best.pt"))

_state = {}
_lock = threading.Lock()


def _load_lpr_at(path, verbose=True):
    import torch
    from ulm.lprnet import LPRNet, PlateCharset
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    charset = PlateCharset(ck["charset"])
    model = LPRNet(len(charset), dropout=0.0)
    model.load_state_dict(ck["model"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    if verbose:
        print(f"[infer] spec lprnet: {path} classes={len(charset)+1}")
    return model, charset


def color_route(crop):
    """与评测脚本一致的配电色：黑/白底走特殊牌模型。"""
    if crop.size == 0:
        return "blue"
    b = cv2.resize(crop, (192, 64))
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (s >= 70) & (v >= 40)
    if m.sum() < 50:
        return "black" if float(np.median(v)) < 90 else "white"
    ang = h[m] * 2 * np.pi / 180.0
    mean = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * 90 / np.pi) % 180
    if 96 <= mean <= 145:
        return "blue"
    if 41 <= mean < 96:
        return "green"
    if 10 <= mean < 41:
        return "yellow"
    return "black" if float(np.median(v)) < 90 else "white"


def load(verbose=True):
    """惰性加载模型（进程内单例）。"""
    with _lock:
        if _state:
            return _state
        from ultralytics import YOLO
        import torch
        from corner_lib import CornerNet

        if verbose:
            print(f"[infer] detector: {DET_WEIGHTS}")
        _state["det"] = YOLO(str(DET_WEIGHTS))
        _state["lpr"], _state["charset"] = load_lpr()
        _state["lpr_spec"], _state["charset_spec"] = None, None
        if SPEC_W and Path(SPEC_W).exists():
            try:
                _state["lpr_spec"], _state["charset_spec"] = _load_lpr_at(SPEC_W, verbose)
            except Exception as e:  # noqa: BLE001
                print("[infer] spec model load failed:", e)
        ck = torch.load(str(CORNER_W), map_location="cpu", weights_only=False)
        corner = CornerNet(width=ck.get("width", 32))
        corner.load_state_dict(ck["model"])
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        corner.to(dev).eval()
        _state["corner"] = corner
        _state["dev"] = dev
        if verbose:
            print("[infer] corner + lprnet ready")
        return _state


def _quad(box):
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def _crop_margin(img, box, mx=0.10, my=0.22):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * mx, (y2 - y1) * my
    x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    x2, y2 = min(w, int(x2 + dx)), min(h, int(y2 + dy))
    return img[y1:y2, x1:x2].copy()


def _best_read(model, charset, crop):
    """多视角识别，取平均置信度最高的一次。"""
    if crop.size == 0:
        return "", 0.0
    try:
        reads = lpr_views(model, charset, crop, N_VIEWS)
    except Exception as e:  # noqa: BLE001
        print("[infer] recog failed:", e)
        return "", 0.0
    best = ("", 0.0)
    for txt, confs in reads:
        sc = float(np.mean(confs)) if confs else 0.0
        if sc > best[1]:
            best = (txt, sc)
    return best


def recognize(bgr, conf=0.20, max_plates=2):
    """返回 {plates:[{plate,color,conf,box,crop}], annotated(bgr), ms}"""
    import time
    t0 = time.time()
    st = load(verbose=False)
    det = st["det"]
    img = bgr
    h, w = img.shape[:2]
    res = det.predict(img, imgsz=640, conf=conf, verbose=False)[0]
    boxes = []
    for b in res.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        ar = (x2 - x1) / max(1e-6, (y2 - y1))
        if ar < 1.4 or ar > 9.0:
            continue
        boxes.append((float(b.conf), [x1, y1, x2, y2]))
    boxes.sort(key=lambda t: -t[0])
    boxes = boxes[:max_plates]

    plates = []
    if not boxes:
        q = pipeline.colour_quad(img)
        if q is not None:
            x1, y1 = q.min(0)
            x2, y2 = q.max(0)
            boxes = [(0.30, [float(x1), float(y1), float(x2), float(y2)])]

    for conf_, box in boxes:
        x1, y1, x2, y2 = box
        crop, quad, how = None, None, "box"
        try:
            import torch
            from corner_lib import order_quad, predict_quad, warp_plate
            with torch.no_grad():
                q, _s = predict_quad(st["corner"], img, [x1, y1, x2, y2], st["dev"])
            if q is not None:
                q = order_quad(np.asarray(q, np.float32))
                wc = warp_plate(img, q, 192, 64)
                if wc is not None and wc.size:
                    crop, quad, how = wc, q, "corner"
        except Exception as e:  # noqa: BLE001
            print("[infer] corner failed:", e)
        if quad is None:
            quad = _quad(box)
            crop = _crop_margin(img, box)
        color = classify_color(img, quad) if crop.size else "blue"
        txt, rconf = _best_read(st["lpr"], st["charset"], crop)
        plates.append({"plate": txt, "color": color, "conf": round(max(conf_, rconf), 4),
                       "det_conf": round(conf_, 4), "box": [round(v, 1) for v in box],
                       "rect": how, "crop": crop})

    ann = draw(img.copy(), plates)
    return {"plates": plates, "annotated": ann, "ms": int((time.time() - t0) * 1000)}


def draw(img, plates):
    for p in plates:
        x1, y1, x2, y2 = [int(round(v)) for v in p["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
        label = f"{p['plate']} {p['conf']:.2f}"
        cv2.putText(img, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    return img


def jpg_b64(img, quality=85):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def save_jpg(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", img)[1].tofile(str(path))
    return str(path)


if __name__ == "__main__":
    import sys as _s
    paths = _s.argv[1:] or [str(ROOT5 / "data" / "clip_scene" / "index.json")]
    import json
    if paths[0].endswith(".json"):
        idx = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
        paths = [it["path"] for it in idx if it.get("type") in ("closeup", "scene")][:6]
    for p in paths[:6]:
        im = imread(p)
        if im is None:
            print("skip", p)
            continue
        r = recognize(im)
        print(Path(p).name, r["ms"], "ms",
              [(x["plate"], x["color"], x["conf"]) for x in r["plates"]])
