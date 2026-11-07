# -*- coding: utf-8 -*-
"""特殊车牌端到端：检测 → 裁剪 → 词表闸门识别，与 GT 文本比对。"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, torch

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, r"D:\desktop\车辆闸机项目\runs_2026-11-02\code")
sys.path.insert(0, r"D:\desktop\复现代码文件夹\yolo-ulm")
from eval_cascade import decode2, rec_all, SPEC_CH  # noqa
import importlib.util
spec = importlib.util.spec_from_file_location("eval_rec", r"D:\desktop\车辆闸机项目\runs_2026-11-02\code\eval_rec.py")
er = importlib.util.module_from_spec(spec); spec.loader.exec_module(er)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1); inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    from ultralytics import YOLO
    det = YOLO(a.weights)
    m70, cs70 = er.load_model(Path(r"D:\desktop\车辆闸机项目\runs_2026-11-02\models\lprnet_final8\best.pt"), DEV)
    m73, cs73 = er.load_model(ROOT / "models" / "rec" / "lprnet_special73" / "best.pt", DEV)

    gt = json.loads((ROOT / "data" / "quads_special.json").read_text(encoding="utf-8"))
    by_file = {}
    for r in gt:
        by_file.setdefault(r["file"], []).append(r)

    n_gt = det_ok = rec_ok = 0
    for f, rows in by_file.items():
        img = cv2.imdecode(np.fromfile(f, np.uint8), 1)
        if img is None:
            continue
        boxes = det.predict(img, conf=0.25, iou=0.7, imgsz=640, verbose=False)[0].boxes.xyxy.cpu().numpy()
        for r in rows:
            q = np.array(json.loads(r["quad"]) if isinstance(r["quad"], str) else r["quad"], np.float32)
            gb = np.array([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()], np.float32)
            best, bb = 0.0, None
            for b in boxes:
                v = iou(gb, b)
                if v > best:
                    best, bb = v, b
            n_gt += 1
            if bb is None or best < 0.5:
                continue
            det_ok += 1
            x1, y1, x2, y2 = [int(round(float(v))) for v in bb]
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (192, 64))
            item = [{"path": "", "text": r["text"], "crop": crop, "source": "special"}]
            p70 = rec_all(m70, cs70, [{"path": "", "text": r["text"], "crop": crop}])[0]
            p73 = rec_all(m73, cs73, [{"path": "", "text": r["text"], "crop": crop}])[0]
            text = p73["pred"] if (any(c in p73["pred"] for c in SPEC_CH) and p73["conf"] >= 0.3) else p70["pred"]
            if text == r["text"]:
                rec_ok += 1
    print(f"{a.tag}: GT={n_gt} 检出={det_ok} ({det_ok/n_gt*100:.1f}%) 端到端整牌={rec_ok} ({rec_ok/n_gt*100:.1f}%)")


if __name__ == "__main__":
    main()
