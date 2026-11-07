# -*- coding: utf-8 -*-
"""从 CCPD 生成同口径 GT 裁剪图（排除训练集用过的图）+ 检测评测标签。"""
import json, random, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
CCPD = Path(r"D:\CCPD\dataset\CCPD2019")
OUT = ROOT / "data" / "ccpd_eval"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

PROV = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁",
        "豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新"]
ALPHA = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
ADS = ALPHA + list("0123456789")

def decode(num):
    idx = [int(x) for x in num.split("_")]
    return PROV[idx[0]] + ALPHA[idx[1]] + "".join(ADS[i] for i in idx[2:])

def read(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

def warp(img, quad, w=192, h=64):
    q = np.array(quad, np.float32)
    s = q.sum(1); d = np.diff(q, axis=1).ravel()
    r = np.zeros((4, 2), np.float32)
    r[0] = q[np.argmin(s)]; r[2] = q[np.argmax(s)]
    r[1] = q[np.argmin(d)]; r[3] = q[np.argmax(d)]
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(r, dst), (w, h))

def main():
    used = {p.name for p in (ROOT/"det_mix"/"images"/"train").iterdir()}
    used |= {p.name for p in (ROOT/"det_mix"/"images"/"val").iterdir()}
    files = [p for p in sorted((CCPD/"ccpd_base").glob("*.jpg")) if p.name not in used]
    print(f"候选 {len(files)} 张（已排除训练集 {len(used)} 个文件名）")
    random.seed(2026)
    sel = random.sample(files, min(N, len(files)))
    (OUT/"images").mkdir(parents=True, exist_ok=True)
    rec, det = [], []
    for k, f in enumerate(sel):
        parts = f.stem.split("-")
        if len(parts) != 7:
            continue
        bbox = [int(v) for seg in parts[2].split("_") for v in seg.split("&")]
        quad = [[int(v) for v in seg.split("&")] for seg in parts[3].split("_")]
        if len(quad) != 4:
            continue
        text = decode(parts[4])
        img = read(f)
        if img is None:
            continue
        crop = warp(img, quad)
        name = f"{k:05d}.jpg"
        cv2.imencode(".jpg", crop)[1].tofile(str(OUT/"images"/name))
        rec.append({"file": str(OUT/"images"/name), "text": text, "color": "blue"})
        det.append({"file": str(f), "name": f.name, "bbox": bbox, "quad": quad, "text": text,
                    "w": int(img.shape[1]), "h": int(img.shape[0])})
    (OUT/"rec_manifest.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT/"det_manifest.json").write_text(json.dumps(det, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {len(rec)} 张裁剪图 + {len(det)} 条检测标签 -> {OUT}")

if __name__ == "__main__":
    main()
