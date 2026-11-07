# -*- coding: utf-8 -*-
"""部署优化实验 2/3：INT8 静态量化（QDQ）+ 体积/延迟/QPS/内存 基准。

用法：python deploy_quant_bench.py [--n-det 120] [--threads 4]
产出：runs_2026-11-05\deploy\{*_int8.onnx, bench.json, bench.md}
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\desktop\车辆闸机项目")
R = ROOT / "runs_2026-11-05"
DEP = R / "deploy"
CROPS = Path(r"D:\车牌图片\plate_dataset")
LABELS = CROPS / "rec_labels.txt"


def read_img(p):
    import cv2
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def load_labels():
    items = []
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        rel, txt = line.split("\t")
        p = CROPS / rel
        if p.exists():
            items.append((p, txt.strip()))
    return items


# ---------------- 量化 ----------------
def quant_lpr(src, dst, items):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static
    import cv2

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(items[:200])

        def get_next(self):
            for p, _ in self.it:
                im = read_img(p)
                if im is None:
                    continue
                v = cv2.resize(im, (94, 24))
                x = (v.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
                return {"input": x}
            return None

    quantize_static(str(src), str(dst), Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, reduce_range=False,
                    activation_type=__import__("onnxruntime.quantization.quant_utils",
                                               fromlist=["QuantType"]).QuantType.QUInt8,
                    weight_type=__import__("onnxruntime.quantization.quant_utils",
                                           fromlist=["QuantType"]).QuantType.QInt8)


def quant_det(src, dst, paths):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static
    from onnxruntime.quantization.quant_utils import QuantType
    import cv2

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(paths)

        def get_next(self):
            for p in self.it:
                im = read_img(p)
                if im is None:
                    continue
                h, w = im.shape[:2]
                s = 640 / max(h, w)
                im = cv2.resize(im, (int(round(w * s)), int(round(h * s))))
                c = np.full((640, 640, 3), 114, np.uint8)
                y, x = (640 - im.shape[0]) // 2, (640 - im.shape[1]) // 2
                c[y:y + im.shape[0], x:x + im.shape[1]] = im
                x_ = c[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
                return {"images": x_}
            return None

    quantize_static(str(src), str(dst), Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, reduce_range=False,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)


# ---------------- 基准 ----------------
def bench_session(path, feed_fn, n, warm=5):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for i in range(warm):
        sess.run(None, {name: feed_fn(i)})
    ts = []
    for i in range(n):
        x = feed_fn(i)
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"p50": round(statistics.median(ts), 2), "p90": round(ts[int(len(ts) * 0.9)], 2),
            "mean": round(sum(ts) / len(ts), 2), "qps": round(1000 / (sum(ts) / len(ts)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-det", type=int, default=120)
    args = ap.parse_args()

    from PIL import Image  # noqa: F401  只是确认依赖在
    import onnx
    from onnxruntime.quantization import quantize_static  # noqa: F401

    items = load_labels()
    print(f"识别标注 {len(items)} 张", flush=True)

    # --- LPRNet 量化 ---
    print("[1] 量化 LPRNet ...", flush=True)
    quant_lpr(DEP / "lprnet_fp32.onnx", DEP / "lprnet_int8.onnx", items)

    # --- YOLO 量化 ---
    import csv
    rows = list(csv.DictReader(open(R / "data" / "clip_scene" / "images.csv", encoding="utf-8-sig")))
    import random
    random.seed(7)
    cal = [r["path"] for r in random.sample(rows, min(200, len(rows)))]
    print("[2] 量化 YOLO ...", flush=True)
    quant_det(DEP / "yolo12n_mix_fp32.onnx", DEP / "yolo12n_mix_int8.onnx", cal[:120])

    # --- 体积 ---
    size = {p.name: round(p.stat().st_size / 1e6, 2) for p in sorted(DEP.glob("*.onnx"))}
    print("体积", size, flush=True)

    # --- 延迟 ---
    import cv2

    def lpr_feed(i):
        p, _ = items[i % len(items)]
        im = cv2.resize(read_img(p), (94, 24))
        return (im.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]

    def det_feed(i):
        im = read_img(cal[i % len(cal)])
        h, w = im.shape[:2]
        s = 640 / max(h, w)
        im = cv2.resize(im, (int(round(w * s)), int(round(h * s))))
        c = np.full((640, 640, 3), 114, np.uint8)
        y, x = (640 - im.shape[0]) // 2, (640 - im.shape[1]) // 2
        c[y:y + im.shape[0], x:x + im.shape[1]] = im
        return c[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    res = {}
    res["lprnet_fp32"] = bench_session(DEP / "lprnet_fp32.onnx", lpr_feed, 300)
    res["lprnet_int8"] = bench_session(DEP / "lprnet_int8.onnx", lpr_feed, 300)
    nd = min(args.n_det, len(cal))
    res["det_fp32"] = bench_session(DEP / "yolo12n_mix_fp32.onnx", det_feed, nd, warm=2)
    res["det_int8"] = bench_session(DEP / "yolo12n_mix_int8.onnx", det_feed, nd, warm=2)
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)

    out = {"size_mb": size, "bench": res, "threads": 4}
    (DEP / "bench.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", DEP / "bench.json")


if __name__ == "__main__":
    main()
