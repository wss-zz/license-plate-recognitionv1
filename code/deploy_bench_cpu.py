# -*- coding: utf-8 -*-
"""CPU（同一台机器，intra_op=4）上 torch / ONNX fp32 / ONNX int8 的逐阶段延迟与体积对比。
   输入: 人工四点图 20 张做 warm-up 与计时。
"""
import json, time
from pathlib import Path
import numpy as np
import cv2
import torch
import onnxruntime as ort
from ultralytics import YOLO
import sys

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
sys.path.insert(0, r"D:\desktop\复现代码文件夹\yolo-ulm")
from ulm.lprnet import LPRNet, PlateCharset  # noqa: E402

OUT = ROOT / "deploy/bench_cpu.json"
QUADS = json.load(open(ROOT / "data/quads_manual.json", encoding="utf-8"))


def read_img(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), 1)


def bench(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


def letterbox(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    c = np.full((size, size, 3), 114, np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    c[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh))
    return c, r, x0, y0


def sess_of(p):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def main():
    img = read_img(QUADS[0]["file"])
    q = np.array(QUADS[0]["quad"], np.float32)
    dst = np.array([[0, 0], [191, 0], [191, 63], [0, 63]], np.float32)
    crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (192, 64))
    rec_in = (cv2.resize(crop, (94, 24)).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]

    ck = torch.load(r"D:\desktop\车辆闸机项目\runs_2026-11-02\models\lprnet_final8\best.pt",
                    map_location="cpu", weights_only=False)
    cs = PlateCharset(ck["charset"])
    net = LPRNet(len(cs), dropout=0.0)
    net.load_state_dict(ck["model"])
    net.eval()
    lt = torch.from_numpy(rec_in)

    canvas, *_ = letterbox(img)
    det_in = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255)

    yolo = YOLO(str(ROOT / "models/det/yolo12n_mix/weights/best.pt"))
    s_det32 = sess_of(ROOT / "deploy/yolo12n_mix_fp32.onnx")
    s_det8 = sess_of(ROOT / "deploy/yolo12n_mix_int8.onnx")
    s_rec32 = sess_of(ROOT / "deploy/lprnet_fp32.onnx")
    s_rec8 = sess_of(ROOT / "deploy/lprnet_int8.onnx")

    res = {}
    res["lpr_torch_cpu"] = bench(lambda: net(lt))
    res["lpr_onnx_fp32"] = bench(lambda: s_rec32.run(None, {s_rec32.get_inputs()[0].name: rec_in}))
    res["lpr_onnx_int8"] = bench(lambda: s_rec8.run(None, {s_rec8.get_inputs()[0].name: rec_in}))
    res["det_torch_cpu"] = bench(lambda: yolo.predict(img, imgsz=640, conf=0.25, device="cpu", verbose=False))
    res["det_onnx_fp32"] = bench(lambda: s_det32.run(None, {s_det32.get_inputs()[0].name: det_in}))
    try:
        res["det_onnx_int8"] = bench(lambda: s_det8.run(None, {s_det8.get_inputs()[0].name: det_in}))
    except Exception as e:  # noqa: BLE001
        res["det_onnx_int8"] = f"n/a ({e})"

    size = {p.name: round(p.stat().st_size / 1e6, 2) for p in (ROOT / "deploy").glob("*.onnx")}
    res["sizes_mb"] = size
    res["fps_single_lane_est"] = {k: round(1000 / v, 1) for k, v in res.items() if isinstance(v, float)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
