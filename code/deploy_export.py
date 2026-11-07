# -*- coding: utf-8 -*-
"""部署优化实验 1/3：导出 ONNX 并做 INT8 静态量化。

产出目录 runs_2026-11-05\deploy\
  lprnet_fp32.onnx / lprnet_int8.onnx
  yolo12n_mix_fp32.onnx / yolo12n_mix_int8.onnx
  export_report.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\desktop\车辆闸机项目")
R = ROOT / "runs_2026-11-05"
OUT = R / "deploy"
OUT.mkdir(parents=True, exist_ok=True)

LPR_W = ROOT / "runs_2026-11-02" / "models" / "lprnet_final8" / "last.pt"
DET_W = R / "models" / "det" / "yolo12n_mix" / "weights" / "best.pt"
sys.path.insert(0, str(ROOT / "runs_2026-11-02" / "code"))
sys.path.insert(0, r"D:\desktop\复现代码文件夹\yolo-ulm")


def export_lpr():
    from ulm.lprnet import LPRNet, PlateCharset

    ck = torch.load(str(LPR_W), map_location="cpu", weights_only=False)
    charset = PlateCharset(ck["charset"])
    model = LPRNet(len(charset), dropout=0.0)
    model.load_state_dict(ck["model"])
    model.eval()
    dummy = torch.zeros(1, 3, 24, 94)
    p = OUT / "lprnet_fp32.onnx"
    torch.onnx.export(model, dummy, str(p), input_names=["input"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    return p, len(charset)


def export_yolo():
    from ultralytics import YOLO

    m = YOLO(str(DET_W))
    p = m.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)
    src = Path(p)
    dst = OUT / "yolo12n_mix_fp32.onnx"
    if src != dst:
        dst.write_bytes(src.read_bytes())
    return dst


if __name__ == "__main__":
    print("[1/2] 导出 LPRNet ...", flush=True)
    lpr, ncls = export_lpr()
    print("   ", lpr.name, f"{lpr.stat().st_size/1e6:.2f} MB", "charset", ncls)

    print("[2/2] 导出 YOLO12n ...", flush=True)
    det = export_yolo()
    print("   ", det.name, f"{det.stat().st_size/1e6:.2f} MB")

    rep = {"lprnet": str(lpr), "detector": str(det)}
    (OUT / "export_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print("done ->", OUT)
