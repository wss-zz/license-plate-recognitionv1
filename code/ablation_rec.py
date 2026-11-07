# -*- coding: utf-8 -*-
"""识别器消融：同一批人工四点矫正图上比较整牌/字符准确率、参数量、体积、单图延迟。

协议与报告 §4.3 的端到端口径一致：用人工四点 warp 到 192×64，排除检测误差，
只比较识别器本身。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, r"D:\desktop\复现代码文件夹\yolo-ulm")
from ulm.lprnet import LPRNet, PlateCharset, ctc_greedy_decode  # noqa: E402

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
QUADS = json.loads((ROOT / "data" / "quads_manual.json").read_text(encoding="utf-8"))

MODELS = {
    "LPRNet-70(ours)":  Path(r"D:\desktop\车辆闸机项目\runs_2026-11-02\models\lprnet_final8\best.pt"),
    "LPRNet-73":        ROOT / "models/lprnet_special73/best.pt",
}
VIEWS = 23


def read_img(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def warp(img, quad, w=192, h=64):
    q = np.array(quad, dtype=np.float32)
    s = q.sum(1)
    d = np.diff(q, axis=1).ravel()
    rect = np.zeros((4, 2), np.float32)
    rect[0] = q[np.argmin(s)]; rect[2] = q[np.argmax(s)]
    rect[1] = q[np.argmin(d)]; rect[3] = q[np.argmax(d)]
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (w, h))


def resize_views(crop, n=VIEWS, w=94, h=24):
    out = [cv2.resize(crop, (w, h))]
    rng = np.random.RandomState(0)
    for _ in range(n - 1):
        dx = rng.randint(-1, 2); dy = rng.randint(-1, 2)
        sc = 1.0 + rng.uniform(-0.05, 0.05)
        M = np.array([[sc, 0, dx], [0, sc, dy]], np.float32)
        out.append(cv2.resize(cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                                             borderMode=cv2.BORDER_REPLICATE), (w, h)))
    return np.stack(out)


def tta_decode(model, charset, views, dev):
    x = views.astype(np.float32) / 127.5 - 1.0
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        logp = model(x)
    texts = ctc_greedy_decode(logp.float().cpu(), charset)
    return texts


def lev(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs, gts = [], []
    for q in QUADS:
        p = Path(q.get("path") or q.get("file"))
        if not p.exists():
            p = Path(r"D:\车牌图片") / p.name
        im = read_img(str(p))
        if im is None:
            continue
        imgs.append(warp(im, q["quad"]))
        gts.append(q["text"].replace("·", ""))
    print("样本", len(imgs))

    rows = []
    for name, w in MODELS.items():
        if not w.exists():
            print("[skip]", name); continue
        ck = torch.load(str(w), map_location="cpu")
        chars = ck.get("charset") or ck.get("chars") or ""
        if isinstance(chars, list):
            chars = "".join(chars)
        charset = PlateCharset(chars)
        ncls = ck["model"]["classifier.weight"].shape[0]
        model = LPRNet(ncls)
        sd = ck.get("model") or ck.get("state_dict") or ck
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
        model.to(dev).eval()
        nparam = sum(p.numel() for p in model.parameters()) / 1e6
        size = w.stat().st_size / 2 ** 20

        ok = 0; dist = 0; tot = 0
        t0 = time.time()
        for crop, gt in zip(imgs, gts):
            views = resize_views(crop)
            texts = tta_decode(model, charset, views, dev)
            # 简单多数投票
            from collections import Counter
            pred = Counter(texts).most_common(1)[0][0]
            ok += int(pred == gt)
            dist += lev(pred, gt); tot += max(len(gt), 1)
        infer_s = time.time() - t0

        # 延迟（单视图）
        v1 = resize_views(imgs[0], n=1)
        for _ in range(5):
            tta_decode(model, charset, v1, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time(); N = 30
        for _ in range(N):
            tta_decode(model, charset, v1, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        lat = (time.time() - t0) / N * 1000

        plate = ok / len(imgs) * 100
        char_acc = (1 - dist / tot) * 100
        rows.append(dict(name=name, plate=plate, char=char_acc, mb=size, mparam=nparam,
                         ms=lat, n=len(imgs), tta_s=infer_s))
        print(f"{name:16s} 整牌 {plate:5.2f}%  字符 {char_acc:5.2f}%  {size:5.2f}MB "
              f"{nparam:5.2f}M  {lat:6.2f}ms/图(单视图)")

    out = ROOT / "deploy" / "ablation_rec.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
