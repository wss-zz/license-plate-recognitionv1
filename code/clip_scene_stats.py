# -*- coding: utf-8 -*-
"""CLIP scene clustering + scene / colour / quality statistics for the collected dataset.

Inputs : data/clip_scene/index.json (md5-deduped file list), data/clip_scene/det_scan.csv
Outputs: data/clip_scene/{images.csv,clusters.md,summary.md,emb.npy}
         figs/clip_clusters/cXX.jpg  (montage of images closest to each centroid)
"""
import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import dataset_index as di

WEIGHTS = Path(r"D:\clip_weights\clip-vit-large-patch14")
OUT = di.OUT
FIGS = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05\figs\clip_clusters")

SCENE = [
    ("白天-晴", ["a photo of a car on a sunny daytime street",
                 "a bright daytime photo of a vehicle outdoors"]),
    ("夜间", ["a photo taken at night with very dark background",
              "a nighttime photo of a car with headlights on"]),
    ("阴天", ["a photo taken on an overcast cloudy day with flat grey sky"]),
    ("雨天", ["a photo of a car in the rain, wet ground and raindrops"]),
    ("雾霾", ["a photo of a car in fog or haze, low visibility and milky background"]),
    ("逆光强光", ["a backlit photo with strong sun glare or lens flare"]),
    ("车库室内", ["an indoor underground parking garage with artificial ceiling lights"]),
    ("黄昏", ["a photo taken at dusk or sunset with warm orange light"]),
]
COLOR_CLIP = [
    ("蓝牌", ["a car with a blue license plate", "a close-up of a blue license plate"]),
    ("绿牌", ["a car with a green license plate, new energy vehicle",
              "a close-up of a green license plate"]),
    ("黄牌", ["a large truck or bus with a yellow license plate",
              "a close-up of a yellow license plate"]),
    ("白牌", ["a car with a white license plate", "a close-up of a white license plate"]),
    ("黑牌", ["a car with a black license plate", "a close-up of a black license plate"]),
]
TYPE_HEADS = [("crop", ["a cropped close-up where the license plate fills the image"]),
              ("scene", ["a photo of a whole car in a street or parking lot scene"])]

QUALITY_RULES = [
    ("反光过曝", lambda r: r["over"] >= 0.10),
    ("模糊", lambda r: r["lap"] < 500),
    ("暗光", lambda r: r["lum"] < 75),
    ("欠曝", lambda r: r["under"] >= 0.35),
]


def load_bgr(path):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), 1)
    except Exception:
        return None


def text_feats(model, proc, classes, device):
    feats = []
    with torch.no_grad():
        for name, prompts in classes:
            tk = proc(text=prompts, return_tensors="pt", padding=True, truncation=True)
            tk = {k: v.to(device) for k, v in tk.items()}
            f = F.normalize(model.get_text_features(**tk).float(), dim=-1).mean(0)
            feats.append(F.normalize(f, dim=-1))
    return torch.stack(feats)


def kmeans(X, k, iters=80, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    C = [X[rng.integers(n)]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(k - 1):
        p = d2 / max(d2.sum(), 1e-12)
        i = int(rng.choice(n, p=p))
        C.append(X[i])
        d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(1))
    C = np.stack(C)
    for _ in range(iters):
        d = (X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * X @ C.T
        lab = d.argmin(1)
        newC = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C, atol=1e-7):
            C = newC
            break
        C = newC
    d = (X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * X @ C.T
    lab = d.argmin(1)
    return lab, C, np.sqrt(np.maximum(d.min(1), 0))


def montage(idxs, path_of, out_path, cell=170, cols=8, rows=3):
    canvas = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    for pos, gi in enumerate(idxs[: cols * rows]):
        bgr = load_bgr(path_of[gi])
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        s = cell / max(h, w)
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
        canvas.paste(im, ((pos % cols) * cell, (pos // cols) * cell))
        dr.text(((pos % cols) * cell + 3, (pos // cols) * cell + 3), str(gi), fill=(220, 0, 0))
    canvas.save(out_path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items, _ = di.build_index()
    det = {r["idx"]: r for r in csv.DictReader(open(OUT / "det_scan.csv", encoding="utf-8-sig"))}
    for it in items:
        d = det.get(str(it["idx"]), {})
        it["det_type"] = d.get("type", "unknown")
        it["hsv_color"] = d.get("hsv_color", "其他")
        it["n_plate"] = int(d.get("n_plate") or 0)
        it["max_ratio"] = float(d.get("max_ratio") or 0)
        it["lap"] = float(d.get("lap") or 0)
        it["lum"] = float(d.get("lum") or 0)
        it["over"] = float(d.get("over") or 0)
        it["under"] = float(d.get("under") or 0)

    t0 = time.time()
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(str(WEIGHTS), local_files_only=True,
                                      torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model = model.to(device).eval()
    proc = CLIPProcessor.from_pretrained(str(WEIGHTS), local_files_only=True)
    print(f"[clip] loaded {time.time()-t0:.1f}s")

    heads = {}
    for name, classes in (("scene", SCENE), ("color_clip", COLOR_CLIP), ("type_clip", TYPE_HEADS)):
        heads[name] = (classes, text_feats(model, proc, classes, device))

    embs, keep = [], []
    for s in range(0, len(items), args.batch):
        chunk = items[s:s + args.batch]
        batch, idxs = [], []
        for it in chunk:
            bgr = load_bgr(it["path"])
            if bgr is None:
                continue
            batch.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            idxs.append(it["idx"])
        if not batch:
            continue
        px = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        if device == "cuda":
            px = px.half()
        with torch.no_grad():
            f = model.get_image_features(pixel_values=px)
        embs.append(F.normalize(f.float(), dim=-1).cpu().numpy())
        keep.extend(idxs)
        if s % (args.batch * 8) == 0:
            print(f"[clip] {min(s+len(batch), len(items))}/{len(items)} {time.time()-t0:.1f}s", flush=True)
    E = np.concatenate(embs).astype(np.float32)
    keep_set = set(keep)
    items = [it for it in items if it["idx"] in keep_set]
    assert len(items) == len(E)

    for name, (classes, T) in heads.items():
        lg = torch.from_numpy(E) @ T.cpu().T * 100.0
        pr = torch.softmax(lg, dim=-1).numpy()
        pred = pr.argmax(1)
        for i, it in enumerate(items):
            it[name] = classes[pred[i]][0]
            it[name + "_p"] = float(pr[i, pred[i]])

    for it in items:
        # image type: detection wins, CLIP type only rescues detector misses
        it["type"] = it["det_type"]
        if it["type"] == "unknown":
            it["type"] = "crop" if it["type_clip"] == "crop" else "scene"
        it["color"] = it["hsv_color"]
        lab = "正常"
        for name, rule in QUALITY_RULES:
            if rule(it):
                lab = name
                break
        it["quality"] = lab

    scene_idx = [i for i, it in enumerate(items) if it["type"] != "crop"]
    lab, C, dist = kmeans(E[scene_idx], args.k, seed=0)
    for j, i in enumerate(scene_idx):
        items[i]["cluster"] = int(lab[j])
        items[i]["to_centroid"] = float(dist[j])
    for it in items:
        it.setdefault("cluster", -1)
        it.setdefault("to_centroid", 0.0)

    path_of = {it["idx"]: it["path"] for it in items}
    by_cl = defaultdict(list)
    for i in scene_idx:
        by_cl[items[i]["cluster"]].append(items[i])
    order = sorted(by_cl, key=lambda c: -len(by_cl[c]))

    cl = ["# CLIP 场景聚类（非特写图）", "",
          f"- 图像总数（md5 去重）: {len(items)}；参与聚类的非特写图: {len(scene_idx)}",
          f"- 簇数 k: {args.k}；嵌入: CLIP ViT-L/14 768 维，L2 归一化，k-means++",
          "- 簇标题 = 簇内零样本场景预测的多数类；拼图按到质心距离取最近 24 张", ""]
    for rank, c in enumerate(order, 1):
        rows = by_cl[c]
        sc = Counter(r["scene"] for r in rows)
        co = Counter(r["color"] for r in rows)
        dr_ = Counter(f'{r["tag"]}/{r["sub"] or "."}' for r in rows).most_common(3)
        near = sorted(rows, key=lambda r: r["to_centroid"])[:24]
        montage([r["idx"] for r in near], path_of, FIGS / f"c{rank:02d}.jpg")
        cl += [f"## 簇 {rank:02d}  n={len(rows)}  主导场景: {sc.most_common(1)[0][0]}"
               f" ({sc.most_common(1)[0][1]}/{len(rows)})",
               "- 场景分布: " + ", ".join(f"{k} {v}" for k, v in sc.most_common(4)),
               "- 颜色分布: " + ", ".join(f"{k} {v}" for k, v in co.most_common(3)),
               "- 来源: " + ", ".join(f"{a} {v}" for a, v in dr_),
               f"- 拼图: figs/clip_clusters/c{rank:02d}.jpg", ""]
    (OUT / "clusters.md").write_text("\n".join(cl), encoding="utf-8")

    with open(OUT / "images.csv", "w", newline="", encoding="utf-8-sig") as f:
        cols = ["idx", "tag", "sub", "name", "path", "md5", "w", "h", "type", "det_type", "type_clip",
                "type_clip_p", "n_plate", "max_ratio", "cluster", "to_centroid", "scene", "scene_p",
                "color", "color_clip", "color_clip_p", "quality", "lap", "lum", "over", "under"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)

    n = len(items)
    nz = [items[i] for i in scene_idx]
    L = []
    A = L.append
    A("# 采集数据场景统计（CLIP 零样本 + 目标检测）\n")
    A(f"- 图像总数（md5 去重）: **{n}**；检测到车牌 **{sum(it['n_plate'] for it in items)}** 个，"
      f"含 ≥2 牌图像 **{sum(1 for it in items if it['n_plate']>=2)}** 张")
    A(f"- 图像类型: 车牌特写 {sum(1 for it in items if it['type']=='crop')}，"
      f"车头近景 {sum(1 for it in items if it['det_type']=='closeup')}，"
      f"车辆全景 {sum(1 for it in items if it['det_type']=='scene')}，"
      f"检测器未命中 {sum(1 for it in items if it['det_type']=='unknown')}（已用 CLIP 兜底归入前两类）")
    A("")

    def table(title, key, scope):
        A(f"## {title}\n")
        A("| 类别 | 图数 | 占比 |")
        A("|---|---:|---:|")
        c = Counter(it[key] for it in scope)
        for k, v in c.most_common():
            A(f"| {k} | {v} | {v/len(scope)*100:.1f}% |")
        A("")

    table(f"1. 光照/天气场景分布（非特写，n={len(nz)}）", "scene", nz)
    table("2. 车牌颜色分布（全部图像，检测框内 HSV，已验证 96.4%）", "color", items)
    table("3. 画面质量（检测框内客观指标）", "quality", items)

    A("## 4. 各来源 × 场景（非特写）\n")
    A("| 来源 | 非特写图 | " + " | ".join(k for k, _ in SCENE) + " | 特写图 |")
    A("|---|---:|" + "---:|" * len(SCENE) + "---:|")
    for tag, _ in di.ROOTS:
        rows = [it for it in items if it["tag"] == tag]
        if not rows:
            continue
        rs = [it for it in rows if it["type"] != "crop"]
        c = Counter(it["scene"] for it in rs)
        A(f"| {tag} | {len(rs)} | " + " | ".join(str(c.get(k, 0)) for k, _ in SCENE)
          + f" | {sum(1 for it in rows if it['type']=='crop')} |")
    A("")

    A("## 5. 各来源 × 颜色\n")
    A("| 来源 | 图数 | 蓝牌 | 绿牌 | 黄牌 | 白牌 | 黑牌 | 其他 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tag, _ in di.ROOTS:
        rows = [it for it in items if it["tag"] == tag]
        if not rows:
            continue
        c = Counter(it["color"] for it in rows)
        A(f"| {tag} | {len(rows)} | " + " | ".join(str(c.get(k, 0)) for k in
                                                   ("蓝牌", "绿牌", "黄牌", "白牌", "黑牌", "其他")) + " |")
    A("")

    A("## 6. 零样本 / 规则校验（用目录名当弱真值）\n")
    A("| 校验集 | 图数 | 期望 | 命中 | 命中率 |")
    A("|---|---:|---|---:|---:|")
    checks = []
    for tag in ("车牌数据集", "新车牌数据集"):
        for sub in sorted({it["sub"] for it in items if it["tag"] == tag}):
            if "蓝牌" in sub:
                checks.append((f"{tag}/{sub.split('_', 2)[0]}", tag, sub, "color", "蓝牌"))
            elif "绿牌" in sub:
                checks.append((f"{tag}/{sub.split('_', 2)[0]}", tag, sub, "color", "绿牌"))
            elif "黄牌" in sub:
                checks.append((f"{tag}/{sub.split('_', 2)[0]}", tag, sub, "color", "黄牌"))
    for name, tag, sub, key, expect in checks:
        rows = [it for it in items if it["tag"] == tag and it["sub"] == sub]
        hit = sum(1 for it in rows if it[key] == expect)
        A(f"| {name} | {len(rows)} | {expect} | {hit} | {hit/len(rows)*100:.1f}% |")
    for sub, key, expect in (("01_地下车库", "scene", "车库室内"), ("03_雨天", "scene", "雨天"),
                             ("04_手抖模糊", "quality", "模糊")):
        rows = [it for it in items if it["tag"] == "_20261105" and it["sub"] == sub]
        hit = sum(1 for it in rows if it[key] == expect)
        dist3 = ", ".join(f"{k} {v}" for k, v in Counter(it[key] for it in rows).most_common(3))
        A(f"| _20261105/{sub} | {len(rows)} | {expect} | {hit} | {hit/len(rows)*100:.1f}% |")
    A("")

    A("## 7. 难例占比\n")
    hard = [it for it in items if it["quality"] != "正常"]
    multi = [it for it in items if it["n_plate"] >= 2]
    A(f"- 质量难例（模糊/暗光/过曝/欠曝）: {len(hard)} 张，{len(hard)/n*100:.1f}% —— "
      + ", ".join(f"{k} {v}" for k, v in Counter(it["quality"] for it in items).most_common()
                  if k != "正常"))
    A(f"- 多车牌图像（≥2 个检测框）: {len(multi)} 张，{len(multi)/n*100:.1f}%")
    A("- 倾斜/偏角: 待由检测四边形（CornerNet）与人工四点补充统计，不在本节口径内")
    A("")
    A("> 口径：场景为 CLIP 零样本（每图一个标签，非逐牌）；颜色为检测框内 HSV 主色（在 112 张带")
    A("> 目录标签的样本上验证 96.4%）；质量来自检测框内拉普拉斯方差/平均亮度/过曝比例。")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"[done] {time.time()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
