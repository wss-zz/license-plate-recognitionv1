# -*- coding: utf-8 -*-
"""Shared image index for the collected plate dataset (md5 de-duplicated)."""
import hashlib
import json
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ROOTS = [
    ("车牌图片", r"D:\车牌图片"),
    ("特殊车牌", r"D:\特殊车牌数据集"),
    ("车牌数据集", r"D:\车牌数据集"),
    ("新车牌数据集", r"D:\新车牌数据集"),
    ("yungui", r"D:\desktop\车辆闸机项目\plate_dataset_yungui"),
    ("_20261105", r"D:\车牌数据集_20261105"),
    ("补充黄牌", r"D:\车牌数据集_补充"),
]
OUT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05\data\clip_scene")


def md5_file(p: Path, buf_size=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_index(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "index.json"
    if cache.exists() and not force:
        items = json.loads(cache.read_text(encoding="utf-8"))
        return items, len(items)
    items, seen, dup = [], set(), 0
    for tag, root in ROOTS:
        root = Path(root)
        if not root.exists():
            print(f"[warn] missing root {root}")
            continue
        for p in sorted(x for x in root.rglob("*") if x.suffix.lower() in EXTS and x.is_file()):
            try:
                h = md5_file(p)
            except Exception:
                continue
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            items.append({"idx": len(items), "tag": tag,
                          "sub": p.parent.relative_to(root).as_posix(),
                          "path": str(p), "name": p.name, "md5": h})
    cache.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print(f"[index] unique={len(items)} dup_skipped={dup}")
    return items, dup


if __name__ == "__main__":
    its, d = build_index(force=True)
    from collections import Counter
    print(Counter(i["tag"] for i in its))
    print("dups", d)
