# -*- coding: utf-8 -*-
"""按 来源×图像类型×场景 分层做 7:2:1 划分，产出可直接训练的清单。

输出:
  data/splits/{train,val,test}.txt   每行一个绝对路径
  data/splits/split_summary.md       分布核对表
"""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
SRC = ROOT / "data" / "clip_scene" / "images.csv"
OUT = ROOT / "data" / "splits"
RATIO = (0.7, 0.2, 0.1)
SEED = 2026


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
    print(f"[in] {len(rows)} images")

    groups = defaultdict(list)
    for r in rows:
        key = (r["tag"], r["type"], r["scene"], r["color"])
        groups[key].append(r)

    rnd = random.Random(SEED)
    split = {"train": [], "val": [], "test": []}
    for key, items in groups.items():
        items = sorted(items, key=lambda r: r["path"])
        rnd.shuffle(items)
        n = len(items)
        n_tr = int(round(n * RATIO[0]))
        n_va = int(round(n * RATIO[1]))
        # 小簇保证：≥3 张时至少各 1 张进 val/test
        if n >= 3:
            n_tr = max(1, min(n_tr, n - 2))
            n_va = max(1, min(n_va, n - n_tr - 1))
        split["train"] += items[:n_tr]
        split["val"] += items[n_tr:n_tr + n_va]
        split["test"] += items[n_tr + n_va:]

    for name, items in split.items():
        with open(OUT / f"{name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(r["path"] for r in items) + "\n")
        print(f"[{name}] {len(items)}")

    lines = ["# 7:2:1 分层划分（按 来源×图像类型×场景×颜色 分层，seed=2026）", "",
             f"- 总数 {len(rows)}：train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])}",
             f"- 实际比例 {len(split['train'])/len(rows):.1%} / {len(split['val'])/len(rows):.1%} / {len(split['test'])/len(rows):.1%}", ""]

    for dim in ("tag", "color", "scene", "quality", "type"):
        lines += [f"## 按 {dim}", "", "| 取值 | train | val | test | 合计 |", "|---|---:|---:|---:|---:|"]
        keys = sorted({r[dim] for r in rows})
        for k in keys:
            c = [sum(1 for r in split[s] if r[dim] == k) for s in ("train", "val", "test")]
            lines.append(f"| {k} | {c[0]} | {c[1]} | {c[2]} | {sum(c)} |")
        lines.append("")

    (OUT / "split_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
