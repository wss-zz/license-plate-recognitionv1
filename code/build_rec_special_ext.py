# -*- coding: utf-8 -*-
"""构建 73 类（补 使/领/港/澳）识别训练集：CRPD 主体 + 特殊牌合成 + CCPD 蓝→黑。"""
import json
from pathlib import Path

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
OUT = ROOT / "data" / "rec_special_ext"
OUT.mkdir(parents=True, exist_ok=True)

old = json.loads((Path(r"D:\desktop\车辆闸机项目\runs_2026-11-02\data\final_rec") / "charset.json").read_text(encoding="utf-8"))
chars = old["chars"]
for ch in "使领港澳":
    if ch not in chars:
        chars += ch
(OUT / "charset.json").write_text(json.dumps({
    "chars": chars, "num_classes": len(chars) + 1, "crop": [94, 24],
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("charset:", len(chars), "classes:", len(chars) + 1)

def read_tsv(p):
    rows = []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            a, b = line.split("\t", 1)
            if Path(a).exists() and b.strip():
                rows.append((a, b.strip()))
    return rows

tr = read_tsv(ROOT / "data" / "rec_special" / "train.tsv")
va = read_tsv(ROOT / "data" / "rec_special" / "val.tsv")
sp = read_tsv(ROOT / "data" / "special_synth" / "train.tsv")
spv = read_tsv(ROOT / "data" / "special_synth" / "val.tsv") if (ROOT / "data" / "special_synth" / "val.tsv").exists() else []
sb = [(m["path"], m["text"]) for m in json.loads((ROOT / "data" / "synth_black" / "manifest.json").read_text(encoding="utf-8"))]
print("crpd", len(tr), "crpd_val", len(va), "special_synth", len(sp), "special_val", len(spv), "synth_black", len(sb))

rows_tr = tr + sp + sb
rows_va = va + spv
(OUT / "train.tsv").write_text("\n".join(f"{a}\t{b}" for a, b in rows_tr), encoding="utf-8")
(OUT / "val.tsv").write_text("\n".join(f"{a}\t{b}" for a, b in rows_va), encoding="utf-8")
print("train", len(rows_tr), "val", len(rows_va), "->", OUT)
