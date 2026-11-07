# -*- coding: utf-8 -*-
"""颜色级联识别评测：黑牌走 73 类模型，其余走原 70 类模型；与单模型基线对比。"""
import json, sys
from pathlib import Path
import cv2, numpy as np, torch

sys.path.insert(0, r"D:\desktop\车辆闸机项目\runs_2026-11-02\code")
sys.path.insert(0, r"D:\desktop\复现代码文件夹\yolo-ulm")
from ulm.lprnet import LPRNet, PlateCharset  # noqa
import importlib.util
spec = importlib.util.spec_from_file_location("eval_rec", r"D:\desktop\车辆闸机项目\runs_2026-11-02\code\eval_rec.py")
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M70 = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-02\models\lprnet_final8\best.pt")
M73 = ROOT / "models" / "rec" / "lprnet_special73" / "best.pt"


def hsv_color(b):
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (s >= 70) & (v >= 40)
    if m.sum() < 50:
        return "black" if np.median(v) < 90 else "white"
    ang = h[m] * 2 * np.pi / 180.0
    mean = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * 90 / np.pi) % 180
    if 96 <= mean <= 145: return "blue"
    if 41 <= mean < 96: return "green"
    if 10 <= mean < 41: return "yellow"
    return "black" if np.median(v) < 90 else "white"


def decode2(logits, charset):
    """返回 (文本, 每个字符的置信度)。"""
    prob = logits.softmax(-1)[0]
    idx = prob.argmax(-1).cpu().numpy()
    pr = prob.max(-1).values.cpu().numpy()
    out, conf, prev = [], [], -1
    for c, q in zip(idx, pr):
        if c != 0 and c != prev:
            out.append(int(c)); conf.append(float(q))
        prev = c
    return charset.decode(out), conf


def fit_len(s, conf, L):
    if not L or len(s) <= L:
        return s
    keep = sorted(range(len(s)), key=lambda i: -conf[i])[:L]
    return "".join(s[i] for i in sorted(keep))


def rec_all(model, charset, items, n_views=23):
    out = []
    for it in items:
        b = it.get("crop")
        if b is None:
            b = cv2.imdecode(np.fromfile(it["path"], np.uint8), 1)
        if b is None:
            out.append({"pred": "", "conf": 0.0}); continue
        b = cv2.resize(b, (192, 64))
        views = er.jitter_views(b, n_views)
        logits = er.run_batch(model, views, DEV)
        res = [decode2(l, charset) for l in logits]
        best = max(res, key=lambda r: sum(r[1]) / max(1, len(r[1])))
        out.append({"pred": best[0], "conf": sum(best[1]) / max(1, len(best[1]))})
    return out


def main():
    man = json.loads((ROOT / "data" / "crops_all" / "manifest.json").read_text(encoding="utf-8"))
    for it in man:
        b = cv2.imdecode(np.fromfile(it["path"], np.uint8), 1)
        it["pred_color"] = hsv_color(b) if b is not None else "unknown"

    m70, cs70 = er.load_model(M70, DEV)
    m73, cs73 = er.load_model(M73, DEV)
    r70 = rec_all(m70, cs70, man)
    r73 = rec_all(m73, cs73, man)

    SPEC_CH = set("使领港澳")
    LEN_BY = {"blue": 7, "yellow": 7, "green": 8, "black": None, "white": None, "unknown": None}
    rows = []
    for it, a, b in zip(man, r70, r73):
        L = LEN_BY.get(it["pred_color"])
        p70 = fit_len(a["pred"], a.get("chars", [1.0] * len(a["pred"])), L) if L else a["pred"]
        p73 = b["pred"]
        # 词表闸门：73 类模型输出含 使/领/港/澳 且置信度不低才采纳
        hit = any(c in p73 for c in SPEC_CH) and b["conf"] >= 0.30
        rows.append({**it, "p70": p70, "p73": p73, "pick": "m73" if hit else "m70",
                     "pred": p73 if hit else p70})

    def acc(sel, key="pred"):
        n = len(sel); ok = sum(1 for r in sel if r[key] == r["text"]); return n, ok, ok / max(1, n)

    sp = [r for r in rows if r["source"] == "special"]
    se = [r for r in rows if r["source"] == "self289"]
    print("== 特殊牌 76")
    for k in ("p70", "p73", "pred"):
        n, ok, a = acc(sp, k); print(f"  {k:5s} {ok}/{n} = {a*100:.2f}%")
    print("== 自采 289")
    for k in ("p70", "p73", "pred"):
        n, ok, a = acc(se, k); print(f"  {k:5s} {ok}/{n} = {a*100:.2f}%")
    from collections import Counter
    print("黑牌分流命中数:", Counter(r["pick"] for r in sp))
    print("颜色预测分布(特殊):", Counter(r["pred_color"] for r in sp).most_common())
    (ROOT / "deploy" / "eval_cascade.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

SPEC_CH = set('使领港澳')
