# -*- coding: utf-8 -*-
"""从 CCPD 蓝牌合成黑底牌：GT 框内 蓝→黑，保字符为白，用于补齐特殊牌的背景域。"""
import json, random
from pathlib import Path
import cv2, numpy as np

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
BASE = Path(r"D:\CCPD\dataset\CCPD2019\ccpd_base")
OUT = ROOT / "data" / "synth_black"
OUT.mkdir(parents=True, exist_ok=True)

PROV = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ"
NUM = "0123456789"
ADS = ALPHA + NUM

def decode(name):
    try:
        p = name.split("-")
        q = [tuple(map(int, xy.split("&"))) for xy in p[3].split("_")]
        codes = p[4].split("_")
        prov = PROV[int(codes[0]) - 1]
        ch = ALPHA[int(codes[1])]
        rest = "".join(ADS[int(c)] for c in codes[2:])
        return prov + ch + rest, q
    except Exception:
        return None, None

def order(q):
    q = np.array(q, np.float32); s = q.sum(1); d = np.diff(q, axis=1).ravel()
    r = np.zeros((4, 2), np.float32)
    r[0] = q[np.argmin(s)]; r[2] = q[np.argmax(s)]
    r[1] = q[np.argmin(d)]; r[3] = q[np.argmax(d)]
    return r

def warp(img, q, w=192, h=64):
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(order(q), dst), (w, h))

def blue2black(b):
    """蓝底 → 黑底：只改蓝色背景，白色字符（高 V）原样保留。"""
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV).astype(np.int32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bg = (h >= 88) & (h <= 145) & (s >= 70) & (v >= 40) & (v <= 185)
    m = bg.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.GaussianBlur(m, (5, 5), 0).astype(np.float32)[..., None] / 255.0
    out = b.astype(np.float32) * (1 - m) + np.array([10, 10, 12], np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)

def rd(p): return cv2.imdecode(np.fromfile(p, np.uint8), 1)
def wr(p, img): cv2.imencode(".png", img)[1].tofile(p)

def main(n=120, seed=2026):
    used = set()
    for f in ("data/ccpd_eval/rec_manifest.json", "data/ccpd2000/train.txt", "data/ccpd2000/val.txt"):
        p = ROOT / f
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8")) if f.endswith(".json") else [l.strip() for l in p.read_text(encoding="utf-8").splitlines()]
                used |= {Path(x["file"] if isinstance(x, dict) else x).stem for x in d}
            except Exception:
                pass
    files = [p for p in BASE.iterdir() if p.suffix.lower() == ".jpg" and p.stem not in used]
    random.seed(seed); random.shuffle(files)
    items, shown = [], []
    for p in files:
        if len(items) >= n:
            break
        txt, q = decode(p.stem)
        if not txt or len(txt) != 7 or not q:
            continue
        b = rd(str(p))
        if b is None:
            continue
        c = warp(b, q)
        if c is None or c.size == 0:
            continue
        blk = blue2black(c)
        if (cv2.cvtColor(blk, cv2.COLOR_BGR2GRAY) > 120).mean() > 0.45:  # 仍偏亮，丢弃
            continue
        outp = OUT / f"synth_{len(items):04d}.png"
        wr(str(outp), blk)
        items.append({"path": str(outp), "text": txt, "color": "black", "src": p.name})
        if len(shown) < 12:
            shown.append((c, blk, txt))
    (OUT / "manifest.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    # 拼图：上=原蓝牌，下=合成黑牌
    H, W = 64, 192
    sheet = np.full((2 * (2 * H + 8) + 18, 4 * (W + 6), 3), 255, np.uint8)
    for i, (c, blk, t) in enumerate(shown[:8]):
        r, col = divmod(i, 4)
        x = col * (W + 6); y = r * (2 * H + 8)
        sheet[y:y + H, x:x + W] = c
        sheet[y + H:y + 2 * H, x:x + W] = blk
        cv2.putText(sheet, t, (x + 2, y + 2 * H + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    wr(str(ROOT / "figs" / "synth_black_check.jpg"), sheet)
    print("synthesized:", len(items), "-> ", OUT)
    for it in items[:5]:
        print("  ", it["text"], it["src"])

if __name__ == "__main__":
    main()
