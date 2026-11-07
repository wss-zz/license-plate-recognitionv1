"""用 Edge DevTools 协议录制系统界面演示视频（无需外部录屏软件）。

思路：headless Edge 打开系统页面 → Page.startScreencast 持续收帧 →
按脚本驱动业务接口（入场 / 结算 / 月卡 / 余额不足 / 跟车告警 / 报表）→
把帧按时间戳写入 mp4。

用法：
    python record_demo.py            # 输出 report/演示视频.mp4
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import websocket  # websocket-client

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "report" / "演示视频.mp4"
FPS = 10
W, H = 1440, 900
PORT = 9222
URL = "http://127.0.0.1:8088/"

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def api(path: str, body=None, base="http://127.0.0.1:8088"):
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    return json.loads(op.open(req, timeout=180).read().decode())


def find_ws():
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(60):
        try:
            tabs = json.loads(op.open(f"http://127.0.0.1:{PORT}/json", timeout=3).read())
            for t in tabs:
                if t.get("type") == "page":
                    return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("未找到可调试页面")


def edge_exe():
    for p in EDGE_CANDIDATES:
        if Path(p).exists():
            return p
    raise RuntimeError("未找到 msedge.exe")


class Recorder:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=30, max_size=32 * 2 ** 20)
        self.frames: list[tuple[float, bytes]] = []
        self.alive = True
        self._id = 0
        self._send({"id": 1, "method": "Page.enable"})
        self._send({"id": 2, "method": "Page.startScreencast",
                    "params": {"format": "jpeg", "quality": 82, "maxWidth": W, "maxHeight": H}})
        self.t0 = time.time()
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _send(self, obj):
        self.ws.send(json.dumps(obj))

    def _pump(self):
        while self.alive:
            try:
                msg = json.loads(self.ws.recv())
            except Exception:
                break
            if msg.get("method") == "Page.screencastFrame":
                p = msg["params"]
                self.frames.append((time.time() - self.t0, base64.b64decode(p["data"])))
                self._send({"id": int(p["sessionId"]) + 1000, "method": "Page.screencastFrameAck",
                            "params": {"sessionId": p["sessionId"]}})

    def js(self, expr: str):
        self._id += 1
        self._send({"id": self._id, "method": "Runtime.evaluate",
                    "params": {"expression": expr, "awaitPromise": False}})
        time.sleep(0.2)

    def stop(self):
        self.alive = False
        try:
            self._send({"id": 9999, "method": "Page.stopScreencast"})
        except Exception:
            pass
        time.sleep(0.2)
        try:
            self.ws.close()
        except Exception:
            pass


def write_video(frames, ffmpeg, out: Path, fps=FPS):
    if not frames:
        raise RuntimeError("没有采集到任何帧")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / "_frames"
    tmp.mkdir(exist_ok=True)
    for f in tmp.glob("*.jpg"):
        f.unlink()
    # 以固定帧率重采样：按时间轴取最近帧
    duration = frames[-1][0]
    total = max(2, int(duration * fps))
    idx = 0
    for i in range(total):
        t = i / fps
        while idx + 1 < len(frames) and frames[idx + 1][0] <= t:
            idx += 1
        (tmp / f"f{i:05d}.jpg").write_bytes(frames[idx][1])
    cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", str(tmp / "f%05d.jpg"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=white",
           str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def main():
    ffmpeg = json.loads(subprocess.run(
        [sys.executable, "-c", "import imageio_ffmpeg,json;print(json.dumps(imageio_ffmpeg.get_ffmpeg_exe()))"],
        capture_output=True, text=True).stdout.strip())
    # 1) 干净启动浏览器
    proc = subprocess.Popen([edge_exe(), "--headless=new", f"--remote-debugging-port={PORT}",
                             "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                             "--user-data-dir=" + str(ROOT / "data" / "_edge_profile"),
                             f"--window-size={W},{H}", URL],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rec = None
    try:
        rec = Recorder(find_ws())
        time.sleep(3)  # 让首屏渲染完

        def step(name, wait=1.8):
            print(f"[{time.time()-rec.t0:5.1f}s] {name}")
            time.sleep(wait)

        step("空场待机（双车道闸机界面）", 2.4)
        a = api("/api/auto", {"lane": 1, "mode": "in"})
        plate = a["plate"]
        step(f"1 号道入场识别：{plate}", 3.4)
        api("/api/recharge", {"plate": plate, "amount": 50})
        step("账户储值 50 元", 2.2)
        api("/api/backdate", {"plate": plate, "minutes": 95})
        step("模拟停放 95 分钟", 2.2)
        r = api("/api/auto", {"lane": 1, "mode": "out"})
        step(f"出场结算 ￥{r['bill']['fee']:.2f}（分时段计费）", 3.6)
        step("账单分段明细：白天 94 分钟 / 计费 79 分钟", 2.6)
        api("/api/recharge", {"plate": "皖A66666", "amount": 0})
        api("/api/enter", {"lane": 2, "plate": "皖A66666", "rule": "monthly"})
        step("2 号道：月卡车辆入场", 2.6)
        e = api("/api/exit", {"lane": 2, "plate": "皖A66666"})
        step(f"月卡车辆出场 ￥{e['bill']['fee']:.2f}（有效期内免费）", 3.0)
        api("/api/auto", {"lane": 1, "mode": "in"})
        step("1 号道：第二辆车入场", 2.6)
        api("/api/alarm", {"lane": 1, "code": "FOLLOW", "msg": "跟车告警"})
        step("异常处置：跟车告警", 3.0)
        api("/api/clear_alarm", {"lane": 1})
        step("确认后复位", 2.0)
        rec.js("document.querySelector(\'#tabs button[data-p=\"cases\"]\').click()")
        step("计费规则与 18 组用例（全部通过）", 4.0)
        rec.js("document.querySelector(\'#tabs button[data-p=\"rep\"]\').click()")
        step("统计报表：按日聚合收入", 3.4)
        rec.js("document.querySelector(\'#tabs button[data-p=\"evt\"]\').click()")
        step("事件日志：告警与人工干预留痕", 3.4)
        rec.js("document.querySelector(\'#tabs button[data-p=\"rec\"]\').click()")
        step("回到通行记录，演示结束", 2.0)
    finally:
        if rec:
            rec.stop()
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:
            proc.kill()

    if rec and rec.frames:
        out = write_video(rec.frames, ffmpeg, OUT)
        print("saved:", out, f"{len(rec.frames)} 帧")
    else:
        print("采集失败")


if __name__ == "__main__":
    main()
