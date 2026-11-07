# -*- coding: utf-8 -*-
"""闸机系统服务：标准库 http.server，零第三方运行时依赖。

启动：python server.py [--port 8088] [--no-tts]
浏览器打开 http://127.0.0.1:8088
"""
import argparse
import json
import mimetypes
import random
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import billing
import inference
import store
import tts

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
SNAP = BASE / "data" / "snaps"
SNAP.mkdir(parents=True, exist_ok=True)

TTS_ON = True
_infer_lock = threading.Lock()
_lane_lock = threading.Lock()
LANES = {}
PLATE_IMG = {}   # plate -> 入场画面路径（演示出口复现同一辆车）


# ------------------------------------------------------------------ 图片池
def build_pool(limit=600):
    roots = [r"D:\车牌数据集_20261105", r"D:\desktop\车辆闸机项目\plate_dataset_yungui\images",
             r"D:\新车牌数据集", r"D:\车牌数据集", r"D:\车牌数据集_补充"]
    out = []
    for r in roots:
        p = Path(r)
        if not p.exists():
            continue
        out += [str(f) for f in p.rglob("*.jpg")]
    random.seed(7)
    random.shuffle(out)
    return out[:limit]


POOL = build_pool()
POOL_I = {"i": 0}


def next_image():
    if not POOL:
        return None
    p = POOL[POOL_I["i"] % len(POOL)]
    POOL_I["i"] += 1
    return p


# ------------------------------------------------------------------ 车道
def lane(i):
    with _lane_lock:
        if i not in LANES:
            LANES[i] = {"id": i, "name": f"{i} 号车道", "state": "idle", "msg": "待机",
                        "plate": "", "color": "", "conf": 0.0, "gate": 0, "snap": "",
                        "fee": None, "bill": None, "entry": "", "alarm": "",
                        "ts": datetime.now().strftime("%H:%M:%S")}
        return dict(LANES[i])


def set_lane(i, **kw):
    with _lane_lock:
        LANES.setdefault(i, {"id": i, "name": f"{i} 号车道", "state": "idle", "msg": "待机",
                             "plate": "", "color": "", "conf": 0.0, "gate": 0, "snap": "",
                             "fee": None, "bill": None, "entry": "", "alarm": "",
                             "ts": ""})
        LANES[i].update(kw)
        LANES[i]["ts"] = datetime.now().strftime("%H:%M:%S")
        return dict(LANES[i])


def _close_later(i, secs=4.0):
    def _f():
        time.sleep(secs)
        with _lane_lock:
            cur = LANES.get(i, {})
        if cur.get("state") == "open":
            set_lane(i, gate=0, state="closed", msg="车辆已通过，闸杆落下")
            time.sleep(1.2)
            set_lane(i, state="idle", msg="待机")
    threading.Thread(target=_f, daemon=True).start()


def snap_save(img, lane_id):
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]}_{lane_id}.jpg"
    p = SNAP / name
    inference.save_jpg(img, p)
    return f"/media/{name}"


# ------------------------------------------------------------------ 业务动作
def do_recognize(lane_id, mode, path=None):
    path = path or next_image()
    if not path:
        return {"ok": False, "msg": "演示图库为空"}
    img = inference.imread(path)
    if img is None:
        return {"ok": False, "msg": f"读取失败 {path}"}
    with _infer_lock:
        r = inference.recognize(img)
    plates = r["plates"]
    snap = snap_save(r["annotated"], lane_id)
    if not plates:
        set_lane(lane_id, state="alarm", alarm="NO_PLATE", plate="", msg="未识别到车牌，转人工处理",
                 snap=snap, gate=0)
        store.log("error", "NO_PLATE", f"车道{lane_id} 未识别到车牌", lane_id)
        return {"ok": False, "msg": "未识别到车牌", "snap": snap, "ms": r["ms"], "lane": lane(lane_id)}
    p = plates[0]
    return {"ok": True, "plate": p["plate"], "color": p["color"], "conf": p["conf"],
            "snap": snap, "ms": r["ms"], "n": len(plates), "rect": p.get("rect"),
            "path": path, "source": Path(path).name}


def op_enter(lane_id, plate, color, conf, snap):
    cur = lane(lane_id)
    if cur.get("state") == "open":
        store.log("warn", "TAILGATING", f"车道{lane_id} 跟车告警：{plate}", lane_id)
        set_lane(lane_id, alarm="TAILGATING", msg=f"跟车告警：闸杆未落又有车（{plate}）")
    s, warn = store.enter(plate, lane_id, snapshot=snap)
    v = store.get_vehicle(plate) or {}
    mu = v.get("monthly_until")
    if mu:
        msg = f"{plate} 月卡车辆，请通行"
    elif warn:
        msg = f"{plate} {warn}"
    else:
        msg = f"{plate} 欢迎光临"
    set_lane(lane_id, state="open", gate=1, plate=plate, color=color or "blue",
             conf=conf or 0, msg=msg, snap=snap, alarm="",
             entry=s.get("entry_time", "") if s else "", fee=None, bill=None)
    if TTS_ON:
        tts.speak(msg.replace("粤", "yue"))
    _close_later(lane_id)
    return {"ok": True, "msg": msg, "warn": warn, "lane": lane(lane_id),
            "vehicle": v, "session": s}


def op_exit(lane_id, plate, color, conf, snap):
    s, bill, msg = store.exit_vehicle(plate, lane_id, snapshot=snap)
    v = store.get_vehicle(plate) or {}
    if s is None:
        set_lane(lane_id, state="alarm", alarm="NO_ENTRY", plate=plate, msg=msg, snap=snap, gate=0)
        return {"ok": False, "msg": msg, "lane": lane(lane_id)}
    mins = bill["parked_min"]
    dur = f"{mins // 60}小时{mins % 60}分" if mins >= 60 else f"{mins}分钟"
    if bill.get("monthly_hit"):
        say = f"{plate}，月卡车辆，一路平安"
    elif bill["fee"] <= 0:
        say = f"{plate}，停车{dur}，无需缴费，一路平安"
    elif "余额不足" in (msg or ""):
        say = f"{plate}，停车{dur}，需缴费{bill['fee']:.2f}元，请扫码支付"
    else:
        say = f"{plate}，停车{dur}，收费{bill['fee']:.2f}元，一路平安"
    set_lane(lane_id, state="open", gate=1, plate=plate, color=color or "blue",
             conf=conf or 0, msg=say, snap=snap, alarm="", fee=bill["fee"], bill=bill)
    if TTS_ON:
        tts.speak(say)
    _close_later(lane_id)
    store.log("info", "BILL", f"{plate} {dur} 收费 {bill['fee']:.2f} 元 余额 {v.get('balance', 0):.2f}", lane_id)
    return {"ok": True, "msg": msg, "bill": bill, "lane": lane(lane_id), "vehicle": v}


def op_auto(body):
    lane_id = int(body.get("lane", 1))
    mode = body.get("mode", "in")
    path = None
    if mode == "out":
        # 出口相机应看到"同一辆车"：优先复用该车入场时的画面，保证车牌一致
        cand = [p for p in store.in_park() if p["plate"] in PLATE_IMG]
        if cand:
            path = PLATE_IMG[cand[0]["plate"]]   # 优先让停放最久的车出场，贴近真实
    rec = do_recognize(lane_id, mode, path)
    if not rec.get("ok"):
        return {"ok": False, "msg": rec.get("msg"), "lane": rec.get("lane") or lane(lane_id),
                "snap": rec.get("snap", ""), "ms": rec.get("ms")}
    if mode == "in":
        out = op_enter(lane_id, rec["plate"], rec["color"], rec["conf"], rec["snap"])
        PLATE_IMG[rec["plate"]] = rec.get("path", "")
    else:
        out = op_exit(lane_id, rec["plate"], rec["color"], rec["conf"], rec["snap"])
    out.update({"ms": rec["ms"], "snap": rec["snap"], "plate": rec["plate"],
                "color": rec["color"], "conf": rec["conf"], "source": rec["source"]})
    return out


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "GateSystem/1.0"

    def log_message(self, fmt, *args):  # 降噪
        pass

    # ---------- 工具
    def _send(self, obj, code=200, ctype="application/json; charset=utf-8"):
        data = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            raw = self.rfile.read(n).decode("utf-8-sig").lstrip("\ufeff")
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            return {}

    def _file(self, path: Path):
        if not path.exists() or not path.is_file():
            return self._send({"error": "not found"}, 404)
        ct = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send(path.read_bytes(), 200, ct)

    # ---------- GET
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path
        if p in ("/", "/index.html"):
            return self._file(STATIC / "index.html")
        if p.startswith("/media/"):
            return self._file(SNAP / Path(p).name)
        if p.startswith("/static/"):
            return self._file(STATIC / Path(p).name)
        if p == "/api/state":
            return self._send({"lanes": [lane(1), lane(2)],
                               "stats": store.stats(),
                               "parked": store.in_park(),
                               "vehicles": _vehicles(),
                               "rules": billing.rule_list(),
                               "pool": len(POOL),
                               "last_bill": store.last_bill(),
                               "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        if p == "/api/records":
            page = int(q.get("page", ["1"])[0])
            size = int(q.get("size", ["8"])[0])
            rows, total = store.sessions(q=q.get("q", [""])[0],
                                         status=q.get("status", [None])[0],
                                         page=page, size=size)
            for r in rows:
                r["snap_in"] = f"/media/{Path(r['snapshot_in']).name}" if r.get("snapshot_in") else ""
                r["snap_out"] = f"/media/{Path(r['snapshot_out']).name}" if r.get("snapshot_out") else ""
            return self._send({"rows": rows, "total": total, "page": page, "size": size})
        if p == "/api/stats":
            return self._send(store.stats(q.get("date", [None])[0]))
        if p == "/api/report":
            return self._send(store.report(q.get("period", ["day"])[0], q.get("ref", [None])[0]))
        if p == "/api/events":
            return self._send({"rows": store.events(40)})
        if p == "/api/billing_cases":
            return self._send(_run_cases())
        if p == "/api/parked":
            return self._send({"rows": store.in_park()})
        return self._send({"error": "not found"}, 404)

    # ---------- POST
    def do_POST(self):
        u = urlparse(self.path)
        b = self._body()
        p = u.path
        try:
            if p == "/api/auto":
                return self._send(op_auto(b))
            if p == "/api/enter":
                return self._send(op_enter(int(b.get("lane", 1)), b["plate"], b.get("color", "blue"),
                                           float(b.get("conf", 1.0)), ""))
            if p == "/api/exit":
                return self._send(op_exit(int(b.get("lane", 1)), b["plate"], b.get("color", "blue"),
                                          float(b.get("conf", 1.0)), ""))
            if p == "/api/manual_out":
                plate = b["plate"]
                note = b.get("note", "出口未识别车牌，人工核对放行")
                s = store.abnormal_close(plate, int(b.get("lane", 1)), note=note)
                set_lane(int(b.get("lane", 1)), state="open", gate=1, plate=plate,
                         msg=f"{plate} 人工放行（异常登记）", alarm="MANUAL")
                _close_later(int(b.get("lane", 1)))
                if TTS_ON:
                    tts.speak(f"{plate}，人工放行")
                return self._send({"ok": bool(s), "msg": note, "lane": lane(int(b.get("lane", 1)))})
            if p == "/api/recharge":
                bal = store.recharge(b["plate"], float(b["amount"]))
                return self._send({"ok": True, "balance": bal})
            if p == "/api/monthly":
                until, err = store.buy_monthly(b["plate"], int(b.get("days", 30)))
                return self._send({"ok": not err, "until": until, "msg": err})
            if p == "/api/tts":
                ok = tts.speak(b.get("text", "欢迎光临"), block=False) and TTS_ON
                return self._send({"ok": ok})
            if p == "/api/alarm":
                code = b.get("code", "TAILGATING")
                txt = {"TAILGATING": "跟车告警：请勿尾随前车", "NO_DROP": "闸杆未落，请检查",
                       "NO_PLATE": "未识别到车牌"}.get(code, code)
                set_lane(int(b.get("lane", 1)), alarm=code, msg=txt)
                store.log("warn", code, txt, int(b.get("lane", 1)))
                if TTS_ON:
                    tts.speak(txt)
                return self._send({"ok": True, "msg": txt})
            if p == "/api/backdate":
                # 演示用：把该车在场流水的入场时间往前拨 N 分钟（模拟长时间停放）
                plate = b["plate"]
                mins = int(b.get("minutes", 60))
                s = store.open_session(plate)
                if not s:
                    return self._send({"ok": False, "msg": "该车不在场"})
                new_t = (datetime.strptime(s["entry_time"], store.FMT)
                         - timedelta(minutes=mins)).strftime(store.FMT)
                with store._lock, store.conn() as c:  # noqa: SLF001
                    c.execute("UPDATE session SET entry_time=? WHERE id=?", (new_t, s["id"]))
                return self._send({"ok": True, "entry_time": new_t, "msg": f"入场时间回拨 {mins} 分钟"})
            if p == "/api/clear_alarm":
                li = int(b.get("lane", 1))
                set_lane(li, alarm="", state="idle", msg="待机", plate="", snap="", gate=0)
                return self._send({"ok": True, "lane": lane(li)})
        except KeyError as e:
            return self._send({"ok": False, "msg": f"缺少参数 {e}"}, 400)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return self._send({"ok": False, "msg": f"内部错误 {e}"}, 500)
        return self._send({"error": "not found"}, 404)


def _vehicles():
    with store._lock, store.conn() as c:  # noqa: SLF001
        rows = c.execute("SELECT * FROM vehicle ORDER BY updated_at DESC LIMIT 20").fetchall()
    return [dict(r) for r in rows]


def _run_cases():
    """把计费用例跑一遍回传给前端展示（测试即产品文档）。"""
    import importlib.util
    p = BASE / "tests" / "test_billing.py"
    spec = importlib.util.spec_from_file_location("tbc", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = []
    for name, plate, rule, tin, tout, mu, want, why in mod.CASES:
        b = billing.calc_fee(plate, tin, tout, rule=rule, monthly_until=mu, now=tout or tin)
        got = None if (b.abnormal and b.exit is None) else b.fee
        ok = (want is None and b.abnormal) or (want is not None and got is not None
                                               and abs(got - want) < 1e-6 and not b.abnormal)
        out.append({"name": name, "want": want, "got": got, "ok": ok, "why": why,
                    "detail": b.as_dict()})
    return {"cases": out, "passed": sum(1 for r in out if r["ok"]), "total": len(out)}


def main():
    global TTS_ON
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--no-tts", action="store_true")
    args = ap.parse_args()
    TTS_ON = not args.no_tts
    store.seed()
    print(f"[gate] {len(POOL)} 张演示图库")
    threading.Thread(target=inference.load, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[gate] http://127.0.0.1:{args.port}  (Ctrl+C 退出)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[gate] bye")


if __name__ == "__main__":
    main()

