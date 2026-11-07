# -*- coding: utf-8 -*-
"""SQLite 持久化：车辆/余额/月卡、进出场流水、账单、钱包流水、事件日志。

表结构见 init_db()。所有时间以字符串 'YYYY-MM-DD HH:MM:SS' 存储，便于 SQL 分组。
"""
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

import billing

DB_PATH = Path(__file__).resolve().parent / "data" / "gate.db"
_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicle (
  plate TEXT PRIMARY KEY, color TEXT DEFAULT 'blue', car_type TEXT DEFAULT '小型车',
  balance REAL DEFAULT 0, monthly_until TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS session (
  id INTEGER PRIMARY KEY AUTOINCREMENT, plate TEXT, lane INTEGER DEFAULT 1,
  entry_time TEXT, exit_time TEXT, fee REAL DEFAULT 0, rule TEXT DEFAULT 'time',
  status TEXT DEFAULT 'in', abnormal TEXT DEFAULT '', snapshot_in TEXT, snapshot_out TEXT);
CREATE TABLE IF NOT EXISTS bill (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, plate TEXT, amount REAL,
  paid REAL DEFAULT 0, detail TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS wallet (
  id INTEGER PRIMARY KEY AUTOINCREMENT, plate TEXT, amount REAL, kind TEXT,
  balance_after REAL, created_at TEXT);
CREATE TABLE IF NOT EXISTS event (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, lane INTEGER, level TEXT,
  code TEXT, msg TEXT);
CREATE INDEX IF NOT EXISTS idx_session_plate ON session(plate, status);
CREATE INDEX IF NOT EXISTS idx_session_entry ON session(entry_time);
"""

FMT = "%Y-%m-%d %H:%M:%S"


def conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=5)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _lock, conn() as c:
        c.executescript(SCHEMA)
    return DB_PATH


def now_str(dt=None):
    return (dt or datetime.now()).strftime(FMT)


def get_vehicle(plate, create=True):
    with _lock, conn() as c:
        r = c.execute("SELECT * FROM vehicle WHERE plate=?", (plate,)).fetchone()
        if r is None and create:
            c.execute("INSERT INTO vehicle(plate,created_at,updated_at) VALUES(?,?,?)",
                      (plate, now_str(), now_str()))
            r = c.execute("SELECT * FROM vehicle WHERE plate=?", (plate,)).fetchone()
    return dict(r) if r else None


def set_vehicle(plate, **kw):
    if not kw:
        return
    kw["updated_at"] = now_str()
    cols = ", ".join(f"{k}=?" for k in kw)
    with _lock, conn() as c:
        c.execute("UPDATE vehicle SET " + cols + " WHERE plate=?", (*kw.values(), plate))


def recharge(plate, amount):
    v = get_vehicle(plate) or {}
    bal = round(float(v.get("balance") or 0) + float(amount), 2)
    set_vehicle(plate, balance=bal)
    with _lock, conn() as c:
        c.execute("INSERT INTO wallet(plate,amount,kind,balance_after,created_at) VALUES(?,?,?,?,?)",
                  (plate, amount, "recharge", bal, now_str()))
    return bal


def buy_monthly(plate, days=30):
    v = get_vehicle(plate) or {}
    bal = float(v.get("balance") or 0) - billing.MONTHLY_PRICE
    if bal < 0:
        return None, f"余额不足，月卡 {billing.MONTHLY_PRICE:.0f} 元，当前 {v.get('balance', 0):.2f} 元"
    base = datetime.now()
    if v.get("monthly_until"):
        try:
            base = max(base, datetime.strptime(v["monthly_until"], "%Y-%m-%d"))
        except ValueError:
            pass
    until = (base + timedelta(days=days)).strftime("%Y-%m-%d")
    set_vehicle(plate, balance=round(bal, 2), monthly_until=until)
    with _lock, conn() as c:
        c.execute("INSERT INTO wallet(plate,amount,kind,balance_after,created_at) VALUES(?,?,?,?,?)",
                  (plate, -billing.MONTHLY_PRICE, "monthly", bal, now_str()))
    return until, ""


def log(level, code, msg, lane=1, ts=None):
    with _lock, conn() as c:
        c.execute("INSERT INTO event(ts,lane,level,code,msg) VALUES(?,?,?,?,?)",
                  (ts or now_str(), lane, level, code, msg))


def open_session(plate, lane=1):
    with _lock, conn() as c:
        r = c.execute("SELECT * FROM session WHERE plate=? AND status='in' "
                      "ORDER BY id DESC LIMIT 1", (plate,)).fetchone()
    return dict(r) if r else None


def enter(plate, lane=1, when=None, snapshot="", rule=None):
    """入口：重复入场判定 + 建流水。返回 (session, warn)。"""
    when = when or datetime.now()
    prev = open_session(plate)
    warn = ""
    if prev:
        gap = (when - datetime.strptime(prev["entry_time"], FMT)).total_seconds()
        if gap < 120:
            warn = f"疑似重复入场/跟车：{gap:.0f} 秒前已入场"
            log("warn", "DUPLICATE_ENTRY", f"{plate} {warn}", lane)
            return prev, warn
        warn = f"该车上次未出场（{prev['entry_time']}），本次按重复入场登记"
        with _lock, conn() as c:
            c.execute("UPDATE session SET status='abnormal', abnormal=?, exit_time=? WHERE id=?",
                      ("重复入场未出场", now_str(when), prev["id"]))
        log("error", "REENTRY", f"{plate} {warn}", lane)
    v = get_vehicle(plate)
    rule = rule or ("monthly" if v and v.get("monthly_until") else "time")
    with _lock, conn() as c:
        cur = c.execute("INSERT INTO session(plate,lane,entry_time,status,rule,snapshot_in) "
                        "VALUES(?,?,?,?,?,?)", (plate, lane, when.strftime(FMT), "in", rule, snapshot))
        sid = cur.lastrowid
        r = c.execute("SELECT * FROM session WHERE id=?", (sid,)).fetchone()
    log("info", "ENTER", f"{plate} 入场", lane, when.strftime(FMT))
    return dict(r), warn


def exit_vehicle(plate, lane=1, when=None, snapshot="", pay=True):
    """出口：结算。返回 (session, bill_dict, msg)。"""
    when = when or datetime.now()
    s = open_session(plate)
    if s is None:
        log("error", "NO_ENTRY", f"{plate} 无入场记录", lane)
        return None, None, "无入场记录，按异常处理（需人工登记入场时间）"
    entry = datetime.strptime(s["entry_time"], FMT)
    v = get_vehicle(plate) or {}
    m_until = None
    if v.get("monthly_until"):
        m_until = datetime.strptime(v["monthly_until"], "%Y-%m-%d").date()
    b = billing.calc_fee(plate, entry, when, rule=s["rule"] or "time",
                         monthly_until=m_until, now=when)
    fee = b.fee
    msg = ""
    paid = 0.0
    if fee > 0 and pay:
        bal = float(v.get("balance") or 0)
        if bal + 1e-9 >= fee:
            set_vehicle(plate, balance=round(bal - fee, 2))
            with _lock, conn() as c:
                c.execute("INSERT INTO wallet(plate,amount,kind,balance_after,created_at) "
                          "VALUES(?,?,?,?,?)", (plate, -fee, "parking", round(bal - fee, 2),
                                                when.strftime(FMT)))
            paid = fee
            msg = f"余额扣款 {fee:.2f} 元，剩余 {bal - fee:.2f} 元"
        else:
            msg = f"余额不足（{bal:.2f} < {fee:.2f}），转人工/扫码支付"
            log("warn", "LOW_BALANCE", f"{plate} {msg}", lane)
    with _lock, conn() as c:
        c.execute("UPDATE session SET exit_time=?,fee=?,status='out',snapshot_out=?,abnormal=? "
                  "WHERE id=?", (when.strftime(FMT), fee, snapshot, b.abnormal, s["id"]))
        c.execute("INSERT INTO bill(session_id,plate,amount,paid,detail,created_at) "
                  "VALUES(?,?,?,?,?,?)",
                  (s["id"], plate, fee, paid, json.dumps(b.as_dict(), ensure_ascii=False),
                   when.strftime(FMT)))
        r = c.execute("SELECT * FROM session WHERE id=?", (s["id"],)).fetchone()
    log("info", "EXIT", f"{plate} 出场 收费 {fee:.2f} 元", lane, when.strftime(FMT))
    return dict(r), b.as_dict(), msg


def abnormal_close(plate, lane=1, when=None, note="未识别出场，人工放行"):
    """未识别出场 / 人工放行：结束流水但不产生正常账单。"""
    when = when or datetime.now()
    s = open_session(plate)
    if not s:
        return None
    with _lock, conn() as c:
        c.execute("UPDATE session SET exit_time=?,status='abnormal',abnormal=? WHERE id=?",
                  (when.strftime(FMT), note, s["id"]))
    log("warn", "MANUAL_PASS", f"{plate} {note}", lane)
    return dict(s)


def sessions(plate=None, page=1, size=10, q="", status=None, day=None):
    where, args = [], []
    if plate:
        where.append("plate=?")
        args.append(plate)
    if q:
        where.append("(plate LIKE ? OR abnormal LIKE ?)")
        args += [f"%{q}%", f"%{q}%"]
    if status:
        where.append("status=?")
        args.append(status)
    if day:
        where.append("substr(entry_time,1,10)=?")
        args.append(day)
    w = ("WHERE " + " AND ".join(where)) if where else ""
    with _lock, conn() as c:
        total = c.execute(f"SELECT COUNT(*) n FROM session {w}", args).fetchone()["n"]
        rows = c.execute(f"SELECT * FROM session {w} ORDER BY id DESC LIMIT ? OFFSET ?",
                         (*args, size, (page - 1) * size)).fetchall()
    return [dict(r) for r in rows], total


def in_park():
    with _lock, conn() as c:
        rows = c.execute("SELECT * FROM session WHERE status='in' ORDER BY entry_time").fetchall()
    return [dict(r) for r in rows]


def last_bill():
    """最近一笔结算，用于页面刷新后回显"本次结算"卡片。"""
    with _lock, conn() as c:
        r = c.execute("SELECT * FROM bill ORDER BY id DESC LIMIT 1").fetchone()
    if not r:
        return None
    try:
        detail = json.loads(r["detail"] or "{}")
    except ValueError:
        detail = {}
    return {"plate": r["plate"], "amount": r["amount"], "paid": r["paid"],
            "at": r["created_at"], "bill": detail, "vehicle": get_vehicle(r["plate"]) or {}}


def stats(day=None):
    day = day or datetime.now().strftime("%Y-%m-%d")
    with _lock, conn() as c:
        income = c.execute("SELECT COALESCE(SUM(paid),0) s, COALESCE(SUM(amount),0) due "
                           "FROM bill WHERE substr(created_at,1,10)=?", (day,)).fetchone()
        cnt = c.execute("SELECT COUNT(*) n FROM session WHERE substr(entry_time,1,10)=?", (day,)).fetchone()["n"]
        parked = c.execute("SELECT COUNT(*) n FROM session WHERE status='in'").fetchone()["n"]
        hourly = c.execute(
            "SELECT substr(entry_time,12,2) h, COUNT(*) n FROM session "
            "WHERE substr(entry_time,1,10)=? GROUP BY h ORDER BY h", (day,)).fetchall()
        abn = c.execute("SELECT COUNT(*) n FROM session WHERE status='abnormal' AND substr(entry_time,1,10)=?",
                        (day,)).fetchone()["n"]
    hours = [{"h": f"{i:02d}", "n": 0} for i in range(24)]
    for r in hourly:
        hours[int(r["h"])]["n"] = r["n"]
    return {"date": day, "income": round(income["s"], 2), "due": round(income["due"], 2),
            "unpaid": round(income["due"] - income["s"], 2),
            "entries": cnt, "parked": parked, "abnormal": abn, "hourly": hours}


def report(period="day", ref=None):
    """period: day | week | month，返回按日聚合的收入/车次。"""
    base = datetime.strptime(ref, "%Y-%m-%d") if ref else datetime.now()
    if period == "week":
        start = base - timedelta(days=base.weekday())
        n = 7
    elif period == "month":
        start = base.replace(day=1)
        nxt = (start + timedelta(days=32)).replace(day=1)
        n = (nxt - start).days
    else:
        start, n = base, 1
    days = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]
    out = []
    with _lock, conn() as c:
        for d in days:
            inc = c.execute("SELECT COALESCE(SUM(paid),0) s, COALESCE(SUM(amount),0) due, "
                            "COUNT(*) n FROM bill WHERE substr(created_at,1,10)=?", (d,)).fetchone()
            ent = c.execute("SELECT COUNT(*) n FROM session WHERE substr(entry_time,1,10)=?",
                            (d,)).fetchone()["n"]
            out.append({"date": d, "income": round(inc["s"], 2), "due": round(inc["due"], 2),
                        "bills": inc["n"], "entries": ent})
    total = round(sum(r["income"] for r in out), 2)
    due = round(sum(r["due"] for r in out), 2)
    return {"period": period, "start": days[0], "end": days[-1],
            "total_income": total, "total_due": due, "total_unpaid": round(due - total, 2),
            "total_entries": sum(r["entries"] for r in out), "rows": out}


def events(limit=30):
    with _lock, conn() as c:
        rows = c.execute("SELECT * FROM event ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def seed():
    """演示数据：三辆有余额的车 + 一张月卡 + 一张已出场流水。"""
    init_db()
    with _lock, conn() as c:
        n = c.execute("SELECT COUNT(*) n FROM vehicle").fetchone()["n"]
    if n:
        return
    demo = [("皖A88888", "blue", 200.0, None),
            ("皖A66666", "green", 50.0, (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d")),
            ("皖A12345", "blue", 500.0, None)]
    for p, col, bal, mu in demo:
        get_vehicle(p)
        set_vehicle(p, balance=bal, color=col, monthly_until=mu)
    log("info", "SYSTEM", "系统初始化完成")


if __name__ == "__main__":
    seed()
    print("db:", DB_PATH)
    print(stats())
