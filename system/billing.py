# -*- coding: utf-8 -*-
"""计费引擎：三种规则 + 免费时长 + 昼夜分段 + 每 24h 封顶 + 月卡。

规则（可在 RULES 中改参数，改后 test_billing.py 会同步校验）：
  time     计时：15 分钟免费；8:00-20:00 5 元/小时，20:00-8:00 3 元/小时；
           按 30 分钟为一个计费单位（不足 30 分钟按 30 分钟计，按分段分别取整）；
           自入场时刻起每满 24 小时封顶 60 元。
  per_visit 按次：10 元/次，15 分钟内免费。
  monthly  月卡：有效期内 0 元；过期按 time 规则。

所有时间均为本地时间（naive datetime）。金额单位：元，保留 2 位小数。
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math

FREE_MIN = 15
DAY_START, DAY_END = 8, 20
RATE_DAY, RATE_NIGHT = 5.0, 3.0
CAP_PER_24H = 60.0
PER_VISIT_FEE = 10.0
MONTHLY_PRICE = 300.0
UNIT_MIN = 30
MAX_DAYS = 60

RULES = {
    "time": "计时收费（15 分钟免费，白天 5 元/h，夜间 3 元/h，每 24h 封顶 60 元）",
    "per_visit": "按次收费（10 元/次，15 分钟内免费）",
    "monthly": "月卡（300 元/月，有效期内免费）",
}


@dataclass
class Segment:
    kind: str          # day / night
    start: datetime
    minutes: int       # 该段实际停放分钟
    bill_min: int      # 扣除免费后进入计费的分钟
    units: int         # 计费单位个数（30 分钟）
    rate: float
    fee: float


@dataclass
class Bill:
    plate: str
    rule: str
    entry: datetime
    exit: datetime
    parked_min: int = 0
    fee: float = 0.0
    uncapped: float = 0.0
    cap: float = 0.0
    free_min: int = 0
    segments: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    abnormal: str = ""     # 非空表示无法正常结算（如未识别出场）
    monthly_hit: bool = False

    def as_dict(self):
        return {
            "plate": self.plate, "rule": self.rule,
            "entry": self.entry.strftime("%Y-%m-%d %H:%M:%S"),
            "exit": self.exit.strftime("%Y-%m-%d %H:%M:%S") if self.exit else "",
            "parked_min": self.parked_min, "fee": round(self.fee, 2),
            "uncapped": round(self.uncapped, 2), "cap": round(self.cap, 2),
            "free_min": self.free_min, "monthly_hit": self.monthly_hit,
            "abnormal": self.abnormal, "notes": self.notes,
            "segments": [{"kind": s.kind, "from": s.start.strftime("%H:%M"),
                          "to": (s.start + timedelta(minutes=s.minutes)).strftime("%H:%M"),
                          "minutes": s.minutes, "bill_min": s.bill_min,
                          "units": s.units, "rate": s.rate, "fee": round(s.fee, 2)}
                         for s in self.segments],
        }


def _rate(hour):
    return RATE_DAY if DAY_START <= hour < DAY_END else RATE_NIGHT


def _kind(hour):
    return "day" if DAY_START <= hour < DAY_END else "night"


def _split(entry, exit_):
    """把 [entry, exit) 按昼夜与整点切开，返回 [(kind, start, minutes)]。"""
    out = []
    t = entry
    while t < exit_:
        # 当前段结束点：下一个整点，或昼夜切换点，或出场时刻
        nxt = (t + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        if nxt.hour == DAY_START or nxt.hour == DAY_END:
            pass
        cand = [nxt, exit_]
        for h in (DAY_START, DAY_END):
            switch = t.replace(hour=h, minute=0, second=0, microsecond=0)
            if switch <= t:
                switch += timedelta(days=1)
            cand.append(switch)
        end = min(cand)
        out.append((_kind(t.hour), t, int((end - t).total_seconds() // 60)))
        t = end
    # 合并相邻同类段（整点切分只为算时长，计费单位应按"连续白天/夜间"整体取整）
    merged = []
    for kind, st, mi in out:
        if merged and merged[-1][0] == kind:
            merged[-1][2] += mi
        else:
            merged.append([kind, st, mi])
    return merged


def calc_fee(plate, entry, exit_, rule="time", monthly_until=None, now=None):
    """计算停车费。entry/exit_ 为 datetime；monthly_until 为月卡到期日（date 或 None）。"""
    b = Bill(plate=plate, rule=rule, entry=entry, exit=exit_)
    if exit_ is None:
        b.abnormal = "未识别出场"
        b.notes.append("出口未识别到车牌，需人工核对后补录出场时间")
        return b
    if exit_ < entry:
        b.abnormal = "出场时间早于入场时间"
        return b

    total_min = int((exit_ - entry).total_seconds() // 60)
    b.parked_min = total_min
    if total_min > MAX_DAYS * 1440:
        b.notes.append(f"停车超过 {MAX_DAYS} 天，请人工处理")
        total_min = MAX_DAYS * 1440

    # 月卡优先
    if rule == "monthly" or monthly_until is not None:
        ref = (now or exit_).date()
        if monthly_until is not None and ref <= monthly_until:
            b.monthly_hit = True
            b.free_min = total_min
            b.fee = 0.0
            b.notes.append(f"月卡有效（至 {monthly_until}），免费放行")
            return b
        if rule == "monthly":
            b.notes.append("月卡已过期，按计时规则计费")
        rule = "time"
        b.rule = "time"

    if rule == "per_visit":
        b.free_min = FREE_MIN
        b.fee = 0.0 if total_min <= FREE_MIN else PER_VISIT_FEE
        if b.fee == 0:
            b.notes.append(f"停车 {total_min} 分钟，未超过 {FREE_MIN} 分钟免费时长")
        return b

    # 计时规则：先扣免费分钟，再按昼夜分段各取整到 30 分钟
    free_left = FREE_MIN
    segs = _split(entry, entry + timedelta(minutes=total_min))
    for kind, st, mi in segs:
        use_free = min(free_left, mi)
        free_left -= use_free
        bill_min = mi - use_free
        units = math.ceil(bill_min / UNIT_MIN) if bill_min > 0 else 0
        rate = RATE_DAY if kind == "day" else RATE_NIGHT
        fee = units * (UNIT_MIN / 60.0) * rate
        b.segments.append(Segment(kind, st, mi, bill_min, units, rate, fee))

    b.uncapped = sum(s.fee for s in b.segments)
    cycles = max(1, math.ceil(total_min / 1440)) if total_min > 0 else 1
    b.cap = CAP_PER_24H * cycles
    b.fee = min(b.uncapped, b.cap) if b.uncapped > 0 else 0.0
    if b.uncapped > b.cap:
        b.notes.append(f"超过封顶（每 24 小时 {CAP_PER_24H:.0f} 元），按封顶收取")
    if b.fee == 0:
        b.notes.append(f"停车 {total_min} 分钟，未超过 {FREE_MIN} 分钟免费时长")
    b.fee = round(b.fee + 1e-9, 2)
    return b


def rule_list():
    return [{"value": k, "label": v} for k, v in RULES.items()]
