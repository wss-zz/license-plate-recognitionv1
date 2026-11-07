# -*- coding: utf-8 -*-
"""计费用例：15 组，覆盖跨免费、跨昼夜、跨午夜、24h/48h 封顶、超长、月卡、未识别出场。

运行：python tests/test_billing.py
"""
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from billing import calc_fee, RULES  # noqa: E402

D = datetime
CASES = [
    # 名称, 车牌, 规则, 入场, 出场, 月卡到期, 期望金额, 说明
    ("免费内-5分钟", "皖A11111", "time", D(2026, 11, 3, 9, 0), D(2026, 11, 3, 9, 5), None, 0.0, "≤15 分钟免费"),
    ("免费边界-15分钟", "皖A11112", "time", D(2026, 11, 3, 9, 0), D(2026, 11, 3, 9, 15), None, 0.0, "恰好 15 分钟仍免费"),
    ("超出免费-20分钟", "皖A11113", "time", D(2026, 11, 3, 9, 0), D(2026, 11, 3, 9, 20), None, 2.5, "计费 5 分钟按 30 分钟单位 → 0.5h×5"),
    ("白天-2小时", "皖A11114", "time", D(2026, 11, 3, 9, 0), D(2026, 11, 3, 11, 0), None, 10.0, "扣 15 分钟后 105 分钟 → 4 单位 = 2h"),
    ("跨昼夜-19进21出", "皖A11115", "time", D(2026, 11, 3, 19, 0), D(2026, 11, 3, 21, 0), None, 8.0, "白天 45 分钟(2 单位=5 元)+夜间 1h(3 元)"),
    ("跨午夜-23进1出", "皖A11116", "time", D(2026, 11, 3, 23, 0), D(2026, 11, 4, 1, 0), None, 6.0, "全为夜间，105 分钟→4 单位=2h×3"),
    ("满24小时封顶", "皖A11117", "time", D(2026, 11, 3, 8, 0), D(2026, 11, 4, 8, 0), None, 60.0, "未封顶 96 元 → 封顶 60"),
    ("48小时封顶", "皖A11118", "time", D(2026, 11, 3, 8, 0), D(2026, 11, 5, 8, 0), None, 120.0, "2 个 24h 周期 → 封顶 120"),
    ("超长30小时", "皖A11119", "time", D(2026, 11, 3, 10, 0), D(2026, 11, 4, 16, 0), None, 120.0, "未封顶 126 → 2 周期封顶 120"),
    ("按次-20分钟", "皖A11120", "per_visit", D(2026, 11, 3, 9, 0), D(2026, 11, 3, 9, 20), None, 10.0, "按次 10 元"),
    ("按次-10分钟", "皖A11121", "per_visit", D(2026, 11, 3, 9, 0), D(2026, 11, 3, 9, 10), None, 0.0, "免费时长内"),
    ("月卡有效", "皖A11122", "monthly", D(2026, 11, 3, 9, 0), D(2026, 11, 3, 18, 0), date(2026, 12, 31), 0.0, "月卡期内免费"),
    ("月卡过期", "皖A11123", "monthly", D(2026, 11, 3, 9, 0), D(2026, 11, 3, 11, 0), date(2026, 10, 31), 10.0, "过期回落计时规则"),
    ("未识别出场", "皖A11124", "time", D(2026, 11, 3, 9, 0), None, None, None, "abnormal，需人工补录"),
    ("45分钟-取整", "皖A11125", "time", D(2026, 11, 3, 9, 0), D(2026, 11, 3, 9, 45), None, 2.5, "30 分钟计费 → 1 单位"),
    ("跨整点-3小时", "皖A11126", "time", D(2026, 11, 3, 13, 1), D(2026, 11, 3, 16, 1), None, 15.0, "同属白天应整体取整（165 分钟→6 单位=3h）"),
    ("跨整点-95分钟", "皖A11127", "time", D(2026, 11, 3, 14, 26), D(2026, 11, 3, 16, 1), None, 7.5, "跨整点但同为白天 → 80 分钟→3 单位"),
    ("跨整点-跨昼夜", "皖A11128", "time", D(2026, 11, 3, 19, 55), D(2026, 11, 3, 20, 30), None, 1.5, "免费 15 分钟按时间顺序先扣（白天 5 分钟）→ 夜间剩 20 分钟 1 单位×3"),
]


def main():
    ok = 0
    rows = []
    for name, plate, rule, t_in, t_out, m_until, want, why in CASES:
        b = calc_fee(plate, t_in, t_out, rule=rule, monthly_until=m_until,
                     now=t_out or t_in)
        got = None if b.abnormal and b.exit is None else b.fee
        if want is None:
            good = bool(b.abnormal)
        else:
            good = got is not None and abs(got - want) < 1e-6 and not b.abnormal
        ok += good
        rows.append((name, "PASS" if good else "FAIL", got, want, why, b.notes))
    w = max(len(r[0]) for r in rows)
    for name, st, got, want, why, notes in rows:
        print(f"{name:<{w}}  {st}  got={got}  want={want}   {why}")
    print(f"\n{ok}/{len(rows)} passed")
    assert ok == len(rows), "计费用例未全部通过"
    print("规则：")
    for k, v in RULES.items():
        print("  -", k, ":", v)


if __name__ == "__main__":
    main()
