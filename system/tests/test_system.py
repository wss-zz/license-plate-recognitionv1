# -*- coding: utf-8 -*-
"""系统级自测：对运行中的服务发请求，验证进/出场、计费、异常、报表。

先启动 server.py，再运行：python tests/test_system.py
"""
import json
import sys
import urllib.request
import urllib.parse

BASE = "http://127.0.0.1:8088"
# 本机请求不走系统代理（环境里配置了 HTTP_PROXY 时会连不上 127.0.0.1）
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(path, body=None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    with _OPENER.open(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def show(name, ok, extra=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {extra}")
    return ok


def main():
    res = []
    # 1) 正常入场
    r = call("/api/enter", {"lane": 1, "plate": "皖A88888", "color": "blue", "conf": 0.99})
    res.append(show("入场开门", r["ok"] and r["lane"]["gate"] == 1, r["msg"]))
    # 2) 回拨 95 分钟后出场，应 7.5 元
    call("/api/backdate", {"plate": "皖A88888", "minutes": 95})
    r = call("/api/exit", {"lane": 1, "plate": "皖A88888"})
    res.append(show("计时收费 95 分钟", abs(r["bill"]["fee"] - 7.5) < 1e-6,
                    f"fee={r['bill']['fee']} 余额={r['vehicle']['balance']}"))
    # 3) 月卡车免费
    r = call("/api/enter", {"lane": 2, "plate": "皖A66666"})
    call("/api/backdate", {"plate": "皖A66666", "minutes": 300})
    r = call("/api/exit", {"lane": 2, "plate": "皖A66666"})
    res.append(show("月卡免费", r["bill"]["fee"] == 0 and r["bill"]["monthly_hit"],
                    f"fee={r['bill']['fee']}"))
    # 4) 余额不足
    call("/api/enter", {"lane": 1, "plate": "皖A00001"})
    call("/api/backdate", {"plate": "皖A00001", "minutes": 180})
    r = call("/api/exit", {"lane": 1, "plate": "皖A00001"})
    res.append(show("余额不足转人工", "余额不足" in (r["msg"] or ""), r["msg"]))
    # 5) 重复入场
    call("/api/enter", {"lane": 1, "plate": "皖A77777"})
    r = call("/api/enter", {"lane": 1, "plate": "皖A77777"})
    res.append(show("重复入场告警", bool(r["warn"]), r["warn"]))
    # 6) 无入场记录出场
    r = call("/api/exit", {"lane": 2, "plate": "皖A55555"})
    res.append(show("无入场记录异常", r["ok"] is False, r["msg"]))
    # 7) 未识别出场：人工放行
    call("/api/enter", {"lane": 2, "plate": "皖A33333"})
    r = call("/api/manual_out", {"lane": 2, "plate": "皖A33333"})
    res.append(show("未识别出场人工放行", r["ok"], r["msg"]))
    # 8) 跟车告警
    r = call("/api/alarm", {"lane": 1, "code": "TAILGATING"})
    res.append(show("跟车告警", r["ok"], r["msg"]))
    call("/api/clear_alarm", {"lane": 1})
    # 9) 计费用例
    c = call("/api/billing_cases")
    res.append(show("计费用例", c["passed"] == c["total"], f"{c['passed']}/{c['total']}"))
    # 10) 报表
    rep = call("/api/report?period=week")
    res.append(show("周报表", rep["total_income"] > 0, f"￥{rep['total_income']}"))
    # 11) 记录检索与分页
    rec = call("/api/records?page=1&size=5&q=" + urllib.parse.quote("皖A"))
    res.append(show("记录检索分页", rec["total"] >= 5 and len(rec["rows"]) == 5,
                    f"total={rec['total']}"))
    # 12) 查无此牌
    rec = call("/api/records?q=Zzz999")
    res.append(show("空结果", rec["total"] == 0, f"total={rec['total']}"))

    print(f"\n{sum(res)}/{len(res)} passed")
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(main())

