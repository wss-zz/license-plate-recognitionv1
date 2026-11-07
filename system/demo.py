# -*- coding: utf-8 -*-
"""一键演示脚本：清库 → 车辆入场（真实识别）→ 时间快进 → 出场结算 → 打印账单。

用法：先启动 server.py，再 python demo.py [--keep]
用于录制演示视频与生成报告截图。
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8088"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(path, body=None, timeout=120):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def step(t, s=""):
    print(f"\n=== {t} ===")
    if s:
        print(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="保留数据库")
    ap.add_argument("--pause", type=float, default=1.6)
    args = ap.parse_args()
    if not args.keep:
        print("[demo] 清空数据库需要重启服务后生效（请先运行 python reset_demo.py）")

    step("1. 车辆驶入 1 号车道")
    r = call("/api/auto", {"lane": 1, "mode": "in"})
    print(f"识别车牌 {r.get('plate')}  置信度 {r.get('conf')}  耗时 {r.get('ms')} ms  来源 {r.get('source')}")
    print(f"闸机：{r['lane']['msg']}  闸杆={'抬起' if r['lane']['gate'] else '落下'}")
    time.sleep(args.pause)

    plate = r.get("plate")
    step("2. 车主扫码储值 50 元（模拟电子支付）")
    r = call("/api/recharge", {"plate": plate, "amount": 50})
    print(f"{plate} 余额 {r['balance']:.2f} 元")
    step("3. 时间快进 95 分钟（模拟车辆停放）")
    call("/api/backdate", {"plate": plate, "minutes": 95})
    time.sleep(args.pause)

    step("4. 出口识别与结算")
    r = call("/api/auto", {"lane": 1, "mode": "out"})
    b = r.get("bill") or {}
    if not b:
        print("结算失败：", r.get("msg"), "（该车无入场记录）")
        return 1
    print(f"识别 {r.get('plate')}  停车 {b.get('parked_min')} 分钟")
    for s in b.get("segments", []):
        cc = "白天" if s["kind"] == "day" else "夜间"
        print(f"  {cc} {s['from']}-{s['to']}  停放{s['minutes']}分  计费{s['bill_min']}分  "
              f"{s['units']}×0.5h×{s['rate']}元 = {s['fee']:.2f}元")
    print(f"  未封顶 {b.get('uncapped'):.2f}  封顶 {b.get('cap'):.2f}  →  实收 {b.get('fee'):.2f} 元")
    print(f"  余额 {r['vehicle']['balance']:.2f} 元")
    print(f"语音：{r['lane']['msg']}")
    time.sleep(args.pause)

    step("5. 第二辆车（月卡，直接登记车牌演示）")
    r = call("/api/enter", {"lane": 2, "plate": "皖A66666"})
    print(f"登记 {r['lane']['plate']}  闸机：{r['lane']['msg']}")
    call("/api/backdate", {"plate": "皖A66666", "minutes": 300})
    r = call("/api/exit", {"lane": 2, "plate": "皖A66666"})
    print(f"月卡命中={r['bill']['monthly_hit']}  实收 {r['bill']['fee']:.2f} 元  语音：{r['lane']['msg']}")
    time.sleep(args.pause)

    step("6. 第三辆车：余额不足 → 转人工")
    call("/api/enter", {"lane": 2, "plate": "皖A99001"})
    call("/api/backdate", {"plate": "皖A99001", "minutes": 180})
    r = call("/api/exit", {"lane": 2, "plate": "皖A99001"})
    print(f"应收 {r['bill']['fee']:.2f} 元  提示：{r.get('msg')}  语音：{r['lane']['msg']}")
    r = call("/api/manual_out", {"lane": 2, "plate": "皖A99001", "note": "余额不足，扫码补缴后人工放行"})
    print(f"人工放行后状态：{r['lane']['msg']}")
    time.sleep(args.pause)

    step("7. 异常演示：跟车告警")
    print(call("/api/alarm", {"lane": 1, "code": "TAILGATING"})["msg"])
    call("/api/clear_alarm", {"lane": 1})

    step("8. 今日看板")
    s = call("/api/stats")
    print(f"在场 {s['parked']} 辆 · 今日入场 {s['entries']} 车次 · 实收 {s['income']:.2f} 元 · "
          f"应收 {s['due']:.2f} 元 / 未收 {s['unpaid']:.2f} 元 · 异常 {s['abnormal']} 条")
    print(f"计费用例：{call('/api/billing_cases')['passed']}/{call('/api/billing_cases')['total']} 通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
