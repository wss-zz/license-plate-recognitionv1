# -*- coding: utf-8 -*-
"""重置演示数据：清空 SQLite、抓拍图，重新灌入 3 辆演示车。需先停止 server.py。"""
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
DB = BASE / "data" / "gate.db"
SNAPS = BASE / "data" / "snaps"

for p in (DB, DB.with_suffix(".db-wal"), DB.with_suffix(".db-shm")):
    if p.exists():
        p.unlink()
if SNAPS.exists():
    shutil.rmtree(SNAPS)
SNAPS.mkdir(parents=True, exist_ok=True)

import store  # noqa: E402  必须在删库之后导入
store.seed()
print("演示数据已重置：", DB)
print("演示车辆：皖A88888(余额200) 皖A66666(月卡+50) 皖A12345(余额500)")
