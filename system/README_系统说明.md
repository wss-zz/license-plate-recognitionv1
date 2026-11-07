# 车牌识别一体闸机系统 · 运行说明

## 1. 一键启动

双击 `启动.bat`（或在 PowerShell 中执行 `python server.py --port 8088`），浏览器打开 <http://127.0.0.1:8088>。

首次启动约需 **8–10 秒**（加载 PyTorch、YOLO、LPRNet、CornerNet 四件套并扫描图库），期间页面会显示重连提示，等待即可。

## 2. 运行环境

| 项 | 要求 |
|---|---|
| 操作系统 | Windows 10/11 x64 |
| Python | 3.12（与本机一致） |
| 必需第三方库 | `torch`、`torchvision`、`numpy`、`opencv-python`、`ultralytics` |
| 可选 | `pywin32`（语音播报，缺失时自动静默降级）、`onnxruntime`（ONNX 推理） |
| 不依赖 | Flask / FastAPI / Node.js / 数据库服务 —— Web 服务用标准库 `http.server`，数据用 SQLite 单文件 |

> 语音播报走 Windows 内置 SAPI（`win32com.client`），离线可用；加 `--no-tts` 启动则完全静音。

## 3. 目录结构

```
system/
├─ server.py           HTTP 服务 + 车道状态机 + REST 接口
├─ billing.py          计费引擎（计时/按次/月卡）
├─ store.py            SQLite 持久化（车辆/流水/账单/钱包/事件）
├─ inference.py        检测→四点矫正→识别 推理封装
├─ tts.py              Windows SAPI 语音播报
├─ static/index.html   单页前端（双车道 / 结算 / 看板 / 报表 / 记录）
├─ demo.py             演示脚本（8 步，可自动跑完一遍业务闭环）
├─ reset_demo.py       清空演示数据并写入示例车辆
├─ tests/
│  ├─ test_billing.py  计费 18 组用例
│  └─ test_system.py   系统 12 项端到端自测
└─ data/
   ├─ gate.db          SQLite 数据库（运行时生成）
   └─ snaps/           抓拍图（运行时生成）
```

## 4. 模型文件

| 用途 | 路径 |
|---|---|
| 车牌检测（YOLO12n 混合微调） | `../models/det/yolo12n_mix/weights/best.pt`（5.3 MB） |
| 四点矫正（CornerNet） | `D:\CCPD\corner\runs\cornernet\best.pt` |
| 车牌识别（LPRNet，70 字符集） | `..\..\runs_2026-11-02\models\lprnet_final8\best.pt` |
| ONNX 部署包 | `../deploy/lprnet_int8.onnx`（1.61 MB）、`../deploy/yolo12n_mix_fp32.onnx`（10.08 MB） |

路径可用环境变量覆盖：`GATE_DET`（检测权重）、`LPR_CKPT`（识别权重）。

## 5. 演示流程（录制视频用）

```powershell
python reset_demo.py          # 清库 + 写入示例车辆（需先停止服务）
python server.py --port 8088  # 启动
python demo.py                # 自动演示 8 个环节，逐步打印结算明细
```

`demo.py` 的八个环节：① 真实识别入场 → ② 储值 50 元 → ③ 回填 95 分钟 → ④ 出口结算（分段明细 7.50 元）→ ⑤ 月卡车辆免费出场 → ⑥ 余额不足转人工放行 → ⑦ 跟车告警 → ⑧ 看板与 18 条计费用例。

也可在页面上手动点：**模拟入场 / 模拟出场 / 跟车告警 / 未识别 / 复位**。

## 6. 自测

```powershell
python tests\test_billing.py   # 18/18 通过
python tests\test_system.py    # 12/12 通过（需服务已启动）
```

## 7. 常见问题

- **页面一直"等待车辆"**：确认 `data/snaps` 有写权限；首次启动请等 8 秒以上。
- **识别车牌总是"皖 A"**：说明四点矫正模块没加载，检查 `D:\CCPD\corner\runs\cornernet\best.pt` 是否存在；不矫正时模型会系统性偏向训练集主流省份。
- **无声音**：未装 `pywin32`，或启动时带了 `--no-tts`。
- **端口被占用**：`python server.py --port 8090` 换端口。
- **8 GB 显存机器**：不要与 PaddleOCR 推理同时跑（会 OOM），本系统默认只用 PyTorch 路径。
