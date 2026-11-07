# 车牌识别一体闸机系统（Plate-Gate ANPR）

面向出入口场景的车牌识别与道闸业务系统：YOLO12n 检测 → CornerNet 四点矫正 → LPRNet 字符识别 → 双车道闸机业务（计费 / 语音 / 报表），支持 ONNX 部署与 INT8 量化。

## 关键结果

| 环节 | 指标 | 数值 |
| --- | --- | --- |
| 检测（289 幅人工标注） | P / R / F1 | 87.83% / 92.39% / 90.05% |
| 检测（CCPD 1000 幅域外） | F1 | 99.90% |
| 识别（五折交叉验证） | 整牌准确率 | 88.69%（绿牌 71.43%） |
| 识别（端到端，289 幅） | 整牌 / 字符 | 76.82% / 95.25% |
| 部署包 | 体积 / 单车道帧率 | 12.26 MB / 约 40 FPS（CPU） |
| 计费 | 用例通过 | 18 / 18 |

## 目录

```
system/            闸机业务系统（标准库实现，零第三方 Web 依赖）
  server.py        HTTP + JSON API + 静态页面
  billing.py       计费引擎（计时 / 按次 / 月卡）
  store.py         SQLite 持久化
  inference.py     检测 + 四点矫正 + 识别（可选 INT8 ONNX）
  tts.py           Windows SAPI 语音播报
  static/          单页前端
  tests/           计费与系统自测
code/              评测与实验脚本（检测/识别消融、传统方法基线、数据集统计等）
REPORT_MAIN.md     设计报告一～三章（Markdown 源）
REPORT_BODY.md     设计报告四、五章 + 参考文献
md2docx.py         Markdown → docx（按学校模板排版）
数据集与测试统计.xlsx
```

## 快速开始

```bash
# 1. 环境：Python 3.12，torch(cu128) + ultralytics + opencv
pip install torch ultralytics opencv-python numpy

# 2. 启动系统（首次启动约 8–9 秒加载模型）
cd system
python server.py --port 8088
# 浏览器打开 http://127.0.0.1:8088
```

模型路径可用环境变量覆盖：`GATE_DET`（检测权重）、`LPR_CKPT`（识别权重）、`GATE_REC_SPEC`（特殊车牌识别模型，默认关闭）。

自测：

```bash
python tests/test_billing.py     # 18 条计费用例
python tests/test_system.py      # 12 项系统端到端
```

## 复现评测

```bash
python code/eval_manual.py            # 检测 P/R/F1 + 识别整牌/字符准确率（289 幅）
python code/ablation_det.py           # 检测器横向对比（AP50/AP75/mIoU）
python code/ablation_rec.py           # 识别器横向对比（含 TTA）
python code/classic_det_baseline.py   # 颜色空间 / 边缘形态学 / 投影法 / Sobel 基线
```

数据集与训练权重体积较大（含车牌个人信息），不随仓库发布；脚本内路径按本机 `runs_2026-11-05/` 布局组织。

## 口径与限制

1. 1052 幅数据中 34 幅为第三方公开样例（补充黄牌类型），不计入自采 1018 幅；
2. 场景统计由 CLIP 零样本分类 + 人工抽查得到，精度可用但召回偏低，夜间/雨天数量应视为下限；
3. 88.69% 为五折交叉验证（裁剪图）口径，76.82% 为端到端（全景图 + 人工四点）口径，二者不可直接比较；
4. 字符集按 GA 36—2018 不含字母 I、O，故 0/O、1/I 混淆在本系统中不存在；
5. 检测器 ORT 静态 INT8 量化后分类分支输出恒为零，部署时检测走 FP32、识别走 INT8。

## 许可

MIT（见 LICENSE）。数据集与第三方模型遵循其原始许可。
