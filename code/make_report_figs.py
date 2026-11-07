# -*- coding: utf-8 -*-
"""生成设计报告用的示意图（架构图 / 业务流程图 / 识别流水线 / 电气连接图）。
输出: runs_2026-11-05/figs/report/*.png
"""
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05\figs\report")
OUT.mkdir(parents=True, exist_ok=True)

C_BG = "#ffffff"
C_L1 = "#e8f0fe"
C_L2 = "#e6f4ea"
C_L3 = "#fef7e0"
C_L4 = "#fce8e6"
C_LINE = "#5f6368"


def box(ax, x, y, w, h, text, fc, fs=11, bold=False, ec=C_LINE, lw=1.1, r=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.5)
    return p


def arrow(ax, p1, p2, style="-|>", color=C_LINE, lw=1.4, ls="-", rad=0.0, text=None, dy=0.02, fs=9):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13, linewidth=lw,
                        color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + dy, text, ha="center", va="bottom", fontsize=fs, color="#3c4043")


def new_ax(w=11, h=6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10 * h / w)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    return fig, ax


# ---------------------------------------------------------------- 1. 总体架构
def fig_arch():
    fig, ax = new_ax(11, 6.4)
    H = 10 * 6.4 / 11
    layers = [
        (H - 1.15, 1.10, "展 示 层", C_L4, ["Web 前端（车道视图 / 结算 / 看板 / 报表 / 记录检索）"]),
        (H - 2.55, 1.25, "业 务 层", C_L3, ["进场受理", "出场计费", "闸杆控制", "异常处置", "账单与报表"]),
        (H - 3.95, 1.25, "算 法 层", C_L2, ["目标检测\nYOLO12n", "四点矫正\nCornerNet", "车牌识别\nLPRNet", "辅助识别\nPP-OCRv5"]),
        (H - 5.45, 1.35, "感 知 层", C_L1, ["图像采集", "补光/曝光", "车辆触发", "语音播报", "显示屏"]),
    ]
    for y, h, name, color, items in layers:
        box(ax, 0.35, y, 1.55, h, name, color, fs=13, bold=True)
        n = len(items)
        gap = 0.12
        w = (9.3 - 0.35 - 1.75 - gap * (n - 1)) / n
        for i, it in enumerate(items):
            x = 1.95 + i * (w + gap)
            box(ax, x, y, w, h, it, "#ffffff", fs=10.5)
    # 左侧数据流标识
    top = layers[0][0] + layers[0][1]
    bot = layers[-1][0]
    arrow(ax, (0.12, bot), (0.12, top), style="<|-|>", lw=1.8)
    ax.text(0.05, (top + bot) / 2, "数据上行 · 指令下行", rotation=90, fontsize=9,
            color="#5f6368", va="center", ha="right")
    ax.text(0.35, H - 6.15, "数据与权重：自采 1052 幅标注数据 → YOLO12n 检测 + CornerNet 四点 + LPRNet/PP-OCRv5 识别；"
                            "模型导出 ONNX 后由部署包加载", fontsize=9.5, color="#3c4043")
    fig.tight_layout()
    fig.savefig(OUT / "arch.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)


# ---------------------------------------------------------------- 2. 业务流
def fig_flow():
    fig, ax = new_ax(11, 8.0)
    H = 10 * 8.0 / 11
    x0, w, h = 1.9, 4.6, 0.6
    y1 = H - 0.75
    top = ["车辆到达 / 触发地感", "抓拍连拍（3 帧）", "检测 → 四点矫正 → 识别"]
    for i, text in enumerate(top):
        box(ax, x0, y1 - i * 0.82, w, h, text, C_L1, fs=10.5)
        if i:
            arrow(ax, (x0 + w / 2, y1 - (i - 1) * 0.82), (x0 + w / 2, y1 - i * 0.82 + h))
    cy = y1 - 3.05
    dia = plt.Polygon([[x0 + w / 2, cy + 0.45], [x0 + w / 2 + 1.3, cy],
                       [x0 + w / 2, cy - 0.45], [x0 + w / 2 - 1.3, cy]],
                      closed=True, facecolor=C_L3, edgecolor=C_LINE, lw=1.1, zorder=2)
    ax.add_patch(dia)
    ax.text(x0 + w / 2, cy, "识别成功？", ha="center", va="center", fontsize=10.5, zorder=3)
    arrow(ax, (x0 + w / 2, y1 - 1.64), (x0 + w / 2, cy + 0.45))

    yes = ["显示车牌 + 语音播报", "入场建流水 / 出场按时长计费",
           "余额充足 → 扣款并生成账单", "抬杆 → 车辆通过 → 落杆"]
    for i, text in enumerate(yes):
        y = cy - 0.45 - 0.32 - i * 0.82 - h
        box(ax, x0, y, w, h, text, C_L2, fs=10.5)
        if i:
            arrow(ax, (x0 + w / 2, y + h + 0.82), (x0 + w / 2, y + h))
    arrow(ax, (x0 + w / 2, cy - 0.45), (x0 + w / 2, cy - 0.77), color="#188038")
    ax.text(x0 + w / 2 + 0.12, cy - 0.66, "是", fontsize=10, color="#188038")

    bx, bw = 7.15, 2.55
    no = ["转人工复核\n（无牌 / 识别失败）", "人工录入或直接放行", "未识别出场 → 异常流水", "管理员补录 → 结算"]
    ny0 = cy + 0.15
    for i, text in enumerate(no):
        y = ny0 - i * 0.92 - 0.72
        box(ax, bx, y, bw, 0.72, text, C_L4, fs=9.5)
        if i:
            arrow(ax, (bx + bw / 2, y + 0.72 + 0.92), (bx + bw / 2, y + 0.72), ls="--", color="#c5221f")
    arrow(ax, (x0 + w / 2 + 1.3, cy), (bx, ny0 - 0.36), color="#c5221f")
    ax.text((x0 + w / 2 + 1.3 + bx) / 2, cy + 0.22, "否", fontsize=10, color="#c5221f")
    arrow(ax, (bx, ny0 - 3 * 0.92 - 0.36), (x0 + w + 0.06, cy - 0.45 - 0.32 - 2 * 0.82 - h / 2),
          ls="--", color="#c5221f", rad=-0.18)
    ax.text(bx + 0.08, ny0 - 3 * 0.92 - 0.62, "补录后回到计费", fontsize=8.5, color="#c5221f")
    fig.tight_layout()
    fig.savefig(OUT / "flow_business.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)


# ---------------------------------------------------------------- 3. 识别流水线
def fig_pipeline():
    fig, ax = new_ax(11, 5.6)
    H = 10 * 5.6 / 11
    ax.text(0.25, H - 0.35, "车牌识别推理流水线", fontsize=12, fontweight="bold")
    y1, h = H - 1.55, 0.78
    names = ["输入图像\n（任意分辨率）", "YOLO12n 检测\n输出车牌框", "CornerNet\n四点回归", "透视变换\n192×64 矫正图"]
    colors = [C_L1, C_L2, C_L3, C_L4]
    xs = [0.25, 2.6, 4.95, 7.3]
    for x, nm, c in zip(xs, names, colors):
        box(ax, x, y1, 2.15, h, nm, c, fs=9.5)
    for i in range(3):
        arrow(ax, (xs[i] + 2.15, y1 + h / 2), (xs[i + 1], y1 + h / 2))

    y2, h2 = y1 - 1.45, 0.72
    box(ax, 1.3, y2, 3.2, h2, "LPRNet 识别（主）\n蓝/黄 7 位 · 绿 8 位", "#ffffff", fs=9.5)
    box(ax, 5.1, y2, 3.2, h2, "PP-OCRv5 识别（辅）", "#ffffff", fs=9.5)
    # 母线式分流：竖段 → 横段 → 分两路下探
    ax.plot([8.37, 8.37], [y1, y2 + h2 + 0.30], color=C_LINE, lw=1.3, zorder=1)
    ax.plot([2.9, 8.37], [y2 + h2 + 0.30, y2 + h2 + 0.30], color=C_LINE, lw=1.3, zorder=1)
    arrow(ax, (2.9, y2 + h2 + 0.30), (2.9, y2 + h2))
    arrow(ax, (6.7, y2 + h2 + 0.30), (6.7, y2 + h2))

    y3, h3 = y2 - 1.3, 0.72
    box(ax, 3.4, y3, 3.6, h3, "多视角 / 双模型投票 → 车牌号 + 置信度", "#d2e3fc", fs=9.5)
    arrow(ax, (2.9, y2), (4.4, y3 + h3), rad=-0.12)
    arrow(ax, (6.7, y2), (6.0, y3 + h3), rad=0.12)
    ax.text(0.25, 0.28, "注：四点矫正消除倾斜与透视畸变（倾斜场景整牌准确率提升 24.7 个百分点）；"
                        "双识别器加权投票用于抑制单模型偶发错误。", fontsize=8.5, color="#3c4043")
    fig.tight_layout()
    fig.savefig(OUT / "flow_recognize.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)


# ---------------------------------------------------------------- 4. 电气连接
def fig_wiring():
    fig, ax = new_ax(11, 6.2)
    H = 10 * 6.2 / 11
    box(ax, 3.55, 2.35, 2.9, 1.15, "工控机 / 服务器\n(USB3.0 · 千兆网口)", C_L2, fs=10, bold=True)
    devs = {
        "cam":  (0.45, H - 1.55, 3.0, 0.95, "USB 摄像头（进/出口）", "USB 3.0"),
        "led":  (7.0, H - 1.55, 3.0, 0.95, "LED 显示屏（车牌/金额）", "HDMI / 串口"),
        "loop": (0.45, 2.35, 3.0, 0.95, "地感线圈 / 红外", "DI 输入"),
        "spk":  (7.0, 2.35, 3.0, 0.95, "语音播报（功放+音箱）", "音频输出"),
        "gate": (3.55, 0.45, 2.9, 0.95, "道闸控制器（抬/落杆）", "DO 继电器"),
        "web":  (7.0, 0.45, 3.0, 0.95, "计费与后台终端", "TCP/IP"),
        "pwr":  (0.45, 0.45, 3.0, 0.95, "AC220V→DC12V/5V 电源", "供电"),
    }
    for x, y, w, h, t, _ in devs.values():
        box(ax, x, y, w, h, t, "#ffffff", fs=9)

    px1, py1, px2, py2 = 3.55, 2.35, 6.45, 3.5
    pairs = [
        ((2.55, H - 1.55), (px1 + 0.6, py2), "USB 3.0"),
        ((8.0, H - 1.55), (px2 - 0.6, py2), "显示信号"),
        ((3.45, 2.83), (px1, 2.83), "DI 信号"),
        ((px2, 2.83), (7.0, 2.83), "音频"),
        ((6.15, 1.40), (6.15, 2.35), "DO 继电器"),
        ((7.0, 0.93), (px2, 2.35), "TCP/IP"),
    ]
    for p1, p2, lab in pairs:
        arrow(ax, p1, p2, style="<|-|>", lw=1.2)
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.1, lab, fontsize=8, color="#5f6368", ha="center")
    arrow(ax, (3.45, 0.93), (4.3, 2.35), ls="--", lw=1.1)
    ax.text(3.05, 1.6, "DC 供电", fontsize=8, color="#5f6368")
    ax.text(0.45, 0.18, "说明：演示环境以「继电器板 + 软件模拟」替代真实道闸与地感线圈，接口协议与现场一致，可直接替换。",
            fontsize=8.5, color="#3c4043")
    fig.tight_layout()
    fig.savefig(OUT / "wiring.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)


if __name__ == "__main__":
    fig_arch()
    fig_flow()
    fig_pipeline()
    fig_wiring()
    print("figures ->", OUT)
