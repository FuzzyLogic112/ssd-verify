# -*- coding: utf-8 -*-
"""生成 README 插图。

数据来源是 2026-09-22 对一块海康威视 WIND 1TB 的真实全盘测试
（934 GiB 写满 + 逐块校验），不是构造的示例数据。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console                      # noqa: E402
from verify_ssd import (                              # noqa: E402
    CHUNK, Device, LiveView, Result, plot, show_report,
)

ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# ── 载入真实实测数据 ───────────────────────────────────────────────────
speeds = [float(x) for x in (ASSETS / "write_speeds.txt").read_text().split()]
print(f"载入真实写入速度 {len(speeds)} 点，峰值 {max(speeds):.1f}，最低 {min(speeds):.1f}")

DEV = Device(
    letter="E", label="HIKVISION", filesystem="exFAT",
    model="HIKVISION", bus="USB",
    nominal_bytes=1024 * 10**9, fs_total=1024 * 10**9, fs_free=1024 * 10**9,
    health="Healthy", serial="ABCDEFA74788", bridge="VID_2109&PID_0715",
)

RES = Result(
    mode="full", started="2026-09-22 12:07:43",
    written_bytes=934 * CHUNK, planned_bytes=934 * CHUNK,
    write_speeds=speeds,
    # 当时的版本只记录了读回的总体均值（319.2 MB/s），没有逐块速度。
    # 图里不画一条铺平的假曲线 —— 那看起来像逐块实测，会误导人。
    # 报告表格里的读取均值另行注入。
    read_speeds=[],
    bad_blocks=[], write_seconds=158.1 * 60, read_seconds=50.2 * 60,
)


class _RealReadAvg(Result):
    """只为出图：读取均值用实测的 319.2，但不伪造逐块曲线。"""

    def r_avg(self) -> float:
        return 319.2


RES.__class__ = _RealReadAvg

# ── 1. 速度曲线 PNG ────────────────────────────────────────────────────
plot(RES, DEV, ASSETS / "speed-curve.png")
print("已生成 assets/speed-curve.png")

# ── 2. 最终报告截图 SVG ────────────────────────────────────────────────
rec = Console(record=True, width=86, force_terminal=True)
import verify_ssd                                     # noqa: E402
orig = verify_ssd.console
verify_ssd.console = rec
try:
    show_report(RES, DEV)
finally:
    verify_ssd.console = orig
rec.save_svg(str(ASSETS / "report.svg"), title="ssd-verify — 最终报告")
print("已生成 assets/report.svg")

# ── 3. 实时进度截图 SVG ────────────────────────────────────────────────
rec2 = Console(record=True, width=86, force_terminal=True)
view = LiveView(DEV, "full", 934)
view.phase, view.phase_no = "写入测试", 2
view.done = 512
view.speeds = speeds[:512]
view.t0 = view.t0 - 2750          # 伪造已用时长，让 ETA 显示真实量级
rec2.print(view.render())
rec2.save_svg(str(ASSETS / "progress.svg"), title="ssd-verify — 实时进度")
print("已生成 assets/progress.svg")

print("\n全部插图生成完毕，均基于真实实测数据。")
