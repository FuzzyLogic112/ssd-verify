#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSD Verify — 移动固态硬盘验收工具

一条命令查清一块新买的移动固态：是不是扩容盘、真实读写速度、SLC 缓存有多大、
缓存耗尽后掉到多少，最后按实测数据给出「留还是退」的判定和使用建议。

核心思路
--------
扩容盘（虚标容量）的伎俩是把超出真实容量的地址悄悄映射回盘的前段，写超了就
覆盖前面的数据。文件管理器里看着文件都在，打开全是损坏的。所以唯一可靠的
检测方式是**写满全盘再逐块读回校验** —— 每块写入唯一可辨认的内容，只要发生
覆盖，读回来时就对不上。

用法
----
    python verify_ssd.py                      # 交互式：列出磁盘让你选
    python verify_ssd.py --drive E: --mode full
    python verify_ssd.py --drive E: --clean   # 清理中断留下的测试文件
"""
from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("需要 numpy:  python -m pip install numpy")

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
except ImportError:
    sys.exit("需要 rich:  python -m pip install rich")

console = Console()

# ── 参数 ──────────────────────────────────────────────────────────────
CHUNK = 1 << 30          # 每个测试文件 1 GiB
HDR = 4096               # 块头，写唯一标识
SENTINELS = 16           # 块内散布的哨兵数，把覆盖检测粒度降到 64 MiB
SENT_LEN = 64
BASE_LEN = 16 << 20
FOLDER = "__ssd_verify__"
QUICK_GB = 20

SPARK = "▁▂▃▄▅▆▇█"


# ── 数据生成与校验（核心，已通过回绕/截断/覆盖用例验证）──────────────────
_BASE: bytes | None = None
_BUF: bytearray | None = None


def _base() -> bytes:
    """一次性生成的随机基块。

    若每块都现场生成 1 GiB 伪随机数据，只有约 40 MB/s，比被测硬盘还慢，
    测出来的就成了 CPU 性能。改为生成一次 16 MiB 随机基块后平铺复用。
    基块本身随机，每个 4K 扇区都不可压缩，主控想靠透明压缩虚报也没用。
    """
    global _BASE
    if _BASE is None:
        _BASE = np.random.default_rng(0xC0FFEE).bytes(BASE_LEN)
    return _BASE


def _sentinel_offsets(size: int) -> list[int]:
    body = size - HDR
    step = body // SENTINELS
    return [HDR + k * step for k in range(SENTINELS)]


def _buffer(size: int) -> bytearray:
    global _BUF
    if _BUF is not None and len(_BUF) == size:
        return _BUF
    base = _base()
    buf = bytearray(size)
    pos = 0
    while pos < size:
        n = min(BASE_LEN, size - pos)
        buf[pos:pos + n] = base[:n]
        pos += n
    _BUF = buf
    return buf


def make_chunk(idx: int, size: int) -> bytearray:
    """生成第 idx 块。块头 + 16 个散布哨兵都写入本块序号。

    复用同一缓冲区、只改写 17 处标记，使生成成本趋近于零；同时每块内容仍
    互不相同，既能抓地址回绕，也不会被「内容去重」型的假盘蒙混过关。
    返回共享缓冲区，调用方须立即写盘后再生成下一块。
    """
    buf = _buffer(size)
    head = f"SSD_VERIFY|idx={idx}|size={size}|sent={SENTINELS}|".encode()
    buf[0:HDR] = head.ljust(HDR, b"\0")
    for k, off in enumerate(_sentinel_offsets(size)):
        buf[off:off + SENT_LEN] = f"<SENT|idx={idx}|k={k}>".encode().ljust(SENT_LEN, b"=")
    return buf


def verify_chunk(data: bytes, idx: int, size: int) -> tuple[bool, str]:
    if len(data) < size:
        return False, f"文件被截断（{human(len(data))} < {human(size)}）"
    try:
        text = data[:HDR].rstrip(b"\0").decode()
    except UnicodeDecodeError:
        return False, "块头损坏（非法编码）"
    if f"idx={idx}|" not in text:
        got = text.split("idx=")[1].split("|")[0] if "idx=" in text else "?"
        return False, f"块头序号不符：期望 {idx}，实际 {got} —— 典型的扩容盘地址回绕"
    for k, off in enumerate(_sentinel_offsets(size)):
        want = f"<SENT|idx={idx}|k={k}>".encode().ljust(SENT_LEN, b"=")
        if data[off:off + SENT_LEN] != want:
            got = data[off:off + SENT_LEN].split(b">")[0][:40]
            return False, f"第 {k} 个哨兵（偏移 {human(off)}）不符，实际 {got!r} —— 数据被覆盖"
    return True, ""


# ── 工具 ──────────────────────────────────────────────────────────────
def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def mmss(sec: float) -> str:
    if sec < 0 or sec != sec:
        return "--:--"
    if sec >= 3600:
        return f"{int(sec // 3600)}小时{int(sec % 3600 // 60)}分"
    return f"{int(sec // 60)}分{int(sec % 60):02d}秒"


def sparkline(vals: list[float], width: int = 48) -> str:
    if not vals:
        return ""
    if len(vals) > width:                      # 抽稀到固定宽度
        step = len(vals) / width
        vals = [vals[min(int(i * step), len(vals) - 1)] for i in range(width)]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return SPARK[4] * len(vals)
    return "".join(SPARK[min(int((v - lo) / (hi - lo) * 7.999), 7)] for v in vals)


# ── 绕过系统缓存的读取 ────────────────────────────────────────────────
# Windows 会把刚写入的数据留在文件缓存里，直接 open() 读回来读的是内存副本，
# 测出来是 RAM 带宽（实测可达 1700+ MB/s）而不是硬盘速度。必须用
# FILE_FLAG_NO_BUFFERING 强制走物理设备。该标志要求读长度是扇区大小的整数倍、
# 且缓冲区按页对齐，所以用匿名 mmap 拿对齐内存。
_FILE_FLAG_NO_BUFFERING = 0x20000000
_FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 0x00000001
_OPEN_EXISTING = 3
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_IO_STEP = 8 << 20                   # 每次 ReadFile 的粒度，8 MiB

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                             ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                             ctypes.c_void_p]
_k32.CreateFileW.restype = ctypes.c_void_p
_k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                          ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
_k32.ReadFile.restype = ctypes.c_int
_k32.CloseHandle.argtypes = [ctypes.c_void_p]
_k32.CloseHandle.restype = ctypes.c_int


def read_direct(path: Path, size: int) -> bytes:
    """绕过 Windows 文件缓存读取整个文件，返回实际读到的字节。"""
    h = _k32.CreateFileW(str(path), _GENERIC_READ, _FILE_SHARE_READ, None,
                         _OPEN_EXISTING,
                         _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_SEQUENTIAL_SCAN, None)
    if not h or h == _INVALID_HANDLE:
        raise OSError(ctypes.get_last_error(), f"无法直读 {path}")
    mm = mmap.mmap(-1, size)                       # 匿名映射天然页对齐
    try:
        got = 0
        n_read = ctypes.c_uint32(0)
        while got < size:
            want = min(_IO_STEP, size - got)
            dst = (ctypes.c_char * want).from_buffer(mm, got)
            if not _k32.ReadFile(h, dst, want, ctypes.byref(n_read), None):
                del dst
                raise OSError(ctypes.get_last_error(), f"读取失败 @ {got}")
            del dst                                # 释放 mmap 上的导出视图
            if n_read.value == 0:
                break
            got += n_read.value
        return mm[:got]
    finally:
        mm.close()
        _k32.CloseHandle(h)


def ps(cmd: str) -> str:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                           capture_output=True, text=True, timeout=30,
                           encoding="utf-8", errors="replace")
        return (r.stdout or "").strip()
    except Exception:
        return ""


# ── 设备信息 ───────────────────────────────────────────────────────────
@dataclass
class Device:
    letter: str = ""
    label: str = ""
    filesystem: str = ""
    model: str = ""
    bus: str = ""
    nominal_bytes: int = 0
    fs_total: int = 0
    fs_free: int = 0
    health: str = ""
    serial: str = ""
    bridge: str = ""
    smart: dict = field(default_factory=dict)


def list_drives() -> list[dict]:
    out = ps(
        "Get-Volume | Where-Object {$_.DriveLetter} | ForEach-Object { "
        "$v=$_; $p=Get-Partition -DriveLetter $v.DriveLetter -ErrorAction SilentlyContinue; "
        "$d=if($p){Get-Disk -Number $p.DiskNumber -ErrorAction SilentlyContinue}; "
        "[PSCustomObject]@{letter=$v.DriveLetter; label=$v.FileSystemLabel; "
        "fs=$v.FileSystem; total=$v.Size; free=$v.SizeRemaining; "
        "model=$(if($d){$d.FriendlyName}); bus=$(if($d){$d.BusType}); "
        "health=$(if($d){$d.HealthStatus})} } | ConvertTo-Json -Compress")
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else [data]


def find_bridge(d: Device) -> str:
    """找出这块盘对应的 USB 桥接芯片 VID/PID。

    机器上通常挂着一堆 USB 设备（键鼠、摄像头、蓝牙），直接取第一个带 VID_ 的
    会张冠李戴 —— 报错的硬件信息比不报更糟。
    优先用磁盘序列号匹配：USB 存储设备的 InstanceId 里一般嵌着序列号，
    例如 盘序列号 ABCDEFA74788 ↔ USB\\VID_2109&PID_0715\\MSFT30ABCDEFA74788。
    匹配不上再退回「只看 UAS / 大容量存储类设备」。
    """
    raw = ps("Get-PnpDevice -PresentOnly -Class SCSIAdapter,USB -ErrorAction SilentlyContinue | "
             "Where-Object {$_.InstanceId -like 'USB*'} | "
             "ForEach-Object { $_.Class + '|' + $_.FriendlyName + '|' + $_.InstanceId }")
    lines = [l.strip() for l in raw.splitlines() if "VID_" in l]

    def vidpid(instance_id: str) -> str:
        parts = instance_id.split("\\")
        return parts[1] if len(parts) > 1 else instance_id

    serial = (d.serial or "").strip().rstrip(".")
    if serial:
        for ln in lines:
            iid = ln.split("|")[-1]
            if serial.upper().replace("_", "") in iid.upper().replace("_", ""):
                return vidpid(iid)

    for ln in lines:                       # 退路：只认存储类设备
        cls, name, iid = (ln.split("|") + ["", ""])[:3]
        if cls.strip().upper() == "SCSIADAPTER" or "UAS" in name.upper() \
                or "MASS STORAGE" in name.upper() or "大容量存储" in name:
            return vidpid(iid)
    return ""


def probe_device(letter: str) -> Device:
    d = Device(letter=letter.rstrip(":\\/"))
    for v in list_drives():
        if str(v.get("letter", "")).upper() == d.letter.upper():
            d.label = v.get("label") or ""
            d.filesystem = v.get("fs") or ""
            d.fs_total = int(v.get("total") or 0)
            d.fs_free = int(v.get("free") or 0)
            d.model = (v.get("model") or "").strip()
            d.bus = v.get("bus") or ""
            d.health = v.get("health") or ""
            break
    info = ps(
        f"$p=Get-Partition -DriveLetter {d.letter} -ErrorAction SilentlyContinue; "
        "if($p){$dk=Get-Disk -Number $p.DiskNumber; "
        "[PSCustomObject]@{size=$dk.Size; serial=$dk.SerialNumber} | ConvertTo-Json -Compress}")
    try:
        j = json.loads(info) if info else {}
        d.nominal_bytes = int(j.get("size") or 0)
        d.serial = (j.get("serial") or "").strip()
    except (json.JSONDecodeError, ValueError):
        pass

    d.bridge = find_bridge(d)

    rel = ps(f"$p=Get-Partition -DriveLetter {d.letter} -ErrorAction SilentlyContinue; "
             "if($p){$pd=Get-PhysicalDisk | Where-Object DeviceId -eq $p.DiskNumber; "
             "if($pd){$pd|Get-StorageReliabilityCounter -ErrorAction SilentlyContinue|"
             "Select-Object PowerOnHours,Temperature,ReadErrorsTotal,WriteErrorsTotal,Wear|"
             "ConvertTo-Json -Compress}}")
    try:
        if rel:
            d.smart = json.loads(rel)
    except json.JSONDecodeError:
        pass
    return d


BRIDGES = {
    "VID_2109&PID_0715": "VIA VL715 — USB 3.0 (5Gbps) 转 SATA，内部是 SATA 固态",
    "VID_2109&PID_0711": "VIA VL711 — USB 3.0 (5Gbps) 转 SATA",
    "VID_174C&PID_55AA": "ASMedia ASM1051/1053 — USB 3.0 (5Gbps) 转 SATA",
    "VID_174C&PID_1153": "ASMedia ASM1153 — USB 3.0 (5Gbps) 转 SATA",
    "VID_174C&PID_2362": "ASMedia ASM2362 — USB 3.2 Gen2 (10Gbps) 转 NVMe",
    "VID_174C&PID_2364": "ASMedia ASM2364 — USB 3.2 Gen2x2 (20Gbps) 转 NVMe",
    "VID_152D&PID_0583": "JMicron JMS583 — USB 3.2 Gen2 (10Gbps) 转 NVMe",
    "VID_152D&PID_1561": "JMicron JMS561 — USB 3.0 (5Gbps) 转 SATA",
    "VID_0BDA&PID_9210": "Realtek RTL9210 — USB 3.2 Gen2 (10Gbps) 转 NVMe",
}


def explain_bridge(bridge: str) -> str:
    up = (bridge or "").upper()
    for k, v in BRIDGES.items():
        if k.upper() in up:
            return v
    return ""


# ── 结果 ──────────────────────────────────────────────────────────────
@dataclass
class Result:
    device: dict = field(default_factory=dict)
    mode: str = "quick"
    started: str = ""
    written_bytes: int = 0
    planned_bytes: int = 0
    write_speeds: list = field(default_factory=list)
    read_speeds: list = field(default_factory=list)
    bad_blocks: list = field(default_factory=list)
    write_seconds: float = 0.0
    read_seconds: float = 0.0
    aborted_reason: str = ""

    def w_avg(self) -> float:
        return statistics.fmean(self.write_speeds) if self.write_speeds else 0.0

    def w_peak(self) -> float:
        return max(self.write_speeds) if self.write_speeds else 0.0

    def r_avg(self) -> float:
        return statistics.fmean(self.read_speeds) if self.read_speeds else 0.0

    def cache_knee(self) -> tuple[int | None, float, float]:
        """找 SLC 缓存拐点：首次跌破峰值 60% 且之后没恢复的位置。"""
        s = self.write_speeds
        if len(s) < 10:
            return None, 0.0, 0.0
        peak = max(s)
        for i in range(2, len(s)):
            if s[i] < peak * 0.6 and all(x < peak * 0.8 for x in s[i:i + 5]):
                before = statistics.fmean(s[max(0, i - 5):i])
                after = statistics.fmean(s[i:min(len(s), i + 20)])
                return i, before, after
        return None, 0.0, 0.0

    def sustained(self) -> float:
        """尾段 20% 的平均写入，代表缓存耗尽后的真实持续写入能力。"""
        s = self.write_speeds
        if not s:
            return 0.0
        tail = s[int(len(s) * 0.8):] or s[-1:]
        return statistics.fmean(tail)


# ── 实时界面 ───────────────────────────────────────────────────────────
class LiveView:
    def __init__(self, dev: Device, mode: str, total_chunks: int):
        self.dev, self.mode, self.total = dev, mode, total_chunks
        self.phase = "准备中"
        self.phase_no = 0
        self.done = 0
        self.speeds: list[float] = []
        self.bad = 0
        self.t0 = time.time()

    def header(self) -> Panel:
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bright_black", justify="right")
        t.add_column(style="bold")
        cap = human(self.dev.nominal_bytes) if self.dev.nominal_bytes else "?"
        t.add_row("设备", f"{self.dev.model or '未知'}　[cyan]{self.dev.letter}:[/]　{cap}")
        line = f"{self.dev.bus}　{self.dev.filesystem}"
        if self.dev.bridge:
            exp = explain_bridge(self.dev.bridge)
            line += f"　[bright_black]{exp or self.dev.bridge}[/]"
        t.add_row("链路", line)
        return Panel(t, title="[bold]SSD Verify 移动固态验收[/]", border_style="cyan")

    def body(self) -> Panel:
        pct = self.done / self.total if self.total else 0
        bar_w = 42
        filled = int(bar_w * pct)
        bar = "[green]" + "█" * filled + "[/][bright_black]" + "░" * (bar_w - filled) + "[/]"

        el = time.time() - self.t0
        eta = (el / self.done * (self.total - self.done)) if self.done else 0

        t = Table.grid(padding=(0, 2))
        t.add_column(style="bright_black", justify="right", width=8)
        t.add_column()
        t.add_row("阶段", f"[bold yellow]{self.phase_no}/3　{self.phase}[/]")
        t.add_row("进度", f"{bar}  [bold]{self.done}/{self.total}[/]  {pct*100:5.1f}%")
        t.add_row("已处理", f"{human(self.done * CHUNK)} / {human(self.total * CHUNK)}"
                            f"　[bright_black]已用 {mmss(el)}　剩余约 {mmss(eta)}[/]")
        if self.speeds:
            cur, avg, pk = self.speeds[-1], statistics.fmean(self.speeds), max(self.speeds)
            color = "green" if cur >= pk * 0.6 else "yellow" if cur >= pk * 0.3 else "red"
            t.add_row("速度", f"当前 [{color} bold]{cur:7.1f}[/] MB/s　"
                              f"平均 {avg:6.1f}　峰值 {pk:6.1f}")
            t.add_row("曲线", f"[cyan]{sparkline(self.speeds)}[/]")
        if self.phase_no == 3:
            style = "red bold" if self.bad else "green"
            t.add_row("坏块", f"[{style}]{self.bad}[/]")
        return Panel(t, border_style="bright_black")

    def render(self):
        return Group(self.header(), self.body())


# ── 主流程 ─────────────────────────────────────────────────────────────
def pick_drive() -> str | None:
    drives = list_drives()
    sysdrive = os.environ.get("SystemDrive", "C:").rstrip(":")
    rows = [d for d in drives if str(d.get("letter", "")).upper() != sysdrive.upper()]
    if not rows:
        console.print("[red]没有找到可测试的磁盘（系统盘已排除）[/]")
        return None

    t = Table(title="可测试的磁盘", border_style="cyan", title_style="bold")
    t.add_column("#", justify="right", style="bold cyan")
    t.add_column("盘符")
    t.add_column("卷标")
    t.add_column("型号")
    t.add_column("总线")
    t.add_column("容量", justify="right")
    t.add_column("可用", justify="right")
    t.add_column("格式")
    for i, d in enumerate(rows, 1):
        bus = d.get("bus") or ""
        t.add_row(str(i), f"{d.get('letter')}:", d.get("label") or "-",
                  (d.get("model") or "").strip() or "-",
                  f"[green]{bus}[/]" if bus == "USB" else bus,
                  human(d.get("total") or 0), human(d.get("free") or 0),
                  d.get("fs") or "-")
    console.print(t)
    console.print("[bright_black]提示：移动固态通常显示为 USB 总线[/]\n")

    choice = Prompt.ask("选择要测试的磁盘编号",
                        choices=[str(i) for i in range(1, len(rows) + 1)], default="1")
    return str(rows[int(choice) - 1]["letter"]) + ":"


def confirm(dev: Device, mode: str) -> bool:
    used = dev.fs_total - dev.fs_free
    w = Text()
    w.append("即将在 ", style="white")
    w.append(f"{dev.letter}: ", style="bold cyan")
    w.append(f"（{dev.model or '未知'}，{human(dev.fs_total)}）上执行测试\n\n", style="white")
    w.append("• 只在盘内新建 ", style="white")
    w.append(FOLDER, style="bold yellow")
    w.append(" 文件夹，", style="white")
    w.append("不会删除或修改你已有的文件\n", style="bold green")
    if mode == "full":
        w.append("• full 模式会", style="white")
        w.append("写满整块盘", style="bold yellow")
        w.append("（确诊扩容盘的唯一方式），测试期间可用空间会降到接近 0\n", style="white")
        est = dev.fs_free / (200 * 1024 * 1024)
        w.append(f"• 预计耗时 {mmss(est * 2)} 左右，期间请勿拔盘或让电脑休眠\n", style="white")
    else:
        w.append(f"• quick 模式只写 {QUICK_GB} GiB，", style="white")
        w.append("不足以确诊扩容盘", style="bold yellow")
        w.append("，仅评估速度与缓存\n", style="white")
    w.append("• 结束后自动删除全部测试文件，空间完整归还\n", style="white")
    if used > 1 << 30:
        w.append(f"\n注意：该盘已有 {human(used)} 数据，测试不会动它们，"
                 f"但可用空间不足会缩短测试范围", style="yellow")
    console.print(Panel(w, title="[bold yellow]确认[/]", border_style="yellow"))
    return Confirm.ask("开始测试吗", default=True)


def run_test(dev: Device, mode: str, limit_gb: int = 0) -> Result:
    root = Path(dev.letter + ":\\")
    work = root / FOLDER
    work.mkdir(exist_ok=True)

    free = shutil.disk_usage(root)[2]
    if limit_gb:
        budget = min(limit_gb << 30, int(free * 0.9))
    elif mode == "quick":
        budget = min(QUICK_GB << 30, int(free * 0.9))
    else:
        budget = int(free * 0.98)          # 留 2% 给文件系统元数据
    n = max(1, budget // CHUNK)

    res = Result(device=asdict(dev), mode=mode, planned_bytes=n * CHUNK,
                 started=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    view = LiveView(dev, mode, n)

    with Live(view.render(), console=console, refresh_per_second=4) as live:
        view.phase, view.phase_no = "写入测试", 2
        t_start = time.time()
        try:
            for i in range(n):
                data = make_chunk(i, CHUNK)          # 生成不计入耗时
                p = work / f"chunk_{i:05d}.bin"
                t0 = time.time()
                with open(p, "wb", buffering=0) as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())             # 绕过系统缓存，测真实落盘
                dt = max(time.time() - t0, 1e-6)
                res.write_speeds.append(CHUNK / dt / (1 << 20))
                res.written_bytes += CHUNK
                view.done = i + 1
                view.speeds = res.write_speeds
                live.update(view.render())
        except OSError as e:
            res.aborted_reason = str(e)
        res.write_seconds = time.time() - t_start

        if not res.write_speeds:
            return res

        done_n = len(res.write_speeds)
        view.phase, view.phase_no = "读回校验", 3
        view.total, view.done, view.speeds = done_n, 0, []
        view.t0 = time.time()
        t_start = time.time()
        for i in range(done_n):
            p = work / f"chunk_{i:05d}.bin"
            if not p.exists():
                res.bad_blocks.append({"idx": i, "why": "文件消失"})
                view.bad = len(res.bad_blocks)
                continue
            t0 = time.time()
            try:
                data = read_direct(p, CHUNK)       # 绕过系统缓存，测真实读盘速度
            except OSError:
                with open(p, "rb", buffering=0) as f:
                    data = f.read()                # 直读不可用时退回普通读
            dt = max(time.time() - t0, 1e-6)
            res.read_speeds.append(len(data) / dt / (1 << 20))
            ok, why = verify_chunk(data, i, CHUNK)
            if not ok:
                res.bad_blocks.append({"idx": i, "why": why})
            view.done = i + 1
            view.speeds = res.read_speeds
            view.bad = len(res.bad_blocks)
            live.update(view.render())
        res.read_seconds = time.time() - t_start

    console.print("\n[bright_black]清理测试文件…[/]", end="")
    shutil.rmtree(work, ignore_errors=True)
    console.print(" [green]完成，空间已归还[/]")
    return res


# ── 报告与建议 ─────────────────────────────────────────────────────────
def build_advice(res: Result, dev: Device) -> tuple[str, str, list[str]]:
    """返回 (判定, 一句话结论, 建议列表)。判定 ∈ PASS / PASS_SLOW / WARN / FAIL"""
    adv: list[str] = []
    sust, r = res.sustained(), res.r_avg()
    knee_i, before, after = res.cache_knee()

    if res.bad_blocks:
        return ("FAIL", f"校验失败 {len(res.bad_blocks)} 块 —— 数据写进去读不回来，立即退货。",
                ["这是扩容盘或坏盘的直接证据，保留本报告作为凭证。",
                 "不要再往这块盘存任何重要数据。",
                 "7 天无理由期内退货；超期可依据本报告主张「描述不符」。"])

    nominal = dev.nominal_bytes or dev.fs_total
    coverage = res.written_bytes / nominal if nominal else 0
    if res.mode != "full":
        verdict = "WARN"
        headline = f"{human(res.written_bytes)} 内无坏块，但未写满全盘，扩容盘尚未排除。"
        adv.append("跑 --mode full 写满全盘才能确诊扩容盘，这是唯一可靠的方式。")
    elif coverage < 0.9:
        verdict = "WARN"
        headline = f"只写入 {human(res.written_bytes)}，未覆盖标称容量。"
        adv.append("盘上原有数据占用了空间，清空后重测才能完整验证容量。")
    else:
        verdict = "PASS"
        headline = f"写满 {human(res.written_bytes)} 且逐块校验全部通过 —— 容量真实，不是扩容盘。"

    if r >= 800:
        adv.append(f"读取 {r:.0f} MB/s，属于 USB 3.2 Gen2 水准，可以直接当外接工作盘用。")
    elif r >= 400:
        adv.append(f"读取 {r:.0f} MB/s，适合游戏库外挂、素材调取；剪 4K 建议先拷回本机。")
    else:
        adv.append(f"读取 {r:.0f} MB/s，定位是仓库盘：归档、备份、游戏库都够用，"
                   f"不适合直接在盘上跑程序或剪辑。")

    if knee_i is not None:
        cache = knee_i * CHUNK
        drop = (1 - after / before) * 100 if before else 0
        adv.append(f"SLC 缓存约 {human(cache)}，之后从 {before:.0f} 掉到 {after:.0f} MB/s"
                   f"（跌 {drop:.0f}%）。单次拷贝控制在 {human(cache)} 以内可全程跑满速。")
        if sust < 100:
            adv.append(f"持续写入仅 {sust:.0f} MB/s，比机械硬盘还慢。首次灌入大批存量数据时"
                       f"建议分几批拷，中间留空闲让缓存回写释放。")
    else:
        adv.append(f"未观察到明显掉速，持续写入稳定在 {sust:.0f} MB/s 左右。")

    exp = explain_bridge(dev.bridge)
    if exp and "SATA" in exp:
        adv.append(f"桥接芯片为 {exp.split('—')[0].strip()}，内部是 SATA 固态。厂商标称速度"
                   f"通常取 SATA 接口理论值，经 USB 桥接后达不到属于普遍现象。")

    if verdict == "PASS" and (r < 400 or sust < 100):
        verdict = "PASS_SLOW"
        headline += " 但速度明显低于常见标称值。"
        adv.append("若当初是冲着标称速度买的，可凭本报告与卖家协商部分退款；"
                   "但同价位 USB 3.0 成品盘普遍如此，换货未必更好。")
    return verdict, headline, adv


VERDICT_STYLE = {"PASS": "green", "PASS_SLOW": "yellow", "WARN": "yellow", "FAIL": "red"}
VERDICT_LABEL = {"PASS": "通过　可以留下", "PASS_SLOW": "通过　但速度不及标称",
                 "WARN": "未完整验证", "FAIL": "不通过　建议退货"}


def show_report(res: Result, dev: Device) -> str:
    console.print()
    verdict, headline, adv = build_advice(res, dev)
    style = VERDICT_STYLE[verdict]

    t1 = Table(title="设备信息", border_style="cyan", title_style="bold", show_header=False)
    t1.add_column(style="bright_black", justify="right")
    t1.add_column()
    t1.add_row("型号", dev.model or "未知")
    t1.add_row("盘符 / 格式", f"{dev.letter}:　{dev.filesystem}")
    if dev.nominal_bytes:
        t1.add_row("标称容量", f"{human(dev.nominal_bytes)}　"
                              f"[bright_black]({dev.nominal_bytes/10**9:.0f} GB 十进制)[/]")
    t1.add_row("总线", dev.bus)
    if dev.bridge:
        t1.add_row("桥接芯片", explain_bridge(dev.bridge) or dev.bridge)
    t1.add_row("健康状态", f"[green]{dev.health}[/]" if dev.health == "Healthy" else dev.health)
    if dev.smart:
        for k, cn in (("PowerOnHours", "通电时间"), ("Temperature", "温度"),
                      ("ReadErrorsTotal", "读错误"), ("WriteErrorsTotal", "写错误")):
            if dev.smart.get(k) is not None:
                t1.add_row(cn, str(dev.smart[k]))
    else:
        t1.add_row("SMART", "[bright_black]USB 桥接未透传（常见，不代表有问题）[/]")

    knee_i, before, after = res.cache_knee()
    t2 = Table(title="实测结果", border_style="cyan", title_style="bold")
    t2.add_column("项目", style="bright_black")
    t2.add_column("实测值", justify="right", style="bold")
    t2.add_column("评价")

    bad_n = len(res.bad_blocks)
    t2.add_row("数据完整性", f"{len(res.write_speeds)-bad_n}/{len(res.write_speeds)} 块",
               "[green]全部通过[/]" if not bad_n else f"[red]{bad_n} 块失败[/]")
    t2.add_row("实际写入量", human(res.written_bytes),
               "[green]已写满全盘[/]" if res.mode == "full" and not res.aborted_reason
               else "[yellow]未写满[/]")
    t2.add_row("读取速度", f"{res.r_avg():.1f} MB/s",
               "[green]良好[/]" if res.r_avg() >= 400 else "[yellow]偏低[/]")
    t2.add_row("写入（缓存内）", f"{res.w_peak():.1f} MB/s",
               "[green]良好[/]" if res.w_peak() >= 400 else "[yellow]偏低[/]")
    t2.add_row("写入（持续）", f"{res.sustained():.1f} MB/s",
               "[green]良好[/]" if res.sustained() >= 150
               else "[yellow]偏低[/]" if res.sustained() >= 80 else "[red]很慢[/]")
    if knee_i is not None:
        t2.add_row("SLC 缓存", human(knee_i * CHUNK),
                   f"[bright_black]拐点后 {before:.0f} → {after:.0f} MB/s[/]")
    else:
        t2.add_row("SLC 缓存", "未见拐点", "[bright_black]全程速度稳定[/]")
    t2.add_row("耗时", f"写 {mmss(res.write_seconds)} / 读 {mmss(res.read_seconds)}", "")

    console.print(t1)
    console.print()
    console.print(t2)

    if res.write_speeds:
        console.print()
        console.print(Panel(
            f"[cyan]{sparkline(res.write_speeds, 64)}[/]\n"
            f"[bright_black]左=开始　右=写满　峰值 {res.w_peak():.0f} MB/s　"
            f"最低 {min(res.write_speeds):.0f} MB/s[/]",
            title="写入速度曲线", border_style="bright_black"))

    body = Text()
    body.append(headline + "\n\n", style=f"bold {style}")
    for i, a in enumerate(adv, 1):
        body.append(f"{i}. {a}\n", style="white")
    console.print()
    console.print(Panel(body, title=f"[bold {style}]判定：{VERDICT_LABEL[verdict]}[/]",
                        border_style=style))

    if res.bad_blocks:
        bt = Table(title="失败块明细（前 10）", border_style="red")
        bt.add_column("块号", justify="right")
        bt.add_column("原因")
        for b in res.bad_blocks[:10]:
            bt.add_row(str(b["idx"]), b["why"])
        console.print(bt)
    return verdict


def plot(res: Result, dev: Device, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=130)
    ax.plot(range(len(res.write_speeds)), res.write_speeds, lw=1.1,
            color="#2563eb", label="写入")
    if res.read_speeds:
        ax.plot(range(len(res.read_speeds)), res.read_speeds, lw=1.1,
                color="#16a34a", alpha=.75, label="读取")

    knee_i, before, after = res.cache_knee()
    if knee_i is not None:
        ax.axvline(knee_i, color="#dc2626", ls="--", lw=1.2)
        ax.annotate(f"SLC 缓存耗尽 @ {human(knee_i*CHUNK)}\n{before:.0f} → {after:.0f} MB/s",
                    xy=(knee_i, after),
                    xytext=(knee_i + len(res.write_speeds) * .06, before * .85),
                    color="#dc2626", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1))
    ax.set_xlabel("已写入 (GiB)")
    ax.set_ylabel("速度 (MB/s)")
    ax.set_title(f"{dev.model or '未知设备'}　{human(dev.nominal_bytes)}　读写速度曲线")
    ax.grid(alpha=.25, ls=":")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export(res: Result, dev: Device, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    verdict, headline, adv = build_advice(res, dev)
    payload = asdict(res)
    payload.update(verdict=verdict, headline=headline, advice=adv)
    jf = outdir / f"report-{stamp}.json"
    jf.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[bright_black]结果已导出：{jf}[/]")
    try:
        pf = outdir / f"speed-{stamp}.png"
        plot(res, dev, pf)
        console.print(f"[bright_black]速度曲线：{pf}[/]")
    except Exception as e:
        console.print(f"[bright_black]（跳过图表：{e}）[/]")


# ── 入口 ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="移动固态硬盘验收：扩容盘检测 + 速度 + SLC 缓存分析")
    ap.add_argument("--drive", help="盘符，例如 E:。不给则进入交互选择")
    ap.add_argument("--mode", choices=["quick", "full"], default=None,
                    help="quick=约20GiB速度摸底；full=写满全盘（确诊扩容盘）")
    ap.add_argument("--limit-gb", type=int, default=0, help="限制写入量(GiB)，用于自测")
    ap.add_argument("--out", default=None, help="报告输出目录，默认 ./reports")
    ap.add_argument("--clean", action="store_true", help="只清理残留的测试文件夹")
    ap.add_argument("--yes", action="store_true", help="跳过确认，用于脚本化调用")
    a = ap.parse_args()

    console.print()
    drive = a.drive
    if not drive:
        console.print(Panel("[bold]移动固态硬盘验收工具[/]\n"
                            "[bright_black]检测扩容盘 · 实测读写速度 · 分析 SLC 缓存[/]",
                            border_style="cyan"))
        drive = pick_drive()
        if not drive:
            return 1
    if len(drive.rstrip(":\\/")) == 1:
        drive = drive.rstrip(":\\/") + ":"

    root = Path(drive + "\\")
    if not root.exists():
        console.print(f"[red]找不到 {drive}，确认盘符正确且设备已插好[/]")
        return 1

    buf = ctypes.create_unicode_buffer(260)
    ctypes.windll.kernel32.GetSystemDirectoryW(buf, 260)
    if buf.value[:2].upper() == drive[:2].upper():
        console.print(f"[red]拒绝在系统盘 {drive} 上运行[/]")
        return 1

    if a.clean:
        shutil.rmtree(root / FOLDER, ignore_errors=True)
        console.print(f"[green]已清理 {drive}\\{FOLDER}[/]")
        return 0

    with console.status("[cyan]读取设备信息…[/]"):
        dev = probe_device(drive)

    mode = a.mode
    if mode is None:
        console.print()
        console.print(Panel(
            "[bold]quick[/]　约 20 GiB，2–5 分钟。测速度和缓存，[yellow]不能确诊扩容盘[/]\n"
            "[bold]full[/] 　写满全盘，1TB 约 2–3 小时。"
            "[green]唯一能确诊扩容盘的方式[/]，新买的盘建议选这个",
            title="测试模式", border_style="cyan"))
        mode = Prompt.ask("选择模式", choices=["quick", "full"], default="full")

    if not a.yes and not confirm(dev, mode):
        console.print("[bright_black]已取消[/]")
        return 0

    res = run_test(dev, mode, a.limit_gb)
    if not res.write_speeds:
        console.print("[red]一块都没写成功，盘可能有严重故障。[/]")
        if res.aborted_reason:
            console.print(f"[red]{res.aborted_reason}[/]")
        return 2

    verdict = show_report(res, dev)
    export(res, dev, Path(a.out) if a.out else Path(__file__).resolve().parent / "reports")
    return 2 if verdict == "FAIL" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]已中断。残留文件可用 --clean 清理：[/]")
        console.print("[bright_black]  python verify_ssd.py --drive X: --clean[/]")
        raise SystemExit(130)
