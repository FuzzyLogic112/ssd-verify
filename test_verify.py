# -*- coding: utf-8 -*-
"""解析与校验层回归测试。改动核心逻辑后跑：  python test_verify.py

用例全部来自实际踩过的坑，不是编出来的。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_ssd import (           # noqa: E402
    CHUNK, CHUNK_MAX, CHUNK_MIN, HDR, SENTINELS, TARGET_SAMPLES, Device, Result,
    _sentinel_offsets, explain_bridge, human, make_chunk, mmss,
    build_advice, pick_chunk, sparkline, verify_chunk,
)

FAILS: list[str] = []
SZ = 1 << 24                        # 用 16MiB 跑测试，逻辑与 1GiB 完全一致


def check(name: str, got, exp) -> None:
    if got != exp:
        FAILS.append(f"{name}: 期望 {exp!r}, 得到 {got!r}")
        print(f"  BAD  {name}\n       期望 {exp!r}\n       得到 {got!r}")
    else:
        print(f"  OK   {name}")


def check_true(name: str, cond, detail="") -> None:
    if not cond:
        FAILS.append(f"{name}: {detail}")
        print(f"  BAD  {name}  {detail}")
    else:
        print(f"  OK   {name}")


print("== 块生成与唯一性 ==")
c0 = bytes(make_chunk(0, SZ))
c1 = bytes(make_chunk(1, SZ))
c7 = bytes(make_chunk(7, SZ))
check("块大小正确", len(c0), SZ)
check_true("不同块内容不同", c0 != c1 and c1 != c7)
check_true("块头含自身序号", b"idx=0|" in c0[:HDR])
check("哨兵数量", len(_sentinel_offsets(SZ)), SENTINELS)

print("\n== 扩容盘检测（核心）==")
ok, why = verify_chunk(c0, 0, SZ)
check("正确块通过", ok, True)

ok, why = verify_chunk(c0, 7, SZ)
check("序号错位被拒", ok, False)
check_true("错位原因提到回绕", "回绕" in why, why)

ok, why = verify_chunk(c1, 0, SZ)
check("回绕覆盖被拒", ok, False)

# 块头完好但中段被覆盖 —— 只看块头会漏，哨兵才能抓到
tampered = bytearray(c0)
mid = _sentinel_offsets(SZ)[9]
tampered[mid:mid + 20] = b"XXXXXXXXXXXXXXXXXXXX"
ok, why = verify_chunk(bytes(tampered), 0, SZ)
check("中段覆盖被拒", ok, False)
check_true("中段原因指向哨兵", "哨兵" in why, why)

ok, why = verify_chunk(c0[:5000], 0, SZ)
check("截断被拒", ok, False)

print("\n== 自适应块大小（U盘 / SD卡通用性）==")
GiB = 1 << 30
# 固定 1GiB 块会让小设备废掉：8GB 卡只有 7 个采样点，512MB 卡一块都写不进去。
# U 盘和 SD 卡恰恰是扩容盘最泛滥的地方，必须覆盖到。
for cap_gb, name in [(0.125, "128MB"), (0.5, "512MB"), (2, "2GB"), (8, "8GB"),
                     (32, "32GB"), (128, "128GB"), (512, "512GB"),
                     (1024, "1TB"), (4096, "4TB")]:
    free = int(cap_gb * GiB)
    ck = pick_chunk(free)
    samples = free // ck
    check_true(f"{name}: {samples} 个采样点 (块 {human(ck)})",
               samples >= 25, f"只有 {samples} 个采样点")
    check_true(f"{name}: 块大小在界内", CHUNK_MIN <= ck <= CHUNK_MAX, human(ck))
    check_true(f"{name}: 2的幂且扇区对齐",
               ck % 4096 == 0 and (ck & (ck - 1)) == 0, human(ck))

check("大盘用满 1GiB 块", pick_chunk(2000 * GiB), CHUNK_MAX)
check("极小设备不低于下限", pick_chunk(1 << 20), CHUNK_MIN)
check("零容量不崩", pick_chunk(0), CHUNK_MIN)

# 小块也必须能正常生成和校验
small = CHUNK_MIN
c_a = bytes(make_chunk(3, small))
c_b = bytes(make_chunk(4, small))
check("小块大小正确", len(c_a), small)
check("小块正确校验", verify_chunk(c_a, 3, small)[0], True)
check("小块回绕被拒", verify_chunk(c_b, 3, small)[0], False)
check("小块哨兵数不变", len(_sentinel_offsets(small)), SENTINELS)
check_true("小块哨兵不越界", _sentinel_offsets(small)[-1] + 64 <= small)

print("\n== 格式化工具 ==")
check("human KB", human(2048), "2.0KB")
check("human GB", human(1 << 30), "1.0GB")
check("mmss 秒", mmss(75), "1分15秒")
check("mmss 小时", mmss(7500), "2小时5分")
check("sparkline 长度", len(sparkline([1, 2, 3, 4], 4)), 4)
check("sparkline 空输入", sparkline([]), "")
check_true("sparkline 等值不崩", len(sparkline([5, 5, 5])) == 3)

print("\n== 桥接芯片识别 ==")
check_true("VL715 识别", "VL715" in explain_bridge("VID_2109&PID_0715\\MSFT30ABC"))
check_true("VL715 标明 SATA", "SATA" in explain_bridge("VID_2109&PID_0715"))
check_true("ASM2362 标明 NVMe", "NVMe" in explain_bridge("VID_174C&PID_2362"))
check("未知芯片返回空", explain_bridge("VID_FFFF&PID_0000"), "")
check("空输入不崩", explain_bridge(""), "")

print("\n== SLC 缓存拐点识别 ==")
r = Result()
r.write_speeds = [290.0] * 200 + [85.0] * 700      # 模拟 200GiB 后掉速
knee, before, after = r.cache_knee()
check("拐点位置", knee, 200)
check_true("拐点前速度", 280 < before < 300, str(before))
check_true("拐点后速度", 80 < after < 90, str(after))
check_true("持续写入取尾段", 80 < r.sustained() < 90, str(r.sustained()))

check("缓存容量按块大小换算", r.cache_bytes(), 200 * CHUNK)
r_small = Result(chunk_bytes=CHUNK_MIN)
r_small.write_speeds = [290.0] * 200 + [85.0] * 700
check("小块下缓存容量同样正确", r_small.cache_bytes(), 200 * CHUNK_MIN)

r2 = Result()
r2.write_speeds = [500.0] * 100                    # 全程无掉速
check("无拐点返回 None", r2.cache_knee()[0], None)

r3 = Result()
check("空数据不崩", r3.cache_knee()[0], None)
check("空数据 sustained", r3.sustained(), 0.0)

print("\n== 判定逻辑 ==")
dev = Device(letter="E", model="HIKVISION", bus="USB",
             nominal_bytes=1024 * 10**9, fs_total=1024 * 10**9,
             bridge="VID_2109&PID_0715")

# 有坏块 -> 必须 FAIL
rb = Result(mode="full", written_bytes=1024 * 10**9)
rb.write_speeds = [290.0] * 900
rb.read_speeds = [320.0] * 900
rb.bad_blocks = [{"idx": 500, "why": "块头序号不符"}]
v, head, adv = build_advice(rb, dev)
check("有坏块判 FAIL", v, "FAIL")
check_true("FAIL 建议退货", any("退货" in a for a in adv) or "退货" in head)

# 全盘通过但速度慢 -> PASS_SLOW（本次海康盘的真实情形）
rs = Result(mode="full", written_bytes=1024 * 10**9)
rs.write_speeds = [290.0] * 200 + [85.0] * 700
rs.read_speeds = [319.0] * 900
v, head, adv = build_advice(rs, dev)
check("慢盘判 PASS_SLOW", v, "PASS_SLOW")
check_true("提到容量真实", "不是扩容盘" in head, head)
check_true("提示分批拷贝", any("分几批" in a for a in adv))
check_true("解释 SATA 桥接", any("SATA" in a for a in adv))

# quick 模式 -> WARN，且必须说明没排除扩容盘
rq = Result(mode="quick", written_bytes=20 << 30)
rq.write_speeds = [290.0] * 20
rq.read_speeds = [320.0] * 20
v, head, adv = build_advice(rq, dev)
check("quick 判 WARN", v, "WARN")
check_true("WARN 提示要跑 full", any("full" in a for a in adv))

# 全盘通过且速度好 -> PASS
rf = Result(mode="full", written_bytes=1024 * 10**9)
rf.write_speeds = [900.0] * 900
rf.read_speeds = [1000.0] * 900
dev_fast = Device(letter="F", model="Fast", bus="USB",
                  nominal_bytes=1024 * 10**9, fs_total=1024 * 10**9,
                  bridge="VID_174C&PID_2362")
v, head, adv = build_advice(rf, dev_fast)
check("快盘判 PASS", v, "PASS")
check_true("建议可当工作盘", any("工作盘" in a for a in adv))

print()
if FAILS:
    print(f"失败 {len(FAILS)} 项：")
    for f in FAILS:
        print("  -", f)
    raise SystemExit(1)
print("全部通过")
