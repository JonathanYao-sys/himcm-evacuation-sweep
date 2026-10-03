# -*- coding: utf-8 -*-
"""2025 HiMCM A 题 · 基础场景建筑网络

按《基础场景建模假设.md》把单层办公楼建成无向加权网络：
  节点 V = 出口 / 走廊端点 / 走廊上房门口位置 / 房间
  边   E = 可通行路段
  边权   c_e = d_e / v_e  （通行时间，秒）
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "output"

# ---------------- 参数（全部取自 基础场景建模假设.md） ----------------
V_WALK   = 1.2    # m/s  行走速度（走廊与房门通道）
T0       = 20.0   # s    固定检查时间
ALPHA    = 0.8    # s/m² 面积搜索系数
BETA     = 8.0    # s/人 每人引导时间
AREA     = 20.0   # m²   每间办公室面积
OCC      = 4      # 人   每间预计人数
T_REPORT = 5.0    # s    通信报告时间
T_MARK   = 3.0    # s    清查标记时间

S_SEARCH = T0 + ALPHA * AREA + BETA * OCC          # 68 s 单间搜索与引导
S_ROOM   = S_SEARCH + T_REPORT + T_MARK            # 76 s 单间总处理时间

ROOMS  = ["R1", "R2", "R3", "R4", "R5", "R6"]
STARTS = {"A": "E_L", "B": "E_R"}   # 响应者 A 左出口进入，B 右出口进入

# 节点示意图坐标（m）。注意：边权只取决于 EDGES 中的长度，与绘图坐标无关。
POS = {
    "E_L": (-2.0, 0.0), "H_L": (0.0, 0.0),
    "N1": (5.0, 0.0), "N2": (15.0, 0.0), "N3": (25.0, 0.0),
    "H_R": (30.0, 0.0), "E_R": (32.0, 0.0),
    "R1": (5.0, 2.3), "R2": (15.0, 2.3), "R3": (25.0, 2.3),
    "R4": (5.0, -2.3), "R5": (15.0, -2.3), "R6": (25.0, -2.3),
}

# 边：(端点 u, 端点 v, 长度 m)
# 走廊 30 m；出口至走廊端点 2 m；房门至走廊 1 m；门口节点位于 x=5/15/25。
EDGES = [
    ("E_L", "H_L", 2.0),
    ("H_L", "N1", 5.0),
    ("N1", "N2", 10.0),
    ("N2", "N3", 10.0),
    ("N3", "H_R", 5.0),
    ("H_R", "E_R", 2.0),
    ("N1", "R1", 1.0), ("N2", "R2", 1.0), ("N3", "R3", 1.0),
    ("N1", "R4", 1.0), ("N2", "R5", 1.0), ("N3", "R6", 1.0),
]

# 边权查找表（秒）：WEIGHT[(u, v)] = d_e / v_e
WEIGHT = {}
for _u, _v, _d in EDGES:
    WEIGHT[(_u, _v)] = _d / V_WALK
    WEIGHT[(_v, _u)] = _d / V_WALK


def build_graph(v_walk=V_WALK, blocked=()):
    """返回邻接表 {node: [(neighbor, 通行时间 s), ...]}（无向图）。

    v_walk  — 行走速度（默认基准 1.2 m/s）
    blocked — 被阻断的边（如 [("N2", "N3")]），对应假设 x_e=0，从网络移除
    """
    blocked = {frozenset(e) for e in blocked}
    g = {n: [] for n in POS}
    for u, v, d in EDGES:
        if frozenset((u, v)) in blocked:
            continue
        c = d / v_walk
        g[u].append((v, c))
        g[v].append((u, c))
    return g
