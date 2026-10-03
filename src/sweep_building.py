# -*- coding: utf-8 -*-
"""2025 HiMCM A 题 · 基础场景建筑网络（由 Config 自动生成）。

按《基础场景建模假设.md》把单层办公楼建成无向加权网络：
  节点 V = 出口 / 走廊端点 / 走廊上的房门口位置 / 房间
  边   E = 可通行路段（x_e = 0 的边被删除）
  边权   c_e = d_e / v_e  （通行时间，秒）

**所有几何量都由 `Config` 推导**（走廊长度、出口段、门到走廊长度、
每侧房间数），所以改参数后重跑即可，无需再手工改节点/边/绘图坐标。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from sweep_config import Config

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "output"

RESPONDER_NAMES = "ABCDEFGHIJKLMNOP"


@dataclass
class Scene:
    """一个具体参数下的建筑网络与派生量。"""
    cfg: Config
    pos: Dict[str, Tuple[float, float]]          # 节点绘图坐标 (m)
    edges: List[Tuple[str, str, float]]          # (u, v, 长度 m)
    weight: Dict[Tuple[str, str], float]         # (u, v) -> 通行时间 s
    graph: Dict[str, List[Tuple[str, float]]]    # 邻接表
    rooms: List[str]
    starts: Dict[str, str]                       # 响应者 -> 起点节点
    exits: List[str]
    door_x: Dict[str, float]                     # 房间 -> 房门 x 坐标
    room_rects: Dict[str, Tuple[float, float, float, float]]  # 房间矩形(绘图)
    mirror: Dict[str, str]                       # 左右镜像节点映射
    s_search: float
    s_room: float

    # ---- 便捷属性 ----
    @property
    def room_width(self) -> float:
        return self.cfg.room_width

    @property
    def hall_y(self) -> float:
        return self.cfg.hall_width / 2.0

    @property
    def y_top(self) -> float:
        return self.cfg.hall_width / 2.0 + self.cfg.room_depth

    @property
    def x_bounds(self) -> Tuple[float, float]:
        return (-self.cfg.exit_length, self.cfg.hall_length + self.cfg.exit_length)

    def dist(self, u: str, v: str) -> float:
        return self.weight.get((u, v), math.inf)


def _blocked_set(pairs) -> set:
    out = set()
    for u, v in pairs:
        out.add((str(u), str(v)))
        out.add((str(v), str(u)))
    return out


def build_scene(cfg: Config | None = None) -> Scene:
    """由基础变量生成完整建筑网络（节点、边、边权、镜像映射、绘图几何）。"""
    cfg = cfg or Config()
    n = int(cfg.n_rooms_per_side)
    w = cfg.room_width                     # 每间房宽度
    xs = [(k + 0.5) * w for k in range(n)]  # 各房门（房间中心）x 坐标
    door_nodes = [f"N{k + 1}" for k in range(n)]
    top = [f"R{k + 1}" for k in range(n)]            # 走廊上侧 R1..Rn
    bottom = [f"R{n + k + 1}" for k in range(n)]     # 走廊下侧 R(n+1)..R(2n)
    rooms = top + bottom
    exits = ["E_L", "E_R"]

    # ---------------- 节点坐标 ----------------
    L, Lex = cfg.hall_length, cfg.exit_length
    room_y = cfg.hall_width / 2.0 + 0.26 * cfg.room_depth   # 房间节点绘图高度
    pos: Dict[str, Tuple[float, float]] = {
        "E_L": (-Lex, 0.0), "H_L": (0.0, 0.0),
        "H_R": (L, 0.0), "E_R": (L + Lex, 0.0),
    }
    for name, x in zip(door_nodes, xs):
        pos[name] = (x, 0.0)
    room_rects: Dict[str, Tuple[float, float, float, float]] = {}
    door_x: Dict[str, float] = {}
    for i, x in enumerate(xs):
        pos[top[i]] = (x, room_y)
        room_rects[top[i]] = (x - w / 2.0, cfg.hall_width / 2.0, w, cfg.room_depth)
        door_x[top[i]] = x
    for i, x in enumerate(xs):
        pos[bottom[i]] = (x, -room_y)
        room_rects[bottom[i]] = (x - w / 2.0,
                                 -(cfg.hall_width / 2.0 + cfg.room_depth),
                                 w, cfg.room_depth)
        door_x[bottom[i]] = x

    # ---------------- 边（长度 m）----------------
    edges: List[Tuple[str, str, float]] = [
        ("E_L", "H_L", Lex), ("H_R", "E_R", Lex),
    ]
    prev_x, prev_name = 0.0, "H_L"
    for name, x in zip(door_nodes, xs):
        edges.append((prev_name, name, x - prev_x))
        prev_x, prev_name = x, name
    edges.append((prev_name, "H_R", L - prev_x))
    for i, name in enumerate(door_nodes):
        edges.append((name, top[i], cfg.door_length))
        edges.append((name, bottom[i], cfg.door_length))

    # ---------------- 阻断边 x_e = 0 ----------------
    blocked = _blocked_set(cfg.blocked_edges)
    edges = [e for e in edges if (e[0], e[1]) not in blocked]

    # ---------------- 边权与邻接表 ----------------
    weight: Dict[Tuple[str, str], float] = {}
    graph: Dict[str, List[Tuple[str, float]]] = {node: [] for node in pos}
    for u, v, d in edges:
        c = d / cfg.v_hall
        weight[(u, v)] = weight[(v, u)] = c
        graph[u].append((v, c))
        graph[v].append((u, c))

    # ---------------- 响应者起点（默认交替左/右出口，可由参数覆盖）----------------
    if cfg.n_responders > len(RESPONDER_NAMES):
        raise ValueError(f"响应者最多支持 {len(RESPONDER_NAMES)} 名")
    starts = {}
    for i in range(cfg.n_responders):
        override = cfg.responder_starts[i] if i < len(cfg.responder_starts) else None
        starts[RESPONDER_NAMES[i]] = override or ("E_L" if i % 2 == 0 else "E_R")

    # ---------------- 左右镜像映射（用于并列解去重）----------------
    mirror = {"E_L": "E_R", "E_R": "E_L", "H_L": "H_R", "H_R": "H_L"}
    for k in range(n):
        mirror[door_nodes[k]] = door_nodes[n - 1 - k]
        mirror[top[k]] = top[n - 1 - k]
        mirror[bottom[k]] = bottom[n - 1 - k]

    scene = Scene(
        cfg=cfg, pos=pos, edges=edges, weight=weight, graph=graph,
        rooms=rooms, starts=starts, exits=exits, door_x=door_x,
        room_rects=room_rects, mirror=mirror,
        s_search=cfg.s_search, s_room=cfg.s_room,
    )

    # ---------------- 可行性检查：每间房必须可达 ----------------
    reachable = set()
    for s in set(starts.values()):
        reachable |= _reachable(scene, s)
    unreachable = [r for r in rooms if r not in reachable]
    if unreachable:
        raise ValueError(
            "以下房间因阻断边而不可达，请检查 blocked_edges：" + "、".join(unreachable))
    return scene


def _reachable(scene: Scene, source: str) -> set:
    seen, stack = {source}, [source]
    while stack:
        u = stack.pop()
        for v, _c in scene.graph[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def is_mirror_symmetric(scene: Scene) -> bool:
    """建筑网络与响应者起点是否左右镜像对称（对称时才可做镜像去重）。"""
    lengths = {}
    for u, v, d in scene.edges:          # 无向边：两个方向都登记
        lengths[(u, v)] = d
        lengths[(v, u)] = d
    for u, v, d in scene.edges:
        mu, mv = scene.mirror[u], scene.mirror[v]
        if (mu, mv) not in lengths or abs(lengths[(mu, mv)] - d) > 1e-12:
            return False
    mirrored_starts = sorted(scene.mirror[s] for s in scene.starts.values())
    return mirrored_starts == sorted(scene.starts.values())


# ---------------------------------------------------------- 默认场景（基准参数）
DEFAULT_CONFIG = Config()
DEFAULT_SCENE = build_scene(DEFAULT_CONFIG)

# 兼容旧代码的模块级常量（都是基准参数下的取值）
POS = DEFAULT_SCENE.pos
EDGES = DEFAULT_SCENE.edges
WEIGHT = DEFAULT_SCENE.weight
ROOMS = DEFAULT_SCENE.rooms
STARTS = DEFAULT_SCENE.starts
ROOM_RECT = DEFAULT_SCENE.room_rects
S_SEARCH = DEFAULT_SCENE.s_search
S_ROOM = DEFAULT_SCENE.s_room
V_WALK = DEFAULT_CONFIG.v_hall
T0, ALPHA, BETA = DEFAULT_CONFIG.t_check, DEFAULT_CONFIG.alpha, DEFAULT_CONFIG.beta
AREA, OCC = DEFAULT_CONFIG.area, DEFAULT_CONFIG.occupancy
T_REPORT, T_MARK = DEFAULT_CONFIG.t_report, DEFAULT_CONFIG.t_mark


def build_graph(scene: Scene | None = None):
    """返回邻接表 {node: [(neighbor, 通行时间 s), ...]}（无向图）。"""
    return (scene or DEFAULT_SCENE).graph
