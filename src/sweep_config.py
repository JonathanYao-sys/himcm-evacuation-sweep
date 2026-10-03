# -*- coding: utf-8 -*-
"""基础场景参数（单一数据源 / single source of truth）。

《基础场景建模假设.md》里的每一个"基础变量"都集中在本文件的 `Config` 中。
改完之后**只需要重跑一次**

    python3 src/run_sweep.py

就能得到与新参数完全一致的最优解、时间轴、图表和 JSON —— 因为建筑几何
（走廊长度、出口段、门到走廊、每侧房间数）、边权、房间处理时间、图题与
图注全部由这里推导，不再有任何手工硬编码的数字。

也可以**完全不动代码**，用命令行临时覆盖：

    python3 src/run_sweep.py --set v_hall=0.8            # 低能见度/谨慎移动
    python3 src/run_sweep.py --set t_check=25 --set alpha=1.0
    python3 src/run_sweep.py --set area=30 --set occupancy=8
    python3 src/run_sweep.py --responders 1              # 1/2/3 名响应者
    python3 src/run_sweep.py --block N1 N2               # 阻断走廊边 (x_e = 0)
    python3 src/run_sweep.py --config my_params.json     # 从 JSON 读参数
    python3 src/run_sweep.py --config p.json --save-config used.json
    python3 src/run_sweep.py --no-animation               # 跳过 GIF，秒出结果
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Tuple

# 所有参数都取自《基础场景建模假设.md》第 2/3/4 节，单位写在注释里。
FLOAT_FIELDS = {"hall_length", "exit_length", "door_length", "hall_width",
                "room_depth", "area", "v_hall", "v_room",
                "t_check", "alpha", "beta", "t_report", "t_mark"}
INT_FIELDS = {"n_rooms_per_side", "occupancy", "n_responders"}


@dataclass
class Config:
    # ---------------- 建筑几何（m）----------------
    n_rooms_per_side: int = 3      # 每侧房间数 n（基准：3 间，共 6 间）
    hall_length: float = 30.0      # L_h  中央走廊长度
    exit_length: float = 2.0       # L_exit 出口至走廊端点距离
    door_length: float = 1.0       # L_door 房门至走廊路径长度
    hall_width: float = 2.0        # W_h  走廊宽度（绘图与房间定位用）
    room_depth: float = 5.0        # 房间进深（绘图用；房间宽度 = L_h / n）

    # ---------------- 房间属性 ----------------
    area: float = 20.0             # a_i  每间办公室面积 m²
    occupancy: int = 4             # n_i  每间预计人数

    # ---------------- 移动速度（m/s）----------------
    v_hall: float = 1.2            # v_h  走廊/房门通道行走速度（决定边权）
    v_room: float = 0.5            # v_room 房间内搜索移动速度（背景参数）

    # ---------------- 房间处理时间模型（s）----------------
    t_check: float = 20.0          # t_0   固定检查时间
    alpha: float = 0.8             # alpha s/m² 面积搜索系数
    beta: float = 8.0              # beta  s/人 每人引导时间
    t_report: float = 5.0          # t_rep  每间房通信报告时间
    t_mark: float = 3.0            # t_mark 每间房清查标记时间

    # ---------------- 响应者 ----------------
    n_responders: int = 2          # 响应者人数（交替从左/右出口进入）

    # ---------------- 可通行性 ----------------
    # 被阻断的边（x_e = 0，从网络中删除），元素为任意顺序的节点对。
    blocked_edges: List[Tuple[str, str]] = field(default_factory=list)

    # 各响应者起点（可选）：如 ["E_L", "E_L"]；留空则交替从左/右出口进入。
    responder_starts: List[str] = field(default_factory=list)

    # ---------------- 求解选项 ----------------
    # 固定访问顺序中"镜像等价"的重复解（仅用于报告并列解个数，不影响最优值）
    dedup_mirror_ties: bool = False

    # ============ 由基础变量推导出来的量（不要手工改） ============
    @property
    def room_width(self) -> float:
        """每间房宽度 = 走廊长度 / 每侧房间数。"""
        return self.hall_length / self.n_rooms_per_side

    @property
    def n_rooms(self) -> int:
        return 2 * self.n_rooms_per_side

    @property
    def s_search(self) -> float:
        """单间搜索与引导时间 s_i = t_0 + alpha*a_i + beta*n_i。"""
        return self.t_check + self.alpha * self.area + self.beta * self.occupancy

    @property
    def s_room(self) -> float:
        """单间总处理时间 s_i* = s_i + t_rep + t_mark。"""
        return self.s_search + self.t_report + self.t_mark

    # ---------------- 序列化 ----------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["blocked_edges"] = [list(e) for e in self.blocked_edges]
        d["derived"] = {
            "room_width_m": self.room_width,
            "n_rooms": self.n_rooms,
            "s_search_s": self.s_search,
            "s_room_s": self.s_room,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        return cls(**clean)

    def __post_init__(self):
        # JSON 读进来时 blocked_edges 是 list[list]，统一成 tuple 便于哈希/比较
        self.blocked_edges = [tuple(e) for e in self.blocked_edges]
        self.responder_starts = [str(s) for s in (self.responder_starts or [])]
        for s in self.responder_starts:
            if s not in ("E_L", "E_R"):
                raise ValueError(f"响应者起点只能是 E_L 或 E_R，收到 {s!r}")
        if self.n_rooms_per_side < 1:
            raise ValueError("n_rooms_per_side 必须 >= 1")
        if self.n_responders < 1:
            raise ValueError("n_responders 必须 >= 1")
        if self.hall_length <= 0:
            raise ValueError("hall_length 必须 > 0")
        if self.room_depth <= 0:
            raise ValueError("room_depth 必须 > 0")
        if self.v_hall <= 0:
            raise ValueError("v_hall 必须 > 0")


# ----------------------------------------------------------------- 参数覆盖
def set_param(cfg: Config, key: str, raw, quiet: bool = False) -> Config:
    """把字符串/数字写进 cfg[key]，自动按字段类型转换。"""
    valid = {f.name for f in fields(cfg)}
    if key not in valid:
        raise ValueError(f"未知参数 {key!r}；可用参数：{', '.join(sorted(valid))}")
    if key == "blocked_edges":
        if isinstance(raw, str):
            pairs = [p for p in raw.replace(";", ",").split(",") if p.strip()]
            cfg.blocked_edges = [tuple(p.replace(" ", "-").split("-")) for p in pairs]
        else:
            cfg.blocked_edges = [tuple(e) for e in raw]
    elif key == "responder_starts":
        if isinstance(raw, str):
            cfg.responder_starts = [s.strip() for s in raw.replace("、", ",").split(",") if s.strip()]
        else:
            cfg.responder_starts = [str(s) for s in raw]
    elif isinstance(raw, str):
        if key in INT_FIELDS:
            cfg.__dict__[key] = int(float(raw))
        elif key in FLOAT_FIELDS:
            cfg.__dict__[key] = float(raw)
        elif key == "dedup_mirror_ties":
            cfg.__dict__[key] = raw.strip().lower() in ("1", "true", "yes", "y", "on")
        else:
            cfg.__dict__[key] = raw
    else:
        cfg.__dict__[key] = raw
    if not quiet:
        print(f"[参数覆盖] {key} = {getattr(cfg, key)}")
    return cfg


def block_edge(cfg: Config, u: str, v: str, quiet: bool = False) -> Config:
    """把边 (u, v) 设为不可通行（x_e = 0）。"""
    cfg.blocked_edges = list(cfg.blocked_edges) + [(u, v)]
    if not quiet:
        print(f"[参数覆盖] 阻断边 {u}–{v} (x_e = 0)")
    return cfg


def load_config(path) -> Config:
    with open(Path(path), encoding="utf-8") as f:
        return Config.from_dict(json.load(f))


def save_config(cfg: Config, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
    return path


# ----------------------------------------------------------------- 命令行
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="基础场景最优清查方案（参数可全部从命令行覆盖）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", metavar="FILE", help="从 JSON 文件读取基础参数")
    p.add_argument("--set", dest="sets", action="append", default=[],
                   metavar="KEY=VALUE", help="覆盖单个基础变量，可重复")
    p.add_argument("--responders", type=int, metavar="N", help="响应者人数")
    p.add_argument("--block", action="append", nargs=2, default=[],
                   metavar=("U", "V"), help="阻断边 U–V，可重复")
    p.add_argument("--save-config", metavar="FILE", help="把本次实际使用的参数存成 JSON")
    p.add_argument("--no-animation", action="store_true", help="跳过动画 GIF（更快）")
    return p


def resolve_config(argv=None) -> Tuple[Config, argparse.Namespace]:
    """解析命令行，返回 (最终参数, 其它选项)。"""
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else Config()
    for pair in args.sets:
        if "=" not in pair:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式，收到 {pair!r}")
        key, _, raw = pair.partition("=")
        set_param(cfg, key.strip(), raw.strip())
    if args.responders is not None:
        set_param(cfg, "n_responders", args.responders)
    for u, v in args.block:
        block_edge(cfg, u, v)
    cfg.__post_init__()          # 覆盖后重新校验
    return cfg, args
