# -*- coding: utf-8 -*-
"""最优任务分配与访问顺序（支持任意响应者人数）。

模型（公式见《迪杰斯特拉算法公式.md》）：
  对响应者 r 的访问顺序 sigma_r = (p_1, ..., p_m)：
      T_r(sigma_r) = sum_k [ dist(p_{k-1}, p_k) + s*_{p_k} ],   p_0 = 起点
  其中 dist(·,·) 由 Dijkstra 算法在建筑网络上精确求得，s*_i = s_search + t_rep + t_mark。
  优化目标：min_{房间划分 + 访问顺序}  max_r T_r   —— 全局最优。

求解方式（精确，不做任何对称性假设）：
  1) 对每名响应者、每个房间子集 S，用 Held–Karp 动态规划求出该子集的最短
     访问时间 T_r(S) 与对应顺序；
  2) 枚举所有"房间 -> 响应者"的分配（N^m 种），取 max_r T_r(S_r) 最小者；
  3) 对达到最优的分配，再用排列精确统计并列最优方案数。
这样即使建筑不对称、存在阻断边或响应者人数改变，也不会漏掉全局最优解。
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Tuple

from sweep_building import Scene, is_mirror_symmetric
from sweep_dijkstra import dijkstra, shortest_path

MAX_TIE_ENUM = 2_000_000      # 并列解去重时最多枚举的组合数（防止组合爆炸）


# ------------------------------------------------------------ 最短路预处理
def prepare_pairs(scene: Scene):
    """对关键节点（响应者起点 + 房间）各跑一次 Dijkstra，得到 dist / prev 表。"""
    nodes = list(dict.fromkeys(list(scene.starts.values()) + scene.rooms))
    dij = {n: dijkstra(scene.graph, n) for n in nodes}
    dists = {n: dij[n][0] for n in nodes}
    prevs = {n: dij[n][1] for n in nodes}
    return dists, prevs


# ------------------------------------------------------------ 单条路线评估
def route_eval(route, start, dists, s_room):
    """返回 (路线总时间 s, 其中移动时间 s, 逐段列表 [(u, room, 段耗时), ...])。"""
    travel, node, legs = 0.0, start, []
    for room in route:
        d = dists[node][room]
        if math.isinf(d):
            raise ValueError(f"{node} 无法到达 {room}，请检查 blocked_edges")
        legs.append((node, room, d))
        travel += d
        node = room
    return travel + s_room * len(route), travel, legs


# ------------------------------------------------- Held–Karp：子集最优顺序
def _held_karp(start, rooms, dists, s_room):
    """对固定起点，返回 {子集掩码: (最短总时间, 达到该时间的访问顺序)}。

    dp[mask][last] = 从 start 出发、清查完 mask 中所有房间、最后停在 last 的最短时间。
    """
    m = len(rooms)
    size = 1 << m
    inf = math.inf
    dp = [[inf] * m for _ in range(size)]
    par = [[-1] * m for _ in range(size)]
    for i, room in enumerate(rooms):                     # 边界：只查一间房
        d = dists[start][room]
        if not math.isinf(d):
            dp[1 << i][i] = d + s_room
    for mask in range(size):                             # 转移：追加一间未访问房
        row = dp[mask]
        for last in range(m):
            cur = row[last]
            if cur == inf:
                continue
            for nxt in range(m):
                if mask >> nxt & 1:
                    continue
                nm = mask | (1 << nxt)
                nd = cur + dists[rooms[last]][rooms[nxt]] + s_room
                if nd < dp[nm][nxt] - 1e-12:
                    dp[nm][nxt] = nd
                    par[nm][nxt] = last

    result: Dict[int, Tuple[float, Tuple[str, ...]]] = {}
    for mask in range(1, size):
        best_t, best_last = inf, -1
        for last in range(m):
            if dp[mask][last] < best_t - 1e-12:
                best_t, best_last = dp[mask][last], last
        if best_last < 0:
            continue
        route, mm, last = [], mask, best_last           # 由前驱表还原顺序
        while last >= 0:
            route.append(rooms[last])
            pl = par[mm][last]
            mm ^= 1 << last
            last = pl
        route.reverse()
        result[mask] = (best_t, tuple(route))
    return result


def best_full_route(scene: Scene, dists, responder: str):
    """单一响应者清查全部房间的最优 (时间, 顺序)；不可行时返回 None。"""
    rooms = list(scene.rooms)
    table = _held_karp(scene.starts[responder], rooms, dists, scene.s_room)
    return table.get((1 << len(rooms)) - 1)


# ------------------------------------------------------------ 全局最优
def _mirror_partner(scene: Scene, responders):
    """镜像后各起点的对应响应者（用于并列解去重）。"""
    partner = {}
    for r in responders:
        target = scene.mirror[scene.starts[r]]
        partner[r] = next(rr for rr in responders if scene.starts[rr] == target)
    return partner


def _canonical_key(scene: Scene, responders, partner, routes):
    """把 (路线) 组合规范化为 {原方案, 镜像方案} 中字典序较小者。"""
    direct = tuple(routes[r] for r in responders)
    mirrored = {}
    for r in responders:
        mirrored[partner[r]] = tuple(scene.mirror[x] for x in routes[r])
    mirror_key = tuple(mirrored[r] for r in responders)
    return min(direct, mirror_key)


def optimize(scene: Scene, dists, verbose: bool = True):
    """精确求解全局最优任务分配，返回结果字典。"""
    responders = list(scene.starts)
    rooms = list(scene.rooms)
    m, n_resp = len(rooms), len(responders)

    # 1) 每名响应者对每个子集的最优时间（Held–Karp）
    best_by_resp = {r: _held_karp(scene.starts[r], rooms, dists, scene.s_room)
                    for r in responders}

    # 2) 枚举全部"房间 -> 响应者"分配，取最小 makespan
    evaluated = 0
    best_makespan = math.inf
    best_assignments: List[Tuple[int, ...]] = []
    for assign in itertools.product(range(n_resp), repeat=m):
        masks, sizes = [0] * n_resp, [0] * n_resp
        for i, r_i in enumerate(assign):
            masks[r_i] |= 1 << i
            sizes[r_i] += 1
        times, ok = [], True
        for r_i, r in enumerate(responders):
            mk = masks[r_i]
            entry = best_by_resp[r].get(mk) if mk else (0.0, ())
            if entry is None:            # 该响应者到不了分到的房间 → 该分配不可行
                ok = False
                break
            times.append(entry[0])
        if not ok:
            continue
        evaluated += math.prod(math.factorial(s) for s in sizes)   # 只统计可行分配
        mt = max(times)
        if mt < best_makespan - 1e-9:
            best_makespan, best_assignments = mt, [tuple(masks)]
        elif abs(mt - best_makespan) <= 1e-9:
            best_assignments.append(tuple(masks))

    if not best_assignments:
        raise ValueError("无可行解：请检查 blocked_edges 是否切断了房间")

    # 3) 并列最优方案计数（并对镜像等价解去重）
    do_dedup = bool(scene.cfg.dedup_mirror_ties) and is_mirror_symmetric(scene)
    est_ties = sum(math.prod(math.factorial(bin(mk).count("1")) for mk in masks)
                   for masks in best_assignments)
    track_keys = do_dedup and est_ties <= MAX_TIE_ENUM
    if do_dedup and not track_keys and verbose:
        print(f"（并列解约 {est_ties} 个，超过去重上限 {MAX_TIE_ENUM}，仅统计总数）")
    partner = _mirror_partner(scene, responders) if track_keys else None
    tie_keys = set()
    n_ties = 0
    rep_key, rep_routes = None, None
    for masks in best_assignments:
        groups = [[rooms[i] for i in range(m) if masks[r_i] >> i & 1]
                  for r_i in range(n_resp)]
        per_resp = []
        for r_i, r in enumerate(responders):
            per_resp.append([(route_eval(p, scene.starts[r], dists, scene.s_room)[0], p)
                             for p in itertools.permutations(groups[r_i])])
        for combo in itertools.product(*per_resp):
            if abs(max(t for t, _ in combo) - best_makespan) > 1e-9:
                continue
            n_ties += 1
            routes = {r: combo[i][1] for i, r in enumerate(responders)}
            key = tuple(routes[r] for r in responders)
            if rep_key is None or key < rep_key:     # 取字典序最小的代表解（结果可复现）
                rep_key, rep_routes = key, routes
            if track_keys:
                tie_keys.add(_canonical_key(scene, responders, partner, routes))

    # 4) 取一个最优代表解（字典序最小的并列最优方案）
    reps = rep_routes or {}
    routes, times, travel, legs = {}, {}, {}, {}
    for r in responders:
        route = reps.get(r, ())
        total, tr, lg = route_eval(route, scene.starts[r], dists, scene.s_room)
        routes[r], times[r], travel[r], legs[r] = list(route), total, tr, lg

    sol = {
        "makespan": best_makespan,
        "routes": routes,
        "times": times,
        "travel": travel,
        "legs": legs,
        "evaluated": evaluated,
        "n_ties": n_ties,
        "n_ties_distinct": len(tie_keys) if track_keys else None,
        "n_assignments_optimal": len(best_assignments),
        "responders": responders,
    }
    if verbose:
        print(f"枚举候选方案 {evaluated} 个（房间分配 × 访问顺序，全局精确求解）")
        print(f"并列最优方案数：{n_ties}"
              + (f"（镜像去重后 {len(tie_keys)} 个）" if track_keys else ""))
    return sol


def build_schedule(scene: Scene, routes, prevs):
    """把最优路线展开为带时间戳的动作序列（移动 / 搜索 / 报告 / 标记）。

    返回 schedules[r] = [ {'kind', 't0', 't1', ...}, ... ] 与每间房清查完成时刻。
    """
    cfg = scene.cfg
    schedules, clear_times = {}, {}
    for r, route in routes.items():
        segs, t, node = [], 0.0, scene.starts[r]
        for room in route:
            path = shortest_path(prevs[node], node, room)
            if not path:
                raise ValueError(f"{node} 无法到达 {room}，请检查 blocked_edges")
            for u, v in zip(path[:-1], path[1:]):
                c = scene.weight[(u, v)]
                segs.append({"kind": "move", "t0": t, "t1": t + c, "u": u, "v": v})
                t += c
            segs.append({"kind": "search", "t0": t, "t1": t + scene.s_search, "room": room})
            t += scene.s_search
            segs.append({"kind": "report", "t0": t, "t1": t + cfg.t_report, "room": room})
            t += cfg.t_report
            segs.append({"kind": "mark", "t0": t, "t1": t + cfg.t_mark, "room": room})
            t += cfg.t_mark
            clear_times[room] = t
            node = room
        schedules[r] = segs
    return schedules, clear_times
