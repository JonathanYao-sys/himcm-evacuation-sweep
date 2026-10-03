# -*- coding: utf-8 -*-
"""两名响应者的最优任务分配与访问顺序。

模型（公式见《迪杰斯特拉算法公式.md》）：
  对响应者 r 的访问顺序 sigma_r = (p_1, ..., p_m)：
      T_r(sigma_r) = sum_k [ dist(p_{k-1}, p_k) + s*_{p_k} ],   p_0 = 起点
  其中 dist(·,·) 由 Dijkstra 算法在建筑网络上精确求得，s*_i = 76 s。
  优化目标：min_{房间划分 + 访问顺序}  max(T_A, T_B)

6 间房、2 名响应者，固定 R1 归 A 消除镜像对称后仅 2520 个候选方案，
直接枚举即得全局最优解。
"""
import itertools

from sweep_dijkstra import dijkstra, shortest_path
from sweep_building import S_SEARCH, T_REPORT, T_MARK


def prepare_pairs(graph, nodes):
    """对关键节点各跑一次 Dijkstra，得到两两最短路表 dists 与前驱表 prevs。"""
    dij = {n: dijkstra(graph, n) for n in nodes}
    dists = {n: dij[n][0] for n in nodes}
    prevs = {n: dij[n][1] for n in nodes}
    return dists, prevs


def route_eval(route, start, dists, s_room):
    """返回 (路线总时间 s, 其中移动时间 s, 逐段列表 [(u, room, 段耗时), ...])。"""
    travel, node, legs = 0.0, start, []
    for room in route:
        d = dists[node][room]
        legs.append((node, room, d))
        travel += d
        node = room
    return travel + s_room * len(route), travel, legs


def optimize(rooms, starts, dists, s_room):
    """枚举全部 房间划分 x 访问顺序，返回最优解字典。"""
    rooms = list(rooms)
    first, rest = rooms[0], rooms[1:]          # 固定 R1 归 A，消除左右镜像对称
    best, n_ties, evaluated = None, 0, 0
    for k in range(len(rest) + 1):
        for extra in itertools.combinations(rest, k):
            roomsA = (first, *extra)
            setA = set(roomsA)
            roomsB = tuple(r for r in rooms if r not in setA)
            evalA = [(route_eval(p, starts["A"], dists, s_room), p)
                     for p in itertools.permutations(roomsA)]
            evalB = [(route_eval(p, starts["B"], dists, s_room), p)
                     for p in itertools.permutations(roomsB)]
            for (tA, trA, legsA), pA in evalA:
                for (tB, trB, legsB), pB in evalB:
                    evaluated += 1
                    m = max(tA, tB)
                    if best is None or m < best["makespan"] - 1e-9:
                        best = {"makespan": m,
                                "routes": {"A": list(pA), "B": list(pB)},
                                "times": {"A": tA, "B": tB},
                                "travel": {"A": trA, "B": trB},
                                "legs": {"A": legsA, "B": legsB}}
                        n_ties = 1
                    elif abs(m - best["makespan"]) <= 1e-9:
                        n_ties += 1
    best["evaluated"] = evaluated
    best["n_ties"] = n_ties
    return best


def build_schedule(routes, starts, prevs, weight,
                   s_search=S_SEARCH, t_report=T_REPORT, t_mark=T_MARK):
    """把最优路线展开为带时间戳的动作序列（移动 / 搜索 / 报告 / 标记）。

    返回 schedules[r] = [ {'kind', 't0', 't1', ...}, ... ] 与每间房清查完成时刻。
    """
    schedules, clear_times = {}, {}
    for r, route in routes.items():
        segs, t, node = [], 0.0, starts[r]
        for room in route:
            path = shortest_path(prevs[node], node, room)
            for u, v in zip(path[:-1], path[1:]):
                c = weight[(u, v)]
                segs.append({"kind": "move", "t0": t, "t1": t + c, "u": u, "v": v})
                t += c
            segs.append({"kind": "search", "t0": t, "t1": t + s_search, "room": room})
            t += s_search
            segs.append({"kind": "report", "t0": t, "t1": t + t_report, "room": room})
            t += t_report
            segs.append({"kind": "mark", "t0": t, "t1": t + t_mark, "room": room})
            t += t_mark
            clear_times[room] = t
            node = room
        schedules[r] = segs
    return schedules, clear_times
