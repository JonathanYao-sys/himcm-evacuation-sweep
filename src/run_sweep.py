# -*- coding: utf-8 -*-
"""主入口：基础场景 Dijkstra 建模 + 最优清查方案 + 动画。

用法：python3 src/run_sweep.py
产出：
  figures/fig_sweep_network.png   建筑网络（边权=通行时间）
  figures/fig_sweep_optimal.png   最优路线静态图
  figures/sweep_animation.gif     最优解清查过程动画
  output/sweep_results.json       全部计算结果
"""
import json

from sweep_building import (ROOT, FIG_DIR, OUT_DIR, V_WALK, S_SEARCH, S_ROOM,
                            ROOMS, STARTS, EDGES, build_graph)
from sweep_dijkstra import dijkstra, shortest_path
from sweep_optimize import prepare_pairs, optimize, build_schedule, route_eval
from sweep_building import WEIGHT
from sweep_visualize import fig_network, fig_optimal_routes, make_animation


def r3(x):
    return round(x, 3)


def main():
    graph = build_graph()
    print(f"建筑网络：|V|={len(graph)}, |E|={len(EDGES)}")
    print(f"单间处理时间 s* = {S_SEARCH:.0f} + 5 + 3 = {S_ROOM:.0f} s")

    # 1) Dijkstra：出口与房间两两最短通行时间
    key_nodes = list(STARTS.values()) + ROOMS
    dists, prevs = prepare_pairs(graph, key_nodes)
    print("\n最短通行时间 dist(u,v)（秒，Dijkstra 求得）：")
    header = "      " + "".join(f"{v:>8}" for v in key_nodes)
    print(header)
    for u in key_nodes:
        print(f"{u:>5} " + "".join(f"{dists[u][v]:8.2f}" for v in key_nodes))

    # 2) 枚举求最优任务分配
    sol = optimize(ROOMS, STARTS, dists, S_ROOM)
    sol["starts"] = STARTS
    T = sol["makespan"]
    print(f"\n枚举候选方案 {sol['evaluated']} 个（固定 R1→A 消除镜像对称）")
    print(f"并列最优方案数：{sol['n_ties']}")
    print("\n===== 最优解 =====")
    for resp in ("A", "B"):
        route = sol["routes"][resp]
        legs = sol["legs"][resp]
        print(f"响应者 {resp}（{STARTS[resp]} 进入）：{' → '.join([STARTS[resp]] + route)}")
        for u, v, d in legs:
            path = shortest_path(prevs[u], u, v)
            print(f"    {u} → {v}: {'-'.join(path)}"
                  f"  {d * V_WALK:5.1f} m = {d:6.2f} s")
        print(f"    移动合计 {sol['travel'][resp] * V_WALK:.1f} m"
              f" = {sol['travel'][resp]:.2f} s，搜索 {len(route)}×76 s，"
              f"T_{resp} = {sol['times'][resp]:.2f} s")
    print(f"\n整楼清查完成时间 T = max(T_A, T_B) = {T:.2f} s ≈ {T / 60:.2f} min")

    # 对照：仅 1 名响应者（全部 6 间房）
    import itertools
    best1 = min(route_eval(p, STARTS["A"], dists, S_ROOM)[0]
                for p in itertools.permutations(ROOMS))
    print(f"对照：仅 1 名响应者时最优 T₁ = {best1:.2f} s ≈ {best1 / 60:.2f} min")

    # 对照：清查完成后退回最近出口
    back = 0.0
    for resp in ("A", "B"):
        last = sol["routes"][resp][-1]
        back = max(back, sol["times"][resp] + min(dists[last]["E_L"], dists[last]["E_R"]))
    print(f"对照：若要求清查后退回最近出口，T = {back:.2f} s ≈ {back / 60:.2f} min")

    # 3) 时间轴与每间房清查完成时刻
    schedules, clear_times = build_schedule(sol["routes"], STARTS, prevs, WEIGHT)
    print("\n各房间清查完成时刻（s）：")
    for r in ROOMS:
        print(f"    {r}: {clear_times[r]:7.2f}")

    # 4) 保存 JSON
    OUT_DIR.mkdir(exist_ok=True)
    result = {
        "params": {"v_walk": V_WALK, "s_search": S_SEARCH, "s_room": S_ROOM,
                   "t_report": 5, "t_mark": 3, "rooms": ROOMS, "starts": STARTS},
        "edges": [{"u": u, "v": v, "length_m": d, "weight_s": r3(d / V_WALK)}
                  for u, v, d in EDGES],
        "dist_s": {u: {v: r3(dists[u][v]) for v in key_nodes} for u in key_nodes},
        "optimal": {
            "makespan_s": r3(T), "makespan_min": r3(T / 60),
            "routes": sol["routes"],
            "times_s": {k: r3(v) for k, v in sol["times"].items()},
            "travel_m": {k: r3(v * V_WALK) for k, v in sol["travel"].items()},
            "clear_times_s": {k: r3(v) for k, v in clear_times.items()},
            "evaluated": sol["evaluated"], "n_ties": sol["n_ties"],
        },
        "baselines": {"single_responder_s": r3(best1),
                      "return_to_exit_s": r3(back)},
        "schedules": schedules,
    }
    with open(OUT_DIR / "sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {OUT_DIR / 'sweep_results.json'}")

    # 5) 图与动画
    FIG_DIR.mkdir(exist_ok=True)
    fig_network(FIG_DIR / "fig_sweep_network.png")
    print(f"已保存 {FIG_DIR / 'fig_sweep_network.png'}")
    fig_optimal_routes(sol, prevs, clear_times, FIG_DIR / "fig_sweep_optimal.png")
    print(f"已保存 {FIG_DIR / 'fig_sweep_optimal.png'}")
    n = make_animation(sol, schedules, T, FIG_DIR / "sweep_animation.gif")
    print(f"已保存 {FIG_DIR / 'sweep_animation.gif'}（{n} 帧）")


if __name__ == "__main__":
    main()
