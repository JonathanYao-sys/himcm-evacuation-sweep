# -*- coding: utf-8 -*-
"""主入口：基础场景 Dijkstra 建模 + 最优清查方案 + 图表动画。

**改基础变量后，重跑这一条命令就够了。**

用法：
    python3 src/run_sweep.py                        # 用 sweep_config.py 里的基准参数
    python3 src/run_sweep.py --set v_hall=0.8       # 低能见度/谨慎移动
    python3 src/run_sweep.py --set t_check=25       # 搜索更困难
    python3 src/run_sweep.py --set occupancy=8      # 人数较多
    python3 src/run_sweep.py --set hall_length=40 --set n_rooms_per_side=4
    python3 src/run_sweep.py --responders 3         # 1/2/3 名响应者
    python3 src/run_sweep.py --block N1 N2          # 阻断走廊边（x_e = 0）
    python3 src/run_sweep.py --config p.json --save-config used.json
    python3 src/run_sweep.py --no-animation         # 跳过 GIF，秒出结果

产出：
  figures/fig_sweep_network.png   建筑网络（边权=通行时间）
  figures/fig_sweep_optimal.png   最优路线静态图
  figures/sweep_animation.gif     最优解清查过程动画
  output/sweep_results.json       全部计算结果（含本次使用的参数）
  output/sweep_params.md          本次参数表（可直接贴进论文）
"""
from __future__ import annotations

import json
import math

from sweep_building import FIG_DIR, OUT_DIR, build_scene
from sweep_config import resolve_config, save_config
from sweep_dijkstra import shortest_path
from sweep_optimize import best_full_route, build_schedule, optimize, prepare_pairs
from sweep_visualize import fig_network, fig_optimal_routes, make_animation


def r3(x):
    return round(x, 3)


def param_table(scene):
    """返回本次运行的实际参数表：[(类别, 符号, 含义, 取值, 单位), ...]。"""
    c = scene.cfg
    rows = [
        ("建筑几何", "n", "每侧房间数", c.n_rooms_per_side, "间"),
        ("建筑几何", "L_h", "中央走廊长度", c.hall_length, "m"),
        ("建筑几何", "L_exit", "出口至走廊端点", c.exit_length, "m"),
        ("建筑几何", "L_door", "房门至走廊路径", c.door_length, "m"),
        ("建筑几何", "W_h", "走廊宽度", c.hall_width, "m"),
        ("建筑几何", "—", "房间进深（绘图）", c.room_depth, "m"),
        ("建筑几何", "—", "每间房宽度 = L_h/n", c.room_width, "m"),
        ("建筑几何", "—", "房间总数", c.n_rooms, "间"),
        ("房间与搜索", "a_i", "每间办公室面积", c.area, "m²"),
        ("房间与搜索", "n_i", "每间预计人数", c.occupancy, "人"),
        ("房间与搜索", "t_0", "固定检查时间", c.t_check, "s"),
        ("房间与搜索", "α", "面积搜索系数", c.alpha, "s/m²"),
        ("房间与搜索", "β", "每人引导时间", c.beta, "s/人"),
        ("房间与搜索", "s_i", "单间搜索与引导 = t_0+αa_i+βn_i", scene.s_search, "s"),
        ("房间与搜索", "t_rep", "通信报告时间", c.t_report, "s"),
        ("房间与搜索", "t_mark", "清查标记时间", c.t_mark, "s"),
        ("房间与搜索", "s_i*", "单间总处理时间", scene.s_room, "s"),
        ("移动", "v_h", "走廊/房门行走速度", c.v_hall, "m/s"),
        ("移动", "v_room", "房间内搜索速度", c.v_room, "m/s"),
        ("响应者", "—", "响应者人数", c.n_responders, "人"),
        ("响应者", "—", "各响应者起点",
         "、".join(f"{r}={s}" for r, s in scene.starts.items()), ""),
        ("可通行性", "x_e", "被阻断的边（x_e=0）",
         "、".join(f"{u}–{v}" for u, v in c.blocked_edges) or "无", ""),
        ("派生网络", "|V|", "网络节点数", len(scene.pos), "个"),
        ("派生网络", "|E|", "网络边数", len(scene.edges), "条"),
    ]
    return rows


def print_params(scene):
    print("=" * 78)
    print("本次运行使用的基础变量")
    print("=" * 78)
    group = None
    for g, sym, meaning, value, unit in param_table(scene):
        if g != group:
            print(f"[{g}]")
            group = g
        val = f"{value:g}" if isinstance(value, (int, float)) else str(value)
        print(f"   {sym:<8} {meaning:<30} = {val} {unit}")
    print()


def write_params_md(scene, path):
    def esc(s):
        return str(s).replace("|", "\\|")

    lines = ["# 本次运行的基础参数（由代码自动生成，勿手改）", "",
             "| 类别 | 符号 | 含义 | 取值 | 单位 |", "|---|---|---|---:|---|"]
    for g, sym, meaning, value, unit in param_table(scene):
        val = f"{value:g}" if isinstance(value, (int, float)) else str(value)
        lines.append(f"| {esc(g)} | {esc(sym)} | {esc(meaning)} | {esc(val)} | {esc(unit)} |")
    lines.append("")
    lines.append("> 复现命令：`python3 src/run_sweep.py`；"
                 "参数集中定义在 `src/sweep_config.py` 的 `Config` 中，"
                 "也可用 `--set KEY=VALUE` 临时覆盖。")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    cfg, args = resolve_config(argv)
    scene = build_scene(cfg)
    print_params(scene)
    full_edges = 3 * cfg.n_rooms_per_side + 3          # 2 出口段 + (n+1) 走廊段 + 2n 门
    removed = full_edges - len(scene.edges)
    print(f"建筑网络：|V|={len(scene.pos)}, |E|={len(scene.edges)}"
          + (f"（因 x_e = 0 删除 {removed} 条边）" if removed else ""))
    print(f"单间处理时间 s* = {scene.s_search:.0f} + {cfg.t_report:g} + "
          f"{cfg.t_mark:g} = {scene.s_room:.0f} s")

    # 1) Dijkstra：响应者起点与房间两两最短通行时间
    key_nodes = list(dict.fromkeys(list(scene.starts.values()) + scene.rooms))
    dists, prevs = prepare_pairs(scene)
    print("\n最短通行时间 dist(u,v)（秒，Dijkstra 求得）：")
    print("      " + "".join(f"{v:>8}" for v in key_nodes))
    for u in key_nodes:
        print(f"{u:>5} " + "".join(f"{dists[u][v]:8.2f}" for v in key_nodes))

    # 2) 精确求解全局最优任务分配
    sol = optimize(scene, dists)
    sol["starts"] = scene.starts
    T = sol["makespan"]
    print("\n===== 最优解 =====")
    for resp in sol["responders"]:
        route = sol["routes"][resp]
        if not route:
            print(f"响应者 {resp}：待命（无房间）")
            continue
        print(f"响应者 {resp}（{scene.starts[resp]} 进入）："
              f"{' → '.join([scene.starts[resp]] + route)}")
        for u, v, d in sol["legs"][resp]:
            path = shortest_path(prevs[u], u, v)
            print(f"    {u} → {v}: {'-'.join(path)}"
                  f"  {d * cfg.v_hall:5.1f} m = {d:6.2f} s")
        print(f"    移动合计 {sol['travel'][resp] * cfg.v_hall:.1f} m"
              f" = {sol['travel'][resp]:.2f} s，搜索 {len(route)}×{scene.s_room:g} s，"
              f"T_{resp} = {sol['times'][resp]:.2f} s")
    print(f"\n整楼清查完成时间 T = max_r T_r = {T:.2f} s ≈ {T / 60:.2f} min")

    # 3) 对照实验
    first = sol["responders"][0]
    single = best_full_route(scene, dists, first)
    if single is None:
        best1 = None
        print(f"对照：仅 1 名响应者时不可行（从 {scene.starts[first]} 出发无法到达全部房间，"
              f"网络被阻断边分割）")
    else:
        best1, _route1 = single
        print(f"对照：仅 1 名响应者（{scene.starts[first]} 进入）时最优 "
              f"T₁ = {best1:.2f} s ≈ {best1 / 60:.2f} min")
    back = 0.0
    for resp in sol["responders"]:
        route = sol["routes"][resp]
        if not route:
            continue
        last = route[-1]
        d = min((dists[last][e] for e in scene.exits), default=math.inf)
        if math.isinf(d):
            back = math.inf
            break
        back = max(back, sol["times"][resp] + d)
    if math.isinf(back):
        back = None
        print("对照：要求清查后退回最近出口时不可行（出口不可达）")
    else:
        print(f"对照：若要求清查后退回最近出口，T = {back:.2f} s ≈ {back / 60:.2f} min")

    # 4) 时间轴与每间房清查完成时刻
    schedules, clear_times = build_schedule(scene, sol["routes"], prevs)
    print("\n各房间清查完成时刻（s）：")
    for r in scene.rooms:
        print(f"    {r}: {clear_times[r]:7.2f}")

    # 5) 保存 JSON 与参数表
    OUT_DIR.mkdir(exist_ok=True)
    result = {
        "params": cfg.to_dict(),
        "scene": {
            "nodes": {n: list(xy) for n, xy in scene.pos.items()},
            "edges": [{"u": u, "v": v, "length_m": d, "weight_s": r3(d / cfg.v_hall)}
                      for u, v, d in scene.edges],
            "rooms": scene.rooms, "starts": scene.starts, "exits": scene.exits,
            "door_x": scene.door_x,
        },
        "dist_s": {u: {v: (r3(dists[u][v]) if not math.isinf(dists[u][v]) else None)
                       for v in key_nodes} for u in key_nodes},
        "optimal": {
            "makespan_s": r3(T), "makespan_min": r3(T / 60),
            "responders": sol["responders"],
            "routes": sol["routes"],
            "times_s": {k: r3(v) for k, v in sol["times"].items()},
            "travel_m": {k: r3(v * cfg.v_hall) for k, v in sol["travel"].items()},
            "clear_times_s": {k: r3(v) for k, v in clear_times.items()},
            "evaluated": sol["evaluated"], "n_ties": sol["n_ties"],
            "n_ties_distinct": sol["n_ties_distinct"],
        },
        "baselines": {"single_responder_s": r3(best1) if best1 is not None else None,
                      "return_to_exit_s": r3(back) if back is not None else None},
        "schedules": schedules,
    }
    with open(OUT_DIR / "sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {OUT_DIR / 'sweep_results.json'}")
    write_params_md(scene, OUT_DIR / "sweep_params.md")
    print(f"已保存 {OUT_DIR / 'sweep_params.md'}")
    if args.save_config:
        p = save_config(cfg, args.save_config)
        print(f"已保存本次参数 {p}")

    # 6) 图与动画
    FIG_DIR.mkdir(exist_ok=True)
    fig_network(scene, FIG_DIR / "fig_sweep_network.png")
    print(f"已保存 {FIG_DIR / 'fig_sweep_network.png'}")
    fig_optimal_routes(scene, sol, prevs, clear_times, FIG_DIR / "fig_sweep_optimal.png")
    print(f"已保存 {FIG_DIR / 'fig_sweep_optimal.png'}")
    if args.no_animation:
        print("（--no-animation：跳过动画 GIF）")
    else:
        n = make_animation(scene, sol, schedules, T, FIG_DIR / "sweep_animation.gif")
        print(f"已保存 {FIG_DIR / 'sweep_animation.gif'}（{n} 帧）")


if __name__ == "__main__":
    main()
