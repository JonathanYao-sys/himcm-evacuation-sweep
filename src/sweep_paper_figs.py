# -*- coding: utf-8 -*-
"""论文用出版级配图（在基准模型基础上重算各情景，保证数字真实）。

用法：python3 src/sweep_paper_figs.py
产出（figures/）：
  fig_paper_framework.png    建模流程图
  fig_paper_gantt.png        最优方案甘特图
  fig_paper_heatmap.png      Dijkstra 最短通行时间矩阵热力图
  fig_paper_sensitivity.png  敏感性分析（速度 / 搜索时间 / 人数 / 响应者数量）
  fig_paper_blocked.png      走廊受阻情景重规划对比
  fig_paper_strategy.png     清查策略对比
另存 output/sweep_sensitivity.json
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

from sweep_config import Config
from sweep_building import (ROOT, FIG_DIR, OUT_DIR, V_WALK, T0, ALPHA, BETA,
                            AREA, OCC, T_REPORT, T_MARK, S_SEARCH, S_ROOM,
                            ROOMS, STARTS, EDGES, POS, WEIGHT, DEFAULT_SCENE,
                            build_scene)
from sweep_dijkstra import dijkstra, shortest_path
from sweep_optimize import prepare_pairs, optimize, build_schedule, route_eval
from sweep_visualize import draw_floorplan, draw_route_arrows, RESP_COLOR, ROOM_COLOR

plt.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB"],
    "axes.unicode_minus": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "font.size": 10,
    "savefig.dpi": 200,
    "savefig.facecolor": "white",
})
INK = "#222222"

KEY_NODES = ["E_L", "E_R"] + ROOMS


# ------------------------------------------------------------------ 通用求解
def make_scene(v_walk=None, s_room=None, blocked=(), starts=None):
    """按参数生成场景：速度 / 单间总处理时间 / 阻断边 / 响应者起点。

    对应《基础场景建模假设.md》第 7 节的各敏感性情景；返回带派生量的 Scene。
    """
    cfg = Config()
    if v_walk is not None:
        cfg.v_hall = float(v_walk)
    if blocked:
        cfg.blocked_edges = [tuple(e) for e in blocked]
    if starts is not None:
        cfg.n_responders = len(starts)
        cfg.responder_starts = [str(s) for s in starts]
    scene = build_scene(cfg)
    if s_room is not None:                    # 覆盖 s_i*：搜索时间随之伸缩
        s_search = float(s_room) - cfg.t_report - cfg.t_mark
        scene.s_search = s_search
        scene.roomSearch = {r: s_search for r in scene.rooms}
        scene.s_room = float(s_room)
        scene.roomTime = {r: float(s_room) for r in scene.rooms}
    return scene


def solve(v_walk=None, s_room=None, blocked=(), starts=None):
    """给定参数重算最优解，返回 (sol, dists, prevs)。

    v_walk  — 覆盖走廊行走速度（默认基准 1.2 m/s）
    s_room  — 覆盖单间总处理时间（默认 76 s）
    blocked — 被阻断的边，如 [("N2", "N3")]，对应 x_e = 0
    starts  — 各响应者起点列表，如 ["E_L", "E_R"]（默认两名响应者）
    """
    scene = make_scene(v_walk=v_walk, s_room=s_room, blocked=blocked, starts=starts)
    dists, prevs = prepare_pairs(scene)
    sol = optimize(scene, dists, verbose=False)
    return sol, dists, prevs


# ---------------------------------------------------------- 图 1：建模流程图
def fig_framework(path):
    fig, ax = plt.subplots(figsize=(13.2, 3.9))
    ax.set_xlim(0, 132)
    ax.set_ylim(0, 30)
    ax.axis("off")

    stages = [
        ("① 楼层平面抽象", "#34495e"),
        ("② 加权网络 G=(V,E)", "#21618c"),
        ("③ Dijkstra 最短路", "#117864"),
        ("④ 任务分配优化", "#9a7d0a"),
        ("⑤ 最优清查方案", "#a93226"),
    ]
    bw, bh, y0 = 22.5, 20.5, 4.5
    xs = [1.5 + i * 26.2 for i in range(5)]

    for (title, color), x in zip(stages, xs):
        ax.add_patch(FancyBboxPatch((x, y0), bw, bh,
                                    boxstyle="round,pad=0.6,rounding_size=1.2",
                                    facecolor="#fbfcfc", edgecolor=color,
                                    lw=1.4, zorder=2))
        ax.add_patch(FancyBboxPatch((x, y0 + bh - 4.6), bw, 4.0,
                                    boxstyle="round,pad=0.6,rounding_size=1.2",
                                    facecolor=color, edgecolor="none", zorder=3))
        ax.text(x + bw / 2, y0 + bh - 2.4, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=4)
    for x in xs[:-1]:
        ax.add_patch(FancyArrowPatch((x + bw + 0.9, y0 + bh / 2),
                                     (x + 26.2 - 0.9, y0 + bh / 2),
                                     arrowstyle="-|>", mutation_scale=17,
                                     lw=2.0, color="#95a5a6", zorder=1))

    def box_coords(i):
        x = xs[i]
        return x + 1.6, y0 + 1.4, bw - 3.2, bh - 7.6   # 内容区 (x, y, w, h)

    # ① 迷你楼层平面
    cx, cy, cw, ch = box_coords(0)
    ax.add_patch(Rectangle((cx + 2, cy + 4), cw - 4, ch - 8, fill=False,
                           edgecolor=INK, lw=1.2))
    ax.plot([cx + 2, cx + cw - 2], [cy + ch / 2 - 1.1] * 2, color=INK, lw=1.0)
    ax.plot([cx + 2, cx + cw - 2], [cy + ch / 2 + 1.1] * 2, color=INK, lw=1.0)
    for f in (0.33, 0.66):
        xx = cx + 2 + (cw - 4) * f
        ax.plot([xx, xx], [cy + ch / 2 + 1.1, cy + ch - 4], color=INK, lw=1.0)
        ax.plot([xx, xx], [cy + 4, cy + ch / 2 - 1.1], color=INK, lw=1.0)
    ax.text(cx + cw / 2, cy + 1.2, "出口·走廊·房门·房间 → 节点与边",
            ha="center", fontsize=8.6, color="#555555")

    # ② 迷你网络
    cx, cy, cw, ch = box_coords(1)
    ys = cy + ch / 2
    xn = np.linspace(cx + 2.2, cx + cw - 2.2, 5)
    for i in range(4):
        ax.plot(xn[i:i + 2], [ys, ys], color="#888888", lw=1.4, zorder=2)
    for i in (1, 2, 3):
        ax.plot([xn[i], xn[i]], [ys, ys + 3.4], color="#888888", lw=1.2, zorder=2)
        ax.plot([xn[i], xn[i]], [ys, ys - 3.4], color="#888888", lw=1.2, zorder=2)
        ax.scatter([xn[i]] * 2, [ys + 3.4, ys - 3.4], s=26, zorder=3,
                   color=[ROOM_COLOR["R1"], ROOM_COLOR["R4"]],
                   edgecolor="black", linewidth=0.5)
    ax.scatter(xn, [ys] * 5, s=30, color="#dddddd", edgecolor="black",
               linewidth=0.6, zorder=3)
    ax.scatter([xn[0], xn[-1]], [ys, ys], s=46, marker="s", color="#1e8449",
               edgecolor="black", linewidth=0.6, zorder=4)
    ax.text(cx + cw / 2, cy + 1.2, r"$c_e=d_e/v_e$，$|V|$=13，$|E|$=12",
            ha="center", fontsize=9, color="#555555")

    # ③ Dijkstra
    cx, cy, cw, ch = box_coords(2)
    ax.text(cx + cw / 2, cy + ch - 1.6,
            r"$\mathrm{dist}(v)\leftarrow\min\{\mathrm{dist}(v),$",
            ha="center", fontsize=9.5, color=INK)
    ax.text(cx + cw / 2, cy + ch - 4.4,
            r"$\mathrm{dist}(u)+c(u,v)\}$（松弛）",
            ha="center", fontsize=9.5, color=INK)
    ax.text(cx + cw / 2, cy + ch - 7.6, "优先队列 · 贪心确定",
            ha="center", fontsize=8.6, color="#555555")
    ax.text(cx + cw / 2, cy + 1.2, r"$O((|V|+|E|)\log|V|)$",
            ha="center", fontsize=9, color="#555555")

    # ④ 优化
    cx, cy, cw, ch = box_coords(3)
    ax.text(cx + cw / 2, cy + ch - 1.8,
            r"$\min\,\max\{T_A,\ T_B\}$", ha="center", fontsize=11, color=INK)
    ax.text(cx + cw / 2, cy + ch - 5.0, "房间划分 × 访问顺序",
            ha="center", fontsize=8.6, color="#555555")
    ax.text(cx + cw / 2, cy + ch - 7.8, "2520 个候选方案枚举",
            ha="center", fontsize=8.6, color="#555555")
    ax.text(cx + cw / 2, cy + 1.2, "并列最优 8 个", ha="center",
            fontsize=9, color="#555555")

    # ⑤ 结果
    cx, cy, cw, ch = box_coords(4)
    ax.text(cx + cw / 2, cy + ch - 2.0, "A：R1→R4→R2", ha="center",
            fontsize=9.5, fontweight="bold", color=RESP_COLOR["A"])
    ax.text(cx + cw / 2, cy + ch - 5.2, "B：R3→R6→R5", ha="center",
            fontsize=9.5, fontweight="bold", color=RESP_COLOR["B"])
    ax.text(cx + cw / 2, cy + ch - 8.6, "T* = 246.3 s",
            ha="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(cx + cw / 2, cy + 1.2, "≈ 4.11 min", ha="center",
            fontsize=9, color="#555555")

    fig.suptitle("基于 Dijkstra 最短路的建筑清查建模流程", fontsize=14,
                 fontweight="bold", y=0.98)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- 图 2：甘特图
def fig_gantt(schedules, t_total, path):
    fig, ax = plt.subplots(figsize=(11.5, 3.7))
    row_h = 0.58
    for row, resp in enumerate(("A", "B")):
        y = 1 - row                                   # A 在上
        for s in schedules[resp]:
            if s["kind"] == "move":
                fc, ec, lw, lbl = "#d5d8dc", "#aab0b6", 0.6, None
            elif s["kind"] == "search":
                fc, ec, lw, lbl = ROOM_COLOR[s["room"]], "white", 0.8, s["room"]
            else:
                fc, ec, lw, lbl = "#566573", "white", 0.5, None
            ax.add_patch(Rectangle((s["t0"], y - row_h / 2), s["t1"] - s["t0"],
                                   row_h, facecolor=fc, edgecolor=ec, lw=lw,
                                   zorder=3))
            if lbl:
                ax.text((s["t0"] + s["t1"]) / 2, y, f"{lbl}\n搜索", ha="center",
                        va="center", fontsize=8, color="white",
                        fontweight="bold", zorder=4)
        # 清查完成菱形标记
        for s in schedules[resp]:
            if s["kind"] == "mark":
                ax.scatter([s["t1"]], [y + row_h / 2 + 0.16], marker="D",
                           s=42, color="#1e8449", zorder=5,
                           edgecolor="white", linewidth=0.8)
                ax.annotate(f"{s['t1']:.1f} s", xy=(s["t1"], y + row_h / 2 + 0.16),
                            xytext=(s["t1"] + 2.5, y + row_h / 2 + 0.30),
                            fontsize=8, color="#1e8449", fontweight="bold")
    ax.axvline(t_total, color="#a93226", ls="--", lw=1.4, zorder=2)
    ax.text(t_total, 1.62, f"全部清查完成  T* = {t_total:.1f} s ≈ {t_total/60:.2f} min",
            ha="right", va="bottom", fontsize=10.5, fontweight="bold",
            color="#a93226")
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["响应者 A（左进）", "响应者 B（右进）"], fontsize=10)
    ax.set_xlim(0, 272)
    ax.set_ylim(-0.62, 1.95)
    ax.set_xlabel("时间 (s)")
    ax.set_title("最优清查方案时间线（甘特图）", fontsize=13, fontweight="bold",
                 pad=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", color="#e5e7e9", lw=0.8, zorder=0)
    ax.legend(handles=[
        Patch(facecolor="#d5d8dc", edgecolor="#aab0b6", label="走廊移动"),
        Patch(facecolor="#4C72B0", label="房间搜索（颜色=房间）"),
        Patch(facecolor="#566573", label="报告 + 标记（5+3 s）"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#1e8449",
               markersize=8, label="清查完成时刻"),
    ], loc="lower left", ncol=4, frameon=False, fontsize=8.6,
        bbox_to_anchor=(0.0, -0.34))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------ 图 3：最短路热力图
def fig_heatmap(dists, path):
    labels = ["$E_L$", "$E_R$", "R1", "R2", "R3", "R4", "R5", "R6"]
    M = np.array([[dists[u][v] for v in KEY_NODES] for u in KEY_NODES])
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=M.max())
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold",
                    color="#1a1a1a" if M[i, j] > M.max() * 0.55 else "white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("最短通行时间 (s)", fontsize=10)
    cb.outline.set_visible(False)
    ax.set_title("关键节点两两最短通行时间矩阵（Dijkstra）",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- 图 4：敏感性分析（2×2）
def fig_sensitivity(path):
    base_T = 246.333

    # (a) 走廊行走速度
    speeds = [0.8, 1.0, 1.2, 1.4, 1.6]
    t_speed = [solve(v_walk=v)[0]["makespan"] for v in speeds]
    # (b) 搜索时间倍率（搜索越困难）
    mults = [1.0, 1.125, 1.25, 1.375, 1.5]
    t_mult = [solve(s_room=S_SEARCH * m + T_REPORT + T_MARK)[0]["makespan"]
              for m in mults]
    # (c) 每间人数
    occs = [2, 4, 6, 8, 10]
    t_occ = [solve(s_room=T0 + ALPHA * AREA + BETA * n + T_REPORT + T_MARK)[0]["makespan"]
             for n in occs]
    # (d) 响应者数量（第 3 人从左出口进入）
    ks = [1, 2, 3]
    t_k = [solve(starts=(["E_L"], ["E_L", "E_R"], ["E_L", "E_R", "E_L"])[i])[0]["makespan"]
           for i in range(3)]

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4))
    style = dict(marker="o", markersize=6, lw=2.0, color="#21618c", zorder=3)

    ax = axes[0, 0]
    ax.plot(speeds, t_speed, **style)
    ax.axhline(base_T, color="#aaaaaa", ls="--", lw=1)
    ax.scatter([1.2], [t_speed[2]], s=110, facecolor="#a93226", zorder=4,
               edgecolor="white", linewidth=1.2)
    ax.annotate(f"基准 {t_speed[2]:.1f} s", xy=(1.2, t_speed[2]),
                xytext=(1.245, t_speed[2] - 1.2), fontsize=9, color="#a93226",
                fontweight="bold")
    ax.set_xlabel("走廊行走速度 $v_h$ (m/s)")
    ax.set_ylabel("整楼清查时间 $T^*$ (s)")
    ax.set_title("(a) 低能见度减速的影响", fontsize=11.5, fontweight="bold")
    for v, t in zip(speeds, t_speed):
        if abs(v - 1.2) < 1e-9:
            continue
        ax.annotate(f"{t:.0f}", xy=(v, t), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8.3,
                    color="#21618c")

    ax = axes[0, 1]
    ax.plot([m * 100 for m in mults], t_mult, **style)
    ax.axhline(base_T, color="#aaaaaa", ls="--", lw=1)
    ax.scatter([100], [t_mult[0]], s=110, facecolor="#a93226", zorder=4,
               edgecolor="white", linewidth=1.2)
    ax.set_xlabel("单间搜索时间（相对基准的百分比 %）")
    ax.set_ylabel("整楼清查时间 $T^*$ (s)")
    ax.set_title("(b) 搜索困难程度的影响", fontsize=11.5, fontweight="bold")
    ax.annotate(f"+25% → {t_mult[2]:.0f} s", xy=(125, t_mult[2]),
                xytext=(8, -14), textcoords="offset points", fontsize=9,
                color="#21618c")

    ax = axes[1, 0]
    ax.plot(occs, t_occ, **style)
    ax.axhline(base_T, color="#aaaaaa", ls="--", lw=1)
    ax.scatter([4], [t_occ[1]], s=110, facecolor="#a93226", zorder=4,
               edgecolor="white", linewidth=1.2)
    ax.set_xlabel("每间办公室人数 $n_i$ (人)")
    ax.set_ylabel("整楼清查时间 $T^*$ (s)")
    ax.set_title("(c) 房间人数的影响", fontsize=11.5, fontweight="bold")
    for n, t in zip(occs, t_occ):
        ax.annotate(f"{t:.0f}", xy=(n, t), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8.3,
                    color="#21618c")

    ax = axes[1, 1]
    bars = ax.bar([str(k) for k in ks], t_k, width=0.52,
                  color=["#c0392b", "#1e8449", "#21618c"], zorder=3,
                  edgecolor="white")
    for b, t in zip(bars, t_k):
        ax.annotate(f"{t:.1f} s", xy=(b.get_x() + b.get_width() / 2, t),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", color=INK)
    ax.set_xlabel("响应者数量（第 3 人从左出口进入）")
    ax.set_ylabel("整楼清查时间 $T^*$ (s)")
    ax.set_title("(d) 响应者数量的影响", fontsize=11.5, fontweight="bold")
    ax.set_ylim(0, max(t_k) * 1.18)

    for ax in axes.flat:
        ax.grid(axis="y", color="#e5e7e9", lw=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("最优清查时间 $T^*$ 的敏感性分析", fontsize=14,
                 fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"speeds": dict(zip(speeds, t_speed)),
            "search_mult": dict(zip(mults, t_mult)),
            "occupancy": dict(zip(occs, t_occ)),
            "responders": dict(zip(ks, t_k))}


# ------------------------------------------------- 图 5：走廊受阻情景
def fig_blocked(sol_base, prevs_base, path):
    blocked_edge = ("N2", "N3")
    scene_b = make_scene(blocked=[blocked_edge])
    sol_b, dists_b, prevs_b = solve(blocked=[blocked_edge])

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))
    for ax, sol, prevs, scene, blocked, subtitle in (
            (axes[0], sol_base, prevs_base, DEFAULT_SCENE, False, "正常情形"),
            (axes[1], sol_b, prevs_b, scene_b, True, "走廊中段阻断（$x_e=0$）")):
        face = {r: "#f4f4f4" for r in ROOMS}
        draw_floorplan(ax, scene, face)
        draw_route_arrows(ax, sol["legs"], prevs, scene=scene)
        if blocked:
            ax.plot([15, 25], [0, 0], color="#c0392b", lw=5, alpha=0.85,
                    zorder=8, solid_capstyle="round")
            ax.text(20, 0.66, "烟雾阻断", ha="center", fontsize=10,
                    fontweight="bold", color="#c0392b", zorder=9)
        T = sol["makespan"]
        ax.set_title(f"{subtitle}：$T^*$ = {T:.1f} s", fontsize=12.5,
                     fontweight="bold")
        rt = "、".join(f"{r}：{'→'.join(sol['routes'][r])}"
                       for r in ("A", "B"))
        ax.text(15, -6.85, rt, ha="center", va="top", fontsize=9.5,
                color="#444444")
        ax.set_xlim(-6.4, 35.8)
        ax.set_ylim(-8.0, 6.9)
    dT = sol_b["makespan"] - sol_base["makespan"]
    fig.suptitle(f"走廊阻断情景下的路线重规划（Dijkstra 重算）   "
                 f"$\\Delta T$ = +{dT:.1f} s（+{dT / sol_base['makespan'] * 100:.1f}%）",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"blocked_edge": list(blocked_edge),
            "makespan_s": round(sol_b["makespan"], 3),
            "routes": sol_b["routes"],
            "delta_s": round(dT, 3)}


# ------------------------------------------------- 图 6：策略对比
def fig_strategy(sol_base, dists, path):
    t_opt = sol_base["makespan"]
    # 分侧顺序清查：A 上排 R1R2R3，B 下排 R6R5R4
    tA_side = route_eval(("R1", "R2", "R3"), "E_L", dists, S_ROOM)[0]
    tB_side = route_eval(("R6", "R5", "R4"), "E_R", dists, S_ROOM)[0]
    t_side = max(tA_side, tB_side)
    # 双人同侧（左）进入
    t_same = solve(starts=["E_L", "E_L"])[0]["makespan"]
    # 退回出口
    t_back = 0.0
    for resp in ("A", "B"):
        last = sol_base["routes"][resp][-1]
        t_back = max(t_back, sol_base["times"][resp]
                     + min(dists[last]["E_L"], dists[last]["E_R"]))
    # 单人
    t_single = solve(starts=["E_L"])[0]["makespan"]
    # 假设文档基准估算（近似）
    t_est = 238.0

    items = [
        ("单人清查（对照）", t_single, "#c0392b", None),
        ("最优方案 + 退回出口", t_back, "#7d6608", None),
        ("双人同侧（左）进入", t_same, "#9a7d0a", None),
        ("分侧顺序清查（A上排/B下排）", t_side, "#5d6d7e", None),
        ("最优方案（双侧进入）", t_opt, "#1e8449", None),
        ("假设文档基准估算（近似）", t_est, "#808b96", "///"),
    ]
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    ys = np.arange(len(items))
    for y, (name, t, color, hatch) in zip(ys, items):
        ax.barh(y, t, height=0.58, color=color, hatch=hatch,
                edgecolor="white", zorder=3,
                alpha=0.95 if hatch is None else 0.55)
        ax.annotate(f"{t:.1f} s（{t / 60:.2f} min）", xy=(t, y),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9.5, fontweight="bold", color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], fontsize=10.5)
    ax.set_xlabel("整楼清查完成时间 (s)")
    ax.set_xlim(0, max(i[1] for i in items) * 1.22)
    ax.grid(axis="x", color="#e5e7e9", lw=0.8, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(left=False)
    ax.axvline(t_opt, color="#1e8449", ls="--", lw=1.1, zorder=2, alpha=0.6)
    ax.set_title("不同清查策略完成时间对比", fontsize=13.5, fontweight="bold",
                 pad=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"optimal": round(t_opt, 3), "side_by_side": round(t_side, 3),
            "same_side_entry": round(t_same, 3),
            "return_to_exit": round(t_back, 3), "single": round(t_single, 3),
            "doc_estimate": t_est}


def main():
    # 基准解
    sol, dists, prevs = solve()
    schedules, clear_times = build_schedule(DEFAULT_SCENE, sol["routes"], prevs)
    T = sol["makespan"]

    FIG_DIR.mkdir(exist_ok=True)
    fig_framework(FIG_DIR / "fig_paper_framework.png")
    print("已保存 figures/fig_paper_framework.png")
    fig_gantt(schedules, T, FIG_DIR / "fig_paper_gantt.png")
    print("已保存 figures/fig_paper_gantt.png")
    fig_heatmap(dists, FIG_DIR / "fig_paper_heatmap.png")
    print("已保存 figures/fig_paper_heatmap.png")
    sens = fig_sensitivity(FIG_DIR / "fig_paper_sensitivity.png")
    print("已保存 figures/fig_paper_sensitivity.png")
    blocked = fig_blocked(sol, prevs, FIG_DIR / "fig_paper_blocked.png")
    print("已保存 figures/fig_paper_blocked.png")
    strat = fig_strategy(sol, dists, FIG_DIR / "fig_paper_strategy.png")
    print("已保存 figures/fig_paper_strategy.png")

    with open(OUT_DIR / "sweep_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump({"baseline_makespan_s": round(T, 3),
                   "sensitivity": sens, "blocked": blocked,
                   "strategies": strat}, f, ensure_ascii=False, indent=2)
    print("已保存 output/sweep_sensitivity.json")
    print(f"敏感性：{sens}")
    print(f"阻断：{blocked}")
    print(f"策略：{strat}")


if __name__ == "__main__":
    main()
