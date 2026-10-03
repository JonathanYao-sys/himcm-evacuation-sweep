# -*- coding: utf-8 -*-
"""绘图：建筑网络图、最优路线静态图、最优解清查过程动画 GIF。

所有图中文字（|V|、|E|、v_e、s_i*、房间面积/人数、走廊长度等）都直接
由 `Scene`/`Config` 生成，不会出现"改了参数但图上还写着旧数字"的情况。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, Rectangle

from sweep_building import DEFAULT_SCENE, Scene
from sweep_dijkstra import shortest_path

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
            "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
_RESP_PALETTE = ["#c0392b", "#1f3a93", "#1e8449", "#8e44ad", "#d35400", "#16a085"]


def room_colors(scene: Scene):
    return {r: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(scene.rooms)}


def responder_colors(scene: Scene):
    return {r: _RESP_PALETTE[i % len(_RESP_PALETTE)] for i, r in enumerate(scene.starts)}


# 基准场景下的颜色字典（供论文配图等旧接口直接取色）
ROOM_COLOR = room_colors(DEFAULT_SCENE)
RESP_COLOR = responder_colors(DEFAULT_SCENE)


# ---------------------------------------------------------------- 平面图
def _wall_with_gaps(ax, x0, x1, y, gaps, half, lw=2.4):
    """在 y 高度画一条水平墙，gaps 处留门洞。"""
    pts = [x0]
    for gx in gaps:
        pts += [gx - half, gx + half]
    pts.append(x1)
    for i in range(0, len(pts), 2):
        ax.plot([pts[i], pts[i + 1]], [y, y], color="black", lw=lw,
                solid_capstyle="butt", zorder=3)


def draw_floorplan(ax, scene: Scene = None, facecolors=None):
    """绘制楼层平面（墙体、门洞、出口、房间标签），返回房间矩形对象。

    兼容两种调用：draw_floorplan(ax, scene, facecolors) 与旧式
    draw_floorplan(ax, facecolors)（后者按基准场景绘制）。
    """
    if isinstance(scene, dict) and facecolors is None:
        scene, facecolors = DEFAULT_SCENE, scene
    scene = scene or DEFAULT_SCENE
    cfg = scene.cfg
    fc = facecolors or {}
    hall_y, w, L = scene.hall_y, cfg.room_width, cfg.hall_length
    half = min(0.75, w * 0.3)
    rects = {}
    for r, (x, y, ww, h) in scene.room_rects.items():
        rect = Rectangle((x, y), ww, h, facecolor=fc.get(r, "#f0f0f0"),
                         edgecolor="none", zorder=1)
        ax.add_patch(rect)
        rects[r] = rect
        up = y > 0
        ax.text(x + ww / 2, y + h - 0.11 * h if up else y + 0.11 * h, r,
                ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)
        ax.text(x + ww / 2, y + h - 0.25 * h if up else y + 0.25 * h,
                f"{cfg.area:g} m² · {cfg.occupancy:g} 人", ha="center", va="center",
                fontsize=8, color="#555555", zorder=4)
    y_top = scene.y_top
    ax.plot([0, L], [y_top, y_top], color="black", lw=2.6, solid_capstyle="butt", zorder=3)
    ax.plot([0, L], [-y_top, -y_top], color="black", lw=2.6, solid_capstyle="butt", zorder=3)
    for x in [i * w for i in range(cfg.n_rooms_per_side + 1)]:
        ax.plot([x, x], [hall_y, y_top], color="black", lw=2.4, solid_capstyle="butt", zorder=3)
        ax.plot([x, x], [-y_top, -hall_y], color="black", lw=2.4, solid_capstyle="butt", zorder=3)
    gap_x = sorted(set(scene.door_x.values()))
    _wall_with_gaps(ax, -cfg.exit_length, L + cfg.exit_length, hall_y, gap_x, half)
    _wall_with_gaps(ax, -cfg.exit_length, L + cfg.exit_length, -hall_y, gap_x, half)
    stub = min(0.7, hall_y * 0.7)
    for x in (-cfg.exit_length, L + cfg.exit_length):
        ax.plot([x, x], [stub, hall_y], color="black", lw=2.4, solid_capstyle="butt", zorder=3)
        ax.plot([x, x], [-hall_y, -stub], color="black", lw=2.4, solid_capstyle="butt", zorder=3)
    theta = np.linspace(0, np.pi / 2, 20)
    for r, dx in scene.door_x.items():
        s = 1 if scene.room_rects[r][1] > 0 else -1
        ax.plot(dx + half * np.cos(theta), s * hall_y + s * half * np.sin(theta),
                color="#999999", lw=0.9, zorder=3)
    for x, outward in ((-cfg.exit_length, -1), (L + cfg.exit_length, 1)):
        ax.annotate("", xy=(x + outward * 1.2, 0), xytext=(x + outward * 0.25, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#1e8449", lw=2.4), zorder=4)
        ax.text(x + outward * 2.1, 0, "EXIT", ha="center", va="center",
                fontsize=12, fontweight="bold", color="#1e8449", zorder=4)
    ax.text(L / 2, -0.55 * hall_y, f"走廊 {L:g} m", ha="center", va="center",
            fontsize=9, color="#888888", zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    return rects


def draw_route_arrows(ax, legs, prevs, label_steps=True, scene: Scene = None):
    """在楼层平面上绘制响应者路线箭头。

    legs: {响应者: [(起点, 房间, 段耗时), ...]}；prevs 为各节点 Dijkstra 前驱表。
    走廊水平段按响应者序号做纵向偏移，避免多人路线重叠。
    """
    scene = scene or DEFAULT_SCENE
    resp_col = responder_colors(scene)
    responders = [r for r in legs if legs[r]]
    for i, resp in enumerate(responders):
        color = resp_col[resp]
        dy = 0.0 if len(responders) < 2 else 0.14 - 0.28 * i / (len(responders) - 1)
        for step, (u, room, _d) in enumerate(legs[resp], 1):
            nodes = shortest_path(prevs[u], u, room)
            for a, b in zip(nodes[:-1], nodes[1:]):
                xa, ya = scene.pos[a]
                xb, yb = scene.pos[b]
                if ya == yb == 0:                # 走廊水平段加纵向偏移
                    ya = yb = dy
                ax.add_patch(FancyArrowPatch(
                    (xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=15,
                    lw=2.2, color=color, alpha=0.95, zorder=6, shrinkA=4, shrinkB=4))
            if label_steps:
                ax.text(scene.pos[room][0] + 0.55, scene.pos[room][1], f"{resp}{step}",
                        fontsize=9.5, fontweight="bold", color=color, zorder=7)


# ---------------------------------------------------------------- 图 1：网络图
def fig_network(scene: Scene, path):
    cfg = scene.cfg
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for u, v, d in scene.edges:
        x = [scene.pos[u][0], scene.pos[v][0]]
        y = [scene.pos[u][1], scene.pos[v][1]]
        ax.plot(x, y, color="#999999", lw=2, zorder=1)
        c = d / cfg.v_hall
        mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
        label = f"{d:g} m → {c:.2f} s"
        if x[0] == x[1]:        # 房门竖直边：标签放右侧，避免压住节点名
            ax.text(mx + 0.35, my, label, ha="left", va="center", fontsize=8.3,
                    color="#333333",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.4),
                    zorder=6)
        else:
            ax.text(mx, my + 0.16, label, ha="center", va="bottom", fontsize=8.3,
                    color="#333333",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.4),
                    zorder=6)
    rcol = room_colors(scene)
    for node, (x, y) in scene.pos.items():
        if node in rcol:
            ax.scatter([x], [y], s=560, color=rcol[node], zorder=4,
                       edgecolor="black", linewidth=1.2)
            ax.text(x, y, node, ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white", zorder=5)
        elif node.startswith("E_"):
            ax.scatter([x], [y], s=420, marker="s", color="#1e8449", zorder=4,
                       edgecolor="black", linewidth=1.2)
            ax.text(x, y - 0.62, "出口 " + node, ha="center", va="top", fontsize=9,
                    color="#1e8449", fontweight="bold", zorder=5)
        else:
            ax.scatter([x], [y], s=200, color="#dddddd", zorder=4,
                       edgecolor="black", linewidth=1.0)
            ax.text(x, y - 0.62, node, ha="center", va="top", fontsize=9,
                    color="#333333", zorder=5)
    blocked = "" if not cfg.blocked_edges else \
        "　阻断边：" + "、".join(f"{u}–{v}" for u, v in cfg.blocked_edges)
    ax.set_title(f"基础场景建筑网络  $G=(V,E)$，$|V|={len(scene.pos)}, "
                 f"|E|={len(scene.edges)}$，边权 $c_e = d_e / v_e$，"
                 f"$v_e = {cfg.v_hall:g}$ m/s{blocked}", fontsize=12)
    x0, x1 = scene.x_bounds
    ax.text((x0 + x1) / 2, -scene.y_top + 1.6,
            "房间处理时间 $s_i^{*} = t_0 + \\alpha a_i + \\beta n_i"
            f" + t_{{rep}} + t_{{mark}}$ = {cfg.t_check:g} + {cfg.alpha:g}×{cfg.area:g}"
            f" + {cfg.beta:g}×{cfg.occupancy:g} + {cfg.t_report:g} + {cfg.t_mark:g}"
            f" = {scene.s_room:g} s（t_rep 报告，t_mark 标记）",
            ha="center", fontsize=10, color="#444444")
    ax.set_xlim(x0 - 2.5, x1 + 2.5)
    rmy = max(scene.pos[r][1] for r in scene.rooms)
    ax.set_ylim(-rmy - 2.5, rmy + 2.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------- 图 2：最优路线静态图
def fig_optimal_routes(scene: Scene, solution, prevs, clear_times, path):
    cfg = scene.cfg
    responders = list(scene.starts)
    rcol, resp_col = room_colors(scene), responder_colors(scene)
    fig = plt.figure(figsize=(13.6, 6.8))
    ax = fig.add_axes([0.015, 0.03, 0.66, 0.9])
    face = {r: to_rgba(rcol[r], 0.30) for r in scene.rooms}
    draw_floorplan(ax, scene, face)
    offsets = np.linspace(0.14, -0.14, len(responders)) if len(responders) > 1 else [0.0]
    for i, resp in enumerate(responders):
        color = resp_col[resp]
        dy = offsets[i]
        for step, (u, room, _d) in enumerate(solution["legs"][resp], 1):
            nodes = shortest_path(prevs[u], u, room)
            for a, b in zip(nodes[:-1], nodes[1:]):
                xa, ya = scene.pos[a]
                xb, yb = scene.pos[b]
                if ya == yb == 0:                # 走廊水平段加纵向偏移
                    ya = yb = dy
                ax.add_patch(FancyArrowPatch(
                    (xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=15,
                    lw=2.2, color=color, alpha=0.95, zorder=6, shrinkA=4, shrinkB=4))
            ax.text(scene.pos[room][0] + 0.55, scene.pos[room][1], f"{resp}{step}",
                    fontsize=9.5, fontweight="bold", color=color, zorder=7)
    for r in scene.rooms:
        ax.text(scene.pos[r][0], scene.pos[r][1] + (1.15 if scene.pos[r][1] > 0 else -1.15),
                f"✓ {clear_times[r]:.1f} s", ha="center", fontsize=9,
                color="#1e8449", fontweight="bold", zorder=7)
    x0, x1 = scene.x_bounds
    ax.set_xlim(x0 - 3.6, x1 + 3.6)
    ax.set_ylim(-scene.y_top - 0.9, scene.y_top + 0.9)

    T = solution["makespan"]
    lines = ["最优任务分配（Held–Karp + 枚举，Dijkstra 精确路长）", ""]
    for resp in responders:
        route = solution["routes"][resp]
        travel_m = solution["travel"][resp] * cfg.v_hall
        if not route:
            lines += [f"响应者 {resp}：待命（无房间）", ""]
            continue
        lines.append(f"响应者 {resp}（{scene.starts[resp]} 进入）")
        lines.append("  路线：" + " → ".join([scene.starts[resp]] + route))
        lines.append(f"  移动 {travel_m:.1f} m = {solution['travel'][resp]:.2f} s"
                     f" ＋ 搜索 {len(route)}×{scene.s_room:g} s")
        lines.append(f"  完成时间 T_{resp} = {solution['times'][resp]:.1f} s")
        lines.append("")
    lines.append(f"整楼清查完成时间  T = max_r T_r = {T:.1f} s ≈ {T/60:.2f} min")
    lines.append(f"（候选方案 {solution['evaluated']} 个，并列最优 {solution['n_ties']} 个）")
    fig.text(0.685, 0.90, "\n".join(lines), fontsize=11.5, va="top",
             family="Arial Unicode MS",
             bbox=dict(facecolor="#f7f7f7", edgecolor="#cccccc", boxstyle="round,pad=0.7"))
    fig.suptitle(f"基础场景最优清查方案（{len(responders)} 名响应者）",
                 fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ 图 3：动画 GIF
def _responder_state(segs, t, pos):
    """时刻 t 响应者的位置与状态文本。"""
    for s in segs:
        if s["t0"] <= t < s["t1"]:
            if s["kind"] == "move":
                p = (t - s["t0"]) / (s["t1"] - s["t0"])
                xa, ya = pos[s["u"]]
                xb, yb = pos[s["v"]]
                return (xa + p * (xb - xa), ya + p * (yb - ya)), \
                       f"移动 {s['u']}→{s['v']}", s
            return pos[s["room"]], None, s
    if not segs:
        return pos[list(pos)[0]], "待命", None
    last = segs[-1]
    return pos[last.get("room") or last["v"]], "完成，待命", None


def _room_phase_times(schedules):
    """每间房：(响应者, 搜索开始, 搜索结束, 标记结束)。"""
    info = {}
    for resp, segs in schedules.items():
        search_t0 = search_t1 = None
        for s in segs:
            if s["kind"] == "search":
                search_t0, search_t1 = s["t0"], s["t1"]
            elif s["kind"] == "mark":
                info[s["room"]] = (resp, search_t0, search_t1, s["t1"])
    return info


def make_animation(scene: Scene, solution, schedules, t_total, path, dt=0.5, fps=15):
    responders = list(scene.starts)
    rcol, resp_col = room_colors(scene), responder_colors(scene)
    room_info = _room_phase_times(schedules)
    epilogue = 4.0
    frames = np.arange(0, t_total + epilogue, dt)
    n_resp = len(responders)

    fig_h = 6.6 if n_resp <= 2 else 5.4 + 0.6 * n_resp
    fig = plt.figure(figsize=(11.0, fig_h))
    ax = fig.add_axes([0.02, 0.16, 0.96, 0.74])
    gx = fig.add_axes([0.06, 0.055, 0.90, 0.075])
    trails = {r: [] for r in responders}
    colors = room_colors(scene)

    def draw(t):
        ax.clear()
        gx.clear()
        face, status, n_cleared = {}, {}, 0
        for r in scene.rooms:
            resp, s0, s1, m1 = room_info[r]
            if t < s0:
                face[r], status[r] = "#eeeeee", ""
            elif t < s1:
                face[r] = to_rgba(colors[r], 0.40)
                status[r] = f"{resp} 搜索 {100 * (t - s0) / (s1 - s0):.0f}%"
            elif t < m1:
                face[r] = to_rgba(colors[r], 0.65)
                status[r] = f"{resp} 报告/标记"
            else:
                face[r] = to_rgba(colors[r], 0.90)
                status[r] = f"✓ {m1:.1f} s"
                n_cleared += 1
        draw_floorplan(ax, scene, face)
        for r in scene.rooms:
            if status[r]:
                ax.text(scene.pos[r][0], scene.pos[r][1] + (1.15 if scene.pos[r][1] > 0 else -1.15),
                        status[r], ha="center", fontsize=9.5, fontweight="bold",
                        color="#1e8449" if status[r].startswith("✓") else "#333333", zorder=7)
        for resp in responders:
            (x, y), txt, _ = _responder_state(schedules[resp], t, scene.pos)
            trails[resp].append((x, y))
            if len(trails[resp]) > 1:
                xs, ys = zip(*trails[resp])
                ax.plot(xs, ys, color=resp_col[resp], lw=1.2, alpha=0.35, zorder=5)
            ax.scatter([x], [y], s=330, color=resp_col[resp], zorder=8,
                       edgecolor="white", linewidth=1.6)
            ax.text(x, y, resp, ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white", zorder=9)
            if txt:
                ax.annotate(txt, xy=(x, y), xytext=(x + 0.5, y), fontsize=8.5,
                            color=resp_col[resp], zorder=9, ha="left", va="center")
        x0, x1 = scene.x_bounds
        ax.set_xlim(x0 - 4.4, x1 + 3.8)
        ax.set_ylim(-scene.y_top - 0.9, scene.y_top + 0.9)
        ax.set_title(f"最优方案清查过程    T = {min(t, t_total):6.1f} s / {t_total:.1f} s"
                     f"    已清查 {n_cleared}/{len(scene.rooms)} 间",
                     fontsize=13, fontweight="bold", pad=10)
        # 甘特进度条
        for row, resp in enumerate(reversed(responders)):
            for s in schedules[resp]:
                if s["kind"] == "move":
                    color = "#c8c8c8"
                elif s["kind"] == "search":
                    color = colors[s["room"]]
                else:
                    color = "#666666"
                gx.add_patch(Rectangle((s["t0"], row + 0.12),
                                       min(s["t1"], t) - s["t0"] if s["t0"] < t else 0,
                                       0.76, facecolor=color, edgecolor="none"))
                if s["kind"] == "search":
                    gx.text((s["t0"] + s["t1"]) / 2, row + 0.5, s["room"],
                            ha="center", va="center", fontsize=7.5,
                            color="white", fontweight="bold")
        gx.axvline(min(t, t_total), color="red", lw=1.4)
        gx.set_xlim(0, t_total + epilogue)
        gx.set_ylim(-0.15, n_resp - 0.1)
        gx.set_yticks([i + 0.5 for i in range(n_resp)])
        gx.set_yticklabels(list(reversed(responders)), fontsize=10, fontweight="bold")
        gx.set_xlabel("时间 (s)　灰=移动，彩色=搜索，深灰=报告+标记", fontsize=9)
        gx.tick_params(labelsize=8)
        for spine in ("top", "right"):
            gx.spines[spine].set_visible(False)

    ani = FuncAnimation(fig, draw, frames=frames)
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return len(frames)
