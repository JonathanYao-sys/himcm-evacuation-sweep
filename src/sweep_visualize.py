# -*- coding: utf-8 -*-
"""绘图：建筑网络图、最优路线静态图、最优解清查过程动画 GIF。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba

from sweep_building import POS, EDGES, ROOMS, V_WALK, S_SEARCH, T_REPORT, T_MARK, S_ROOM
from sweep_dijkstra import shortest_path

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

RESP_COLOR = {"A": "#c0392b", "B": "#1f3a93"}
ROOM_COLOR = {"R1": "#4C72B0", "R2": "#DD8452", "R3": "#55A868",
              "R4": "#C44E52", "R5": "#8172B3", "R6": "#937860"}
# 房间矩形 (x0, y0, w, h)：走廊两侧各三间，每间宽 10 m
ROOM_RECT = {"R1": (0, 1.0, 10, 5.0), "R2": (10, 1.0, 10, 5.0), "R3": (20, 1.0, 10, 5.0),
             "R4": (0, -6.0, 10, 5.0), "R5": (10, -6.0, 10, 5.0), "R6": (20, -6.0, 10, 5.0)}
HALL_Y = 1.0           # 走廊半宽（走廊宽 2 m）
DOOR_X = {r: POS[r][0] for r in ROOMS}
DOOR_HALF = 0.75       # 门洞半宽


def _wall_with_gaps(ax, x0, x1, y, gaps, lw=2.4):
    """在 y 高度画一条水平墙，gaps 处留门洞。"""
    pts = [x0]
    for gx in gaps:
        pts += [gx - DOOR_HALF, gx + DOOR_HALF]
    pts.append(x1)
    for i in range(0, len(pts), 2):
        ax.plot([pts[i], pts[i + 1]], [y, y], color="black", lw=lw,
                solid_capstyle="butt", zorder=3)


def draw_floorplan(ax, facecolors=None):
    """绘制楼层平面（墙体、门洞、出口、房间标签），返回各房间矩形对象。"""
    fc = facecolors or {}
    rects = {}
    for r, (x, y, w, h) in ROOM_RECT.items():
        rect = Rectangle((x, y), w, h, facecolor=fc.get(r, "#f0f0f0"),
                         edgecolor="none", zorder=1)
        ax.add_patch(rect)
        rects[r] = rect
        s = 1 if y > 0 else -1
        ax.text(x + w / 2, y + h - 0.55 if s > 0 else y + 0.55, r,
                ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h - 1.25 if s > 0 else y + 1.25,
                "20 m² · 4 人", ha="center", va="center", fontsize=8,
                color="#555555", zorder=4)
    # 外墙（上下）
    ax.plot([0, 30], [6, 6], color="black", lw=2.6, solid_capstyle="butt", zorder=3)
    ax.plot([0, 30], [-6, -6], color="black", lw=2.6, solid_capstyle="butt", zorder=3)
    # 房间分隔墙
    for x in (0, 10, 20, 30):
        ax.plot([x, x], [HALL_Y, 6], color="black", lw=2.4, solid_capstyle="butt", zorder=3)
        ax.plot([x, x], [-6, -HALL_Y], color="black", lw=2.4, solid_capstyle="butt", zorder=3)
    # 走廊墙（带门洞）
    _wall_with_gaps(ax, -2, 32, HALL_Y, sorted(DOOR_X.values()))
    _wall_with_gaps(ax, -2, 32, -HALL_Y, sorted(DOOR_X.values()))
    # 出口走廊端墙（留出口门洞）
    for x in (-2, 32):
        ax.plot([x, x], [0.7, HALL_Y], color="black", lw=2.4, solid_capstyle="butt", zorder=3)
        ax.plot([x, x], [-HALL_Y, -0.7], color="black", lw=2.4, solid_capstyle="butt", zorder=3)
    # 门扇弧线
    theta = np.linspace(0, np.pi / 2, 20)
    for r in ROOMS:
        dx = DOOR_X[r]
        s = 1 if ROOM_RECT[r][1] > 0 else -1
        ax.plot(dx + DOOR_HALF * np.cos(theta),
                s * HALL_Y + s * DOOR_HALF * np.sin(theta),
                color="#999999", lw=0.9, zorder=3)
    # 出口标识
    for x, tip, txt in ((-2, -3.2, -4.1), (32, 33.2, 34.1)):
        ax.annotate("", xy=(tip, 0), xytext=(x + (-0.25 if x < 0 else 0.25), 0),
                    arrowprops=dict(arrowstyle="-|>", color="#1e8449", lw=2.4), zorder=4)
        ax.text(txt, 0, "EXIT", ha="center", va="center",
                fontsize=12, fontweight="bold", color="#1e8449", zorder=4)
    ax.text(10, -0.55, "走廊 30 m", ha="center", va="center", fontsize=9,
            color="#888888", zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    return rects


# ---------------------------------------------------------------- 图 1：网络图
def fig_network(path):
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for u, v, d in EDGES:
        x = [POS[u][0], POS[v][0]]
        y = [POS[u][1], POS[v][1]]
        ax.plot(x, y, color="#999999", lw=2, zorder=1)
        c = d / V_WALK
        mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
        if x[0] == x[1]:        # 房门竖直边：标签放右侧，避免压住节点名
            ax.text(mx + 0.35, my, f"{d:g} m → {c:.2f} s", ha="left",
                    va="center", fontsize=8.3, color="#333333",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                              pad=0.4), zorder=6)
        else:
            ax.text(mx, my + 0.16, f"{d:g} m → {c:.2f} s", ha="center",
                    va="bottom", fontsize=8.3, color="#333333",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                              pad=0.4), zorder=6)
    for n, (x, y) in POS.items():
        if n in ROOMS:
            ax.scatter([x], [y], s=560, color=ROOM_COLOR[n], zorder=4,
                       edgecolor="black", linewidth=1.2)
            ax.text(x, y, n, ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white", zorder=5)
        elif n.startswith("E_"):
            ax.scatter([x], [y], s=420, marker="s", color="#1e8449", zorder=4,
                       edgecolor="black", linewidth=1.2)
            ax.text(x, y - 0.62, "出口 " + n, ha="center", va="top", fontsize=9,
                    color="#1e8449", fontweight="bold", zorder=5)
        else:
            ax.scatter([x], [y], s=200, color="#dddddd", zorder=4,
                       edgecolor="black", linewidth=1.0)
            ax.text(x, y - 0.62, n, ha="center", va="top", fontsize=9,
                    color="#333333", zorder=5)
    ax.set_title("基础场景建筑网络  $G=(V,E)$，$|V|=13, |E|=12$，"
                 "边权 $c_e = d_e / v_e$，$v_e = 1.2$ m/s", fontsize=12)
    ax.text(15, -4.1,
            "房间处理时间 $s_i^{*} = t_0 + \\alpha a_i + \\beta n_i"
            " + t_{rep} + t_{mark}$ = 20 + 0.8×20 + 8×4 + 5 + 3 = 76 s"
            "（t_rep 报告，t_mark 标记）",
            ha="center", fontsize=10, color="#444444")
    ax.set_xlim(-4.5, 34.5)
    ax.set_ylim(-4.8, 4.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_route_arrows(ax, legs, prevs, label_steps=True):
    """在楼层平面上绘制响应者路线箭头。legs: {resp: [(u, room, 耗时), ...]}"""
    for resp, legs_r in legs.items():
        color = RESP_COLOR[resp]
        dy = 0.14 if resp == "A" else -0.14      # 同一走廊段上两人箭头错开
        for step, (u, room, _) in enumerate(legs_r, 1):
            nodes = shortest_path(prevs[u], u, room)
            for a, b in zip(nodes[:-1], nodes[1:]):
                xa, ya = POS[a]
                xb, yb = POS[b]
                if ya == yb == 0:                # 走廊水平段加纵向偏移
                    ya = yb = dy
                ax.add_patch(FancyArrowPatch(
                    (xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=15,
                    lw=2.2, color=color, alpha=0.95, zorder=6,
                    shrinkA=4, shrinkB=4))
            if label_steps:
                ax.text(POS[room][0] + 0.55, POS[room][1], f"{resp}{step}",
                        fontsize=9.5, fontweight="bold", color=color, zorder=7)


# ------------------------------------------------------- 图 2：最优路线静态图
def fig_optimal_routes(solution, prevs, clear_times, path):
    fig = plt.figure(figsize=(13.6, 6.8))
    ax = fig.add_axes([0.015, 0.03, 0.66, 0.9])
    face = {r: to_rgba(ROOM_COLOR[r], 0.30) for r in ROOMS}
    draw_floorplan(ax, face)
    draw_route_arrows(ax, solution["legs"], prevs)
    # 房间内标注清查完成时刻
    for r in ROOMS:
        ax.text(POS[r][0], POS[r][1] + (1.15 if POS[r][1] > 0 else -1.15),
                f"✓ {clear_times[r]:.1f} s", ha="center", fontsize=9,
                color="#1e8449", fontweight="bold", zorder=7)
    ax.set_xlim(-5.6, 35.6)
    ax.set_ylim(-6.9, 6.9)

    # 右侧文字面板
    T = solution["makespan"]
    lines = ["最优任务分配（全局枚举 + Dijkstra 精确路长）", ""]
    for resp in ("A", "B"):
        route = solution["routes"][resp]
        travel_m = solution["travel"][resp] * V_WALK
        lines.append(f"响应者 {resp}（{('左出口 E_L' if resp=='A' else '右出口 E_R')} 进入）")
        lines.append("  路线：" + (" → ".join([solution['starts'][resp]] + route)))
        lines.append(f"  移动 {travel_m:.1f} m = {solution['travel'][resp]:.2f} s"
                     f" ＋ 搜索 {len(route)}×76 s")
        lines.append(f"  完成时间 T_{resp} = {solution['times'][resp]:.1f} s")
        lines.append("")
    lines.append(f"整楼清查完成时间  T = max(T_A, T_B) = {T:.1f} s ≈ {T/60:.2f} min")
    lines.append(f"（候选方案 {solution['evaluated']} 个，并列最优 {solution['n_ties']} 个）")
    fig.text(0.685, 0.90, "\n".join(lines), fontsize=11.5, va="top",
             family="Arial Unicode MS",
             bbox=dict(facecolor="#f7f7f7", edgecolor="#cccccc", boxstyle="round,pad=0.7"))
    fig.suptitle("基础场景最优清查方案（两名响应者）", fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ 图 3：动画 GIF
def _responder_state(segs, t):
    """时刻 t 响应者的位置与状态文本。"""
    for s in segs:
        if s["t0"] <= t < s["t1"]:
            if s["kind"] == "move":
                p = (t - s["t0"]) / (s["t1"] - s["t0"])
                xa, ya = POS[s["u"]]
                xb, yb = POS[s["v"]]
                return (xa + p * (xb - xa), ya + p * (yb - ya)), \
                       f"移动 {s['u']}→{s['v']}", s
            # 搜索/报告/标记阶段的状态由房间内文字显示，此处不再重复标注
            return POS[s["room"]], None, s
    last = segs[-1]
    return POS[last.get("room") or last["v"]], "完成，待命", None


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


def make_animation(solution, schedules, t_total, path, dt=0.5, fps=15):
    room_info = _room_phase_times(schedules)
    epilogue = 4.0
    frames = np.arange(0, t_total + epilogue, dt)

    fig = plt.figure(figsize=(11.0, 6.6))
    ax = fig.add_axes([0.02, 0.16, 0.96, 0.74])
    gx = fig.add_axes([0.06, 0.055, 0.90, 0.075])
    trails = {"A": [], "B": []}

    def draw(t):
        ax.clear()
        gx.clear()
        # ---- 房间状态 ----
        face, status = {}, {}
        n_cleared = 0
        for r in ROOMS:
            resp, s0, s1, m1 = room_info[r]
            if t < s0:
                face[r], status[r] = "#eeeeee", ""
            elif t < s1:
                face[r] = to_rgba(ROOM_COLOR[r], 0.40)
                status[r] = f"{resp} 搜索 {100 * (t - s0) / (s1 - s0):.0f}%"
            elif t < m1:
                face[r] = to_rgba(ROOM_COLOR[r], 0.65)
                status[r] = f"{resp} 报告/标记"
            else:
                face[r] = to_rgba(ROOM_COLOR[r], 0.90)
                status[r] = f"✓ {m1:.1f} s"
                n_cleared += 1
        draw_floorplan(ax, face)
        for r in ROOMS:
            if status[r]:
                ax.text(POS[r][0], POS[r][1] + (1.15 if POS[r][1] > 0 else -1.15),
                        status[r], ha="center", fontsize=9.5, fontweight="bold",
                        color="#1e8449" if status[r].startswith("✓") else "#333333",
                        zorder=7)
        # ---- 响应者 ----
        for resp in ("A", "B"):
            (x, y), txt, _ = _responder_state(schedules[resp], t)
            trails[resp].append((x, y))
            if len(trails[resp]) > 1:
                xs, ys = zip(*trails[resp])
                ax.plot(xs, ys, color=RESP_COLOR[resp], lw=1.2, alpha=0.35, zorder=5)
            ax.scatter([x], [y], s=330, color=RESP_COLOR[resp], zorder=8,
                       edgecolor="white", linewidth=1.6)
            ax.text(x, y, resp, ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white", zorder=9)
            if txt:
                ax.annotate(txt, xy=(x, y), xytext=(x + 0.5, y),
                            fontsize=8.5, color=RESP_COLOR[resp], zorder=9,
                            ha="left", va="center")
        ax.set_xlim(-6.4, 35.8)
        ax.set_ylim(-6.9, 6.9)
        ax.set_title(f"最优方案清查过程    T = {min(t, t_total):6.1f} s / {t_total:.1f} s"
                     f"    已清查 {n_cleared}/6 间",
                     fontsize=13, fontweight="bold", pad=10)
        # ---- 甘特进度条 ----
        for row, resp in enumerate(("B", "A")):
            for s in schedules[resp]:
                if s["kind"] == "move":
                    color = "#c8c8c8"
                elif s["kind"] == "search":
                    color = ROOM_COLOR[s["room"]]
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
        gx.set_ylim(-0.15, 1.9)
        gx.set_yticks([0.5, 1.5])
        gx.set_yticklabels(["B", "A"], fontsize=10, fontweight="bold")
        gx.set_xlabel("时间 (s)　灰=移动，彩色=搜索，深灰=报告+标记", fontsize=9)
        gx.tick_params(labelsize=8)
        for spine in ("top", "right"):
            gx.spines[spine].set_visible(False)

    ani = FuncAnimation(fig, draw, frames=frames)
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return len(frames)
