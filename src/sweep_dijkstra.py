# -*- coding: utf-8 -*-
"""Dijkstra 最短路径算法（二叉堆优先队列实现）。

符号约定（与《迪杰斯特拉算法公式.md》一致）：
  dist[v] — 从源点 s 到 v 的当前最短通行时间上界
  prev[v] — 最短路径树上 v 的前驱节点
  松弛    — 若 dist[u] + c(u,v) < dist[v]，则更新 dist[v]、prev[v]

复杂度 O((|V| + |E|) log |V|)；要求所有边权 c_e >= 0（通行时间天然满足）。
"""
import heapq
import math


def dijkstra(graph, source):
    """从 source 出发跑 Dijkstra，返回 (dist, prev) 两个字典。"""
    dist = {n: math.inf for n in graph}   # 初始化：dist[s]=0，其余 +inf
    prev = {n: None for n in graph}
    dist[source] = 0.0
    pq = [(0.0, source)]                  # (dist, node) 最小堆
    settled = set()                       # 已确定最短路的节点
    while pq:
        d_u, u = heapq.heappop(pq)        # 取出 dist 最小的未确定节点
        if u in settled:
            continue
        settled.add(u)                    # 贪心确定 u 的最短路
        for v, c in graph[u]:             # 对每条出边做松弛
            nd = d_u + c
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def shortest_path(prev, source, target):
    """由 prev 表还原 source -> target 的最短路径节点序列。"""
    path, node = [], target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path if path and path[0] == source else None
