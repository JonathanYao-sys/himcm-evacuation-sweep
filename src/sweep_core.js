/* sweep_core.js — 基础场景最优清查算法核心（与 Python 版一一对应）
 *
 * 纯函数、无 DOM 依赖，可在浏览器 <script> 中直接使用，也可在 Node 中 require 后测试：
 *     const core = require("./src/sweep_core.js");
 *
 * 与 src/sweep_config.py + sweep_building.py + sweep_dijkstra.py + sweep_optimize.py
 * 保持同一套公式：Dijkstra 求最短路 → Held–Karp 求每名响应者对每个房间子集的
 * 最优访问时间 → 枚举房间分配取全局最优（max_r T_r 最小）。
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) { module.exports = factory(); }
  else { root.SweepCore = factory(); }
})(typeof self !== "undefined" ? self : globalThis, function () {
  "use strict";

  var RESPONDER_NAMES = "ABCDEFGHIJKLMNOP";
  var MAX_TIE_ENUM = 2000000;   // 并列解精确枚举的上限（防组合爆炸）
  var EPS = 1e-9;

  // ------------------------------------------------------------ 参数
  function defaultConfig(over) {
    var cfg = {
      // 建筑几何（m）
      nRoomsPerSide: 3, hallLength: 30, exitLength: 2, doorLength: 1,
      hallWidth: 2, roomDepth: 5,
      // 房间属性（默认值，可被 roomArea / roomOcc 按房间覆盖）
      area: 20, occupancy: 4,
      roomArea: {}, roomOcc: {},
      // 移动速度（m/s）
      vHall: 1.2, vRoom: 0.5,
      // 房间搜索时间模型（s）
      t0: 20, alpha: 0.8, beta: 8, tReport: 5, tMark: 3,
      // 响应者
      nResponders: 2, responderStarts: null,   // 例如 ["E_L","E_R"]
      // 可通行性（x_e = 0 的边），元素为 [u,v] 或 "u-v"
      blockedEdges: [],
      // 并列解是否按左右镜像去重（只影响报告口径）
      dedupMirrorTies: false
    };
    if (over) { for (var k in over) { if (over[k] !== undefined) cfg[k] = over[k]; } }
    return cfg;
  }

  function areaOf(cfg, room) {
    var v = cfg.roomArea ? cfg.roomArea[room] : undefined;
    return (v === undefined || v === null || v === "") ? cfg.area : Number(v);
  }
  function occOf(cfg, room) {
    var v = cfg.roomOcc ? cfg.roomOcc[room] : undefined;
    return (v === undefined || v === null || v === "") ? cfg.occupancy : Number(v);
  }
  // s_i = t_0 + α·a_i + β·n_i（只含搜索与引导）
  function searchTime(cfg, room) {
    return cfg.t0 + cfg.alpha * areaOf(cfg, room) + cfg.beta * occOf(cfg, room);
  }
  // s_i* = s_i + t_rep + t_mark（单间总处理时间）
  function totalTime(cfg, room) {
    return searchTime(cfg, room) + cfg.tReport + cfg.tMark;
  }

  // ------------------------------------------------------------ 建筑网络
  function blockedSet(list) {
    var s = new Set();
    (list || []).forEach(function (p) {
      if (!p) { return; }
      var u, v;
      if (Array.isArray(p)) { u = p[0]; v = p[1]; }
      else { var parts = String(p).split(/[-\u2013|,]/); u = parts[0]; v = parts[1]; }
      if (u && v) { s.add(u + "|" + v); s.add(v + "|" + u); }
    });
    return s;
  }

  function buildScene(cfg) {
    var n = Math.max(1, Math.round(cfg.nRoomsPerSide));
    var w = cfg.hallLength / n;
    var xs = [], doorNodes = [], top = [], bottom = [], i;
    for (i = 0; i < n; i++) {
      xs.push((i + 0.5) * w);
      doorNodes.push("N" + (i + 1));
      top.push("R" + (i + 1));
      bottom.push("R" + (n + i + 1));
    }
    var rooms = top.concat(bottom);
    var L = cfg.hallLength, Lex = cfg.exitLength;
    var roomY = cfg.hallWidth / 2 + 0.26 * cfg.roomDepth;

    var pos = { E_L: [-Lex, 0], H_L: [0, 0], H_R: [L, 0], E_R: [L + Lex, 0] };
    var roomRects = {}, doorX = {};
    for (i = 0; i < n; i++) {
      pos[doorNodes[i]] = [xs[i], 0];
      pos[top[i]] = [xs[i], roomY];
      pos[bottom[i]] = [xs[i], -roomY];
      roomRects[top[i]] = [xs[i] - w / 2, cfg.hallWidth / 2, w, cfg.roomDepth];
      roomRects[bottom[i]] = [xs[i] - w / 2,
        -(cfg.hallWidth / 2 + cfg.roomDepth), w, cfg.roomDepth];
      doorX[top[i]] = xs[i];
      doorX[bottom[i]] = xs[i];
    }

    // 边（长度 m）：出口段 2 条 + 走廊段 n+1 条 + 门 n*2 条
    var edges = [["E_L", "H_L", Lex], ["H_R", "E_R", Lex]];
    var prevX = 0, prevName = "H_L";
    for (i = 0; i < n; i++) {
      edges.push([prevName, doorNodes[i], xs[i] - prevX]);
      prevX = xs[i]; prevName = doorNodes[i];
    }
    edges.push([prevName, "H_R", L - prevX]);
    for (i = 0; i < n; i++) {
      edges.push([doorNodes[i], top[i], cfg.doorLength]);
      edges.push([doorNodes[i], bottom[i], cfg.doorLength]);
    }

    var blocked = blockedSet(cfg.blockedEdges);
    var activeEdges = edges.filter(function (e) { return !blocked.has(e[0] + "|" + e[1]); });

    var weight = {}, graph = {};
    Object.keys(pos).forEach(function (node) { graph[node] = []; });
    activeEdges.forEach(function (e) {
      var c = e[2] / cfg.vHall;
      weight[e[0] + "|" + e[1]] = c;
      weight[e[1] + "|" + e[0]] = c;
      graph[e[0]].push([e[1], c]);
      graph[e[1]].push([e[0], c]);
    });

    // 响应者起点：默认交替从左/右出口进入，可由 responderStarts 覆盖
    var names = [], starts = {};
    var nResp = Math.max(1, Math.round(cfg.nResponders));
    for (i = 0; i < nResp; i++) {
      names.push(RESPONDER_NAMES[i]);
      var ov = cfg.responderStarts ? cfg.responderStarts[i] : null;
      starts[RESPONDER_NAMES[i]] = ov || (i % 2 === 0 ? "E_L" : "E_R");
    }

    // 左右镜像映射（用于并列解去重）
    var mirror = { E_L: "E_R", E_R: "E_L", H_L: "H_R", H_R: "H_L" };
    for (i = 0; i < n; i++) {
      mirror[doorNodes[i]] = doorNodes[n - 1 - i];
      mirror[top[i]] = top[n - 1 - i];
      mirror[bottom[i]] = bottom[n - 1 - i];
    }

    var roomTime = {}, roomSearch = {};
    rooms.forEach(function (r) {
      roomTime[r] = totalTime(cfg, r);
      roomSearch[r] = searchTime(cfg, r);
    });

    var scene = {
      cfg: cfg, n: n, roomWidth: w, pos: pos, edges: activeEdges, allEdges: edges,
      weight: weight, graph: graph, rooms: rooms, top: top, bottom: bottom,
      doorNodes: doorNodes, doorX: doorX, roomRects: roomRects, mirror: mirror,
      responderNames: names, starts: starts, exits: ["E_L", "E_R"],
      roomTime: roomTime, roomSearch: roomSearch,
      hallY: cfg.hallWidth / 2, yTop: cfg.hallWidth / 2 + cfg.roomDepth,
      xBounds: [-Lex, L + Lex]
    };

    // 可行性：每间房必须能从某个起点到达
    var reachable = new Set();
    Object.keys(starts).forEach(function (r) {
      bfs(scene, starts[r]).forEach(function (node) { reachable.add(node); });
    });
    var unreachable = rooms.filter(function (r) { return !reachable.has(r); });
    scene.unreachable = unreachable;      // UI 用它给出友好提示；求解时会再报错
    return scene;
  }

  function bfs(scene, source) {
    var seen = new Set([source]), stack = [source];
    while (stack.length) {
      var u = stack.pop(), nbrs = scene.graph[u] || [];
      for (var i = 0; i < nbrs.length; i++) {
        var v = nbrs[i][0];
        if (!seen.has(v)) { seen.add(v); stack.push(v); }
      }
    }
    return seen;
  }

  function isMirrorSymmetric(scene) {
    var lengths = {};
    scene.edges.forEach(function (e) { lengths[e[0] + "|" + e[1]] = e[2]; lengths[e[1] + "|" + e[0]] = e[2]; });
    for (var i = 0; i < scene.edges.length; i++) {
      var e = scene.edges[i], mu = scene.mirror[e[0]], mv = scene.mirror[e[1]];
      var key = mu + "|" + mv;
      if (!(key in lengths) || Math.abs(lengths[key] - e[2]) > 1e-12) { return false; }
    }
    var s = scene.responderNames.map(function (r) { return scene.mirror[scene.starts[r]]; }).sort();
    var t = scene.responderNames.map(function (r) { return scene.starts[r]; }).sort();
    if (s.join(",") !== t.join(",")) { return false; }
    for (var j = 0; j < scene.rooms.length; j++) {           // 房间处理时间也要镜像相等
      var r = scene.rooms[j];
      if (Math.abs(scene.roomTime[r] - scene.roomTime[scene.mirror[r]]) > 1e-12) { return false; }
    }
    return true;
  }

  // ------------------------------------------------------------ Dijkstra
  function MinHeap() { this.a = []; }
  MinHeap.prototype.push = function (item) {
    var a = this.a; a.push(item); var i = a.length - 1;
    while (i > 0) {
      var p = (i - 1) >> 1;
      if (a[p][0] <= a[i][0]) { break; }
      var t = a[p]; a[p] = a[i]; a[i] = t; i = p;
    }
  };
  MinHeap.prototype.pop = function () {
    var a = this.a;
    if (!a.length) { return null; }
    var top = a[0], last = a.pop();
    if (a.length) {
      a[0] = last;
      var i = 0;
      for (;;) {
        var l = 2 * i + 1, r = l + 1, m = i;
        if (l < a.length && a[l][0] < a[m][0]) { m = l; }
        if (r < a.length && a[r][0] < a[m][0]) { m = r; }
        if (m === i) { break; }
        var t = a[m]; a[m] = a[i]; a[i] = t; i = m;
      }
    }
    return top;
  };

  function dijkstra(graph, source) {
    var dist = {}, prev = {};
    Object.keys(graph).forEach(function (node) { dist[node] = Infinity; prev[node] = null; });
    dist[source] = 0;
    var heap = new MinHeap();
    heap.push([0, source]);
    var settled = new Set();
    while (heap.a.length) {
      var item = heap.pop(), d = item[0], u = item[1];
      if (settled.has(u)) { continue; }
      settled.add(u);
      var nbrs = graph[u] || [];
      for (var i = 0; i < nbrs.length; i++) {
        var v = nbrs[i][0], c = nbrs[i][1], nd = d + c;
        if (nd < dist[v]) { dist[v] = nd; prev[v] = u; heap.push([nd, v]); }
      }
    }
    return { dist: dist, prev: prev };
  }

  function shortestPath(prev, source, target) {
    var path = [], node = target;
    while (node !== null && node !== undefined) { path.push(node); node = prev[node]; }
    path.reverse();
    return (path.length && path[0] === source) ? path : null;
  }

  function preparePairs(scene) {
    var nodes = [], seen = {};
    scene.responderNames.forEach(function (r) {
      var s = scene.starts[r];
      if (!seen[s]) { seen[s] = 1; nodes.push(s); }
    });
    scene.rooms.forEach(function (r) { if (!seen[r]) { seen[r] = 1; nodes.push(r); } });
    var dists = {}, prevs = {};
    nodes.forEach(function (node) {
      var res = dijkstra(scene.graph, node);
      dists[node] = res.dist; prevs[node] = res.prev;
    });
    return { dists: dists, prevs: prevs, nodes: nodes };
  }

  // ------------------------------------------------------------ 路线评估
  function routeEval(route, start, dists, roomTime) {
    var travel = 0, node = start, legs = [];
    for (var i = 0; i < route.length; i++) {
      var room = route[i], d = dists[node][room];
      if (!isFinite(d)) { throw new Error(node + " 无法到达 " + room); }
      legs.push([node, room, d]);
      travel += d; node = room;
    }
    var proc = 0;
    for (var j = 0; j < route.length; j++) { proc += roomTime[route[j]]; }
    return { total: travel + proc, travel: travel, legs: legs };
  }

  // ------------------------------------------- Held–Karp：子集最优顺序
  function heldKarp(start, rooms, dists, roomTime) {
    var m = rooms.length, size = 1 << m;
    if (m === 0) { return new Map(); }
    var dp = new Float64Array(size * m);
    for (var t = 0; t < dp.length; t++) { dp[t] = Infinity; }
    var par = new Int16Array(size * m);
    for (var t2 = 0; t2 < par.length; t2++) { par[t2] = -1; }

    for (var i = 0; i < m; i++) {
      var d = dists[start][rooms[i]];
      if (isFinite(d)) { dp[((1 << i) * m) + i] = d + roomTime[rooms[i]]; }
    }
    for (var mask = 1; mask < size; mask++) {
      for (var last = 0; last < m; last++) {
        if (!((mask >> last) & 1)) { continue; }
        var cur = dp[mask * m + last];
        if (!isFinite(cur)) { continue; }
        for (var nxt = 0; nxt < m; nxt++) {
          if ((mask >> nxt) & 1) { continue; }
          var nm = mask | (1 << nxt);
          var nd = cur + dists[rooms[last]][rooms[nxt]] + roomTime[rooms[nxt]];
          if (nd < dp[nm * m + nxt] - 1e-12) { dp[nm * m + nxt] = nd; par[nm * m + nxt] = last; }
        }
      }
    }
    var best = new Map();
    for (var mk = 1; mk < size; mk++) {
      var bt = Infinity, bl = -1;
      for (var l2 = 0; l2 < m; l2++) {
        var v = dp[mk * m + l2];
        if (v < bt - 1e-12) { bt = v; bl = l2; }
      }
      if (bl < 0) { continue; }
      var route = [], mm = mk, lastIdx = bl;
      while (lastIdx >= 0) {
        route.push(rooms[lastIdx]);
        var p = par[mm * m + lastIdx];
        mm ^= (1 << lastIdx);
        lastIdx = p;
      }
      route.reverse();
      best.set(mk, { time: bt, route: route });
    }
    return best;
  }

  function factorial(k) { var r = 1; for (var i = 2; i <= k; i++) { r *= i; } return r; }

  function permute(arr, cb) {
    var a = arr.slice();
    (function rec(k) {
      if (k === a.length) { cb(a); return; }
      for (var i = k; i < a.length; i++) {
        var t = a[k]; a[k] = a[i]; a[i] = t;
        rec(k + 1);
        t = a[k]; a[k] = a[i]; a[i] = t;
      }
    })(0);
  }

  function forEachCombo(lists, cb) {
    var cur = new Array(lists.length);
    (function rec(i) {
      if (i === lists.length) { cb(cur); return; }
      for (var j = 0; j < lists[i].length; j++) { cur[i] = lists[i][j]; rec(i + 1); }
    })(0);
  }

  function cmpStr(a, b) { return a < b ? -1 : (a > b ? 1 : 0); }
  function cmpArray(a, b) {
    var n = Math.min(a.length, b.length);
    for (var i = 0; i < n; i++) { var c = cmpStr(a[i], b[i]); if (c) { return c; } }
    return a.length - b.length;
  }
  function cmpKey(a, b) {
    var n = Math.min(a.length, b.length);
    for (var i = 0; i < n; i++) { var c = cmpArray(a[i], b[i]); if (c) { return c; } }
    return a.length - b.length;
  }

  // ------------------------------------------------------------ 全局最优
  function optimize(scene, dists, opts) {
    opts = opts || {};
    var responders = scene.responderNames, rooms = scene.rooms;
    var m = rooms.length, N = responders.length, size = 1 << m;
    if (Math.pow(N, m) > (opts.maxAssignments || 5000000)) {
      throw new Error("组合过大（" + N + "^" + m + " = " + Math.pow(N, m).toExponential(2) +
        " 种分配），请减少房间数或响应者人数");
    }

    // 1) Held–Karp
    var timeByMask = {}, routeByMask = {};
    responders.forEach(function (r) {
      var table = heldKarp(scene.starts[r], rooms, dists, scene.roomTime);
      var bt = new Float64Array(size); bt.fill(Infinity);
      var br = new Array(size);
      table.forEach(function (val, key) { bt[key] = val.time; br[key] = val.route; });
      timeByMask[r] = bt; routeByMask[r] = br;
    });

    // 2) 枚举全部房间分配
    var evaluated = 0, bestMakespan = Infinity, bestAssignments = [];
    var fact = []; for (var f = 0; f <= m; f++) { fact.push(factorial(f)); }
    var assign = new Int32Array(m);
    var maskArr = new Int32Array(N), sizeArr = new Int32Array(N);
    for (;;) {
      for (var z = 0; z < N; z++) { maskArr[z] = 0; sizeArr[z] = 0; }
      for (var i = 0; i < m; i++) { maskArr[assign[i]] |= (1 << i); sizeArr[assign[i]]++; }
      var cnt = 1, ok = true, mt = 0;
      for (var r2 = 0; r2 < N; r2++) {
        cnt *= fact[sizeArr[r2]];
        var mk = maskArr[r2];
        var tv = mk === 0 ? 0 : timeByMask[responders[r2]][mk];
        if (!isFinite(tv)) { ok = false; break; }
        if (tv > mt) { mt = tv; }
      }
      if (ok) {
        evaluated += cnt;
        if (mt < bestMakespan - EPS) { bestMakespan = mt; bestAssignments = [Array.prototype.slice.call(maskArr, 0, N)]; }
        else if (Math.abs(mt - bestMakespan) <= EPS) { bestAssignments.push(Array.prototype.slice.call(maskArr, 0, N)); }
      }
      var k = 0;
      while (k < m && ++assign[k] === N) { assign[k] = 0; k++; }
      if (k === m) { break; }
    }
    if (!bestAssignments.length) { throw new Error("无可行解：请检查阻断设置"); }

    // 3) 并列解统计 + 字典序最小代表解（超限时退化为 Held–Karp 解）
    var estTies = 0;
    bestAssignments.forEach(function (masks) {
      var c = 1;
      masks.forEach(function (mk) { c *= fact[popcount(mk)]; });
      estTies += c;
    });
    var doDedup = !!scene.cfg.dedupMirrorTies && isMirrorSymmetric(scene);
    var capped = estTies > (opts.maxTieEnum || MAX_TIE_ENUM);
    var nTies = null, tieKeys = null, repKey = null, repRoutes = null;
    if (doDedup) { tieKeys = new Set(); }
    if (!capped) {
      nTies = 0;
      bestAssignments.forEach(function (masks) {
        var groups = [], r;
        for (r = 0; r < N; r++) {
          var g = [];
          for (var q = 0; q < m; q++) { if ((masks[r] >> q) & 1) { g.push(rooms[q]); } }
          groups.push(g);
        }
        var perResp = groups.map(function (g, rIdx) {
          var list = [];
          permute(g, function (p) {
            list.push({ time: routeEval(p, scene.starts[responders[rIdx]], dists, scene.roomTime).total, route: p.slice() });
          });
          return list;
        });
        forEachCombo(perResp, function (combo) {
          var mx = 0;
          for (var c2 = 0; c2 < combo.length; c2++) { if (combo[c2].time > mx) { mx = combo[c2].time; } }
          if (Math.abs(mx - bestMakespan) > EPS) { return; }
          nTies++;
          var routes = combo.map(function (c3) { return c3.route; });
          if (repKey === null || cmpKey(routes, repKey) < 0) { repKey = routes; }
          if (tieKeys) { tieKeys.add(canonicalKey(scene, routes)); }
        });
      });
    }
    // 代表解
    var rep = repKey;
    if (!rep) {
      var masks0 = bestAssignments[0];
      rep = responders.map(function (r, idx) {
        return masks0[idx] === 0 ? [] : routeByMask[r][masks0[idx]].slice();
      });
    }

    // 4) 展开代表解
    var routes = {}, times = {}, travel = {}, legs = {};
    responders.forEach(function (r, idx) {
      var ev = routeEval(rep[idx], scene.starts[r], dists, scene.roomTime);
      routes[r] = rep[idx].slice(); times[r] = ev.total; travel[r] = ev.travel; legs[r] = ev.legs;
    });

    return {
      makespan: bestMakespan, routes: routes, times: times, travel: travel, legs: legs,
      evaluated: evaluated, nTies: nTies, estTies: estTies, tiesCapped: capped,
      nTiesDistinct: tieKeys ? tieKeys.size : null,
      nAssignmentsOptimal: bestAssignments.length,
      responders: responders.slice()
    };
  }

  function popcount(x) { var c = 0; while (x) { x &= x - 1; c++; } return c; }

  function canonicalKey(scene, routes) {
    var responders = scene.responderNames;
    var partner = {};
    responders.forEach(function (r) {
      var target = scene.mirror[scene.starts[r]];
      partner[r] = responders.filter(function (rr) { return scene.starts[rr] === target; })[0];
    });
    var direct = routes.map(function (x) { return x.slice(); });
    var mirrored = {};
    responders.forEach(function (r, i) {
      mirrored[partner[r]] = routes[i].map(function (x) { return scene.mirror[x]; });
    });
    var mk = responders.map(function (r) { return mirrored[r]; });
    return cmpKey(direct, mk) <= 0 ? JSON.stringify(direct) : JSON.stringify(mk);
  }

  function bestFullRoute(scene, dists, responder) {
    var table = heldKarp(scene.starts[responder], scene.rooms, dists, scene.roomTime);
    return table.get((1 << scene.rooms.length) - 1) || null;
  }

  // ------------------------------------------------------------ 时间轴
  function buildSchedule(scene, routes, prevs) {
    var schedules = {}, clearTimes = {};
    scene.responderNames.forEach(function (r) {
      var route = routes[r] || [], segs = [], t = 0, node = scene.starts[r];
      route.forEach(function (room) {
        var path = shortestPath(prevs[node], node, room);
        if (!path) { throw new Error(node + " 无法到达 " + room); }
        for (var i = 0; i < path.length - 1; i++) {
          var u = path[i], v = path[i + 1], c = scene.weight[u + "|" + v];
          segs.push({ kind: "move", t0: t, t1: t + c, u: u, v: v });
          t += c;
        }
        var sSearch = scene.roomSearch[room];
        segs.push({ kind: "search", t0: t, t1: t + sSearch, room: room }); t += sSearch;
        segs.push({ kind: "report", t0: t, t1: t + scene.cfg.tReport, room: room }); t += scene.cfg.tReport;
        segs.push({ kind: "mark", t0: t, t1: t + scene.cfg.tMark, room: room }); t += scene.cfg.tMark;
        clearTimes[room] = t;
        node = room;
      });
      schedules[r] = segs;
    });
    return { schedules: schedules, clearTimes: clearTimes };
  }

  return {
    RESPONDER_NAMES: RESPONDER_NAMES,
    defaultConfig: defaultConfig,
    areaOf: areaOf, occOf: occOf, searchTime: searchTime, totalTime: totalTime,
    buildScene: buildScene, isMirrorSymmetric: isMirrorSymmetric,
    dijkstra: dijkstra, shortestPath: shortestPath, preparePairs: preparePairs,
    routeEval: routeEval, heldKarp: heldKarp, optimize: optimize,
    bestFullRoute: bestFullRoute, buildSchedule: buildSchedule,
    blockedSet: blockedSet, bfs: bfs
  };
});
