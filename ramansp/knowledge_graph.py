"""Cross-sample knowledge graph.

A corpus of Raman acquisitions is turned into one graph so it can be queried as
a whole: which specimens share a spectral signature, which cluster together by
disorder, which endmembers recur across samples.

Node types
----------
``specimen``    an anonymised physical sample (``S01`` ...)
``material``    a material / reference class (``MAT-*`` / ``REF-*``)
``config``      an acquisition configuration (``CFG-*``)
``acquisition`` one exported file (a map, a fit table, a spectrum)
``endmember``   a VCA endmember recovered from a raw map
``splatmodel``  a fitted Gaussian field for a raw map

Edge types
----------
``HAS_ACQUISITION`` ``USED_CONFIG`` ``OF_MATERIAL`` ``HAS_ENDMEMBER``
``MODELLED_BY``
``SIMILAR_TO``          acquisitions whose feature vectors have cosine >= tau;
                        ``basis`` records whether it was spectral or tabular
``SHARES_ENDMEMBER``    specimens with a matched endmember (cosine >= 0.9)
``DISORDER_NEIGHBOUR``  acquisitions within ``delta`` of each other on the
                        bounded disorder index; every such edge carries the
                        preprocessing ``protocol`` it was computed under,
                        because that choice moves the number (prior finding).

Honesty guards carried from the prior project: tabular (fit-table) similarity is
flagged ``confidence='low'`` (2 of 5 bands, no coordinates); disorder edges are
never comparable across protocols.
"""

from __future__ import annotations

import numpy as np


_COMMON_GRID = np.arange(1000.0, 1800.0 + 1e-6, 2.0)


def _resample(wn, y):
    wn = np.asarray(wn, float)
    y = np.asarray(y, float)
    k = (wn >= 990) & (wn <= 1810)
    if k.sum() < 8:
        return None
    r = np.interp(_COMMON_GRID, wn[k], y[k], left=np.nan, right=np.nan)
    if not np.isfinite(r).all():
        r = np.interp(_COMMON_GRID, wn[k], y[k])
    return r


def _cos(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _feature_matrix(records: list[dict]):
    keys: list[str] = []
    for r in records:
        for k, v in r.get("features", {}).items():
            if isinstance(v, (int, float)) and np.isfinite(v) and k not in keys:
                keys.append(k)
    keys.sort()
    X = np.full((len(records), len(keys)), np.nan)
    for i, r in enumerate(records):
        for j, k in enumerate(keys):
            v = r.get("features", {}).get(k, np.nan)
            if isinstance(v, (int, float)):
                X[i, j] = v
    col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    X = np.where(np.isfinite(X), X, col_mean)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - X.mean(0)) / sd, keys


def _mutual_knn(sim: np.ndarray, k: int, floor: float) -> set[tuple[int, int]]:
    """Edges where each node is in the other's top-k, and cosine >= floor.

    An absolute cosine threshold is the wrong tool for a corpus of carbons:
    every spectrum here has D and G bands, so cosine sits at 0.80 for the
    *median* unrelated pair and any usable cut links most of the graph. A
    mutual-kNN rule instead asks whether two acquisitions are among each
    other's nearest neighbours, which is scale-free and keeps the graph sparse
    enough for community structure to mean something.
    """
    n = sim.shape[0]
    if n < 2:
        return set()
    kk = int(min(max(k, 1), n - 1))
    top = {i: set(np.argsort(sim[i])[::-1][1:kk + 1]) for i in range(n)}
    return {(i, j) for i in range(n) for j in top[i]
            if i < j and i in top[j] and sim[i, j] >= floor}


def build_graph(
    records: list[dict],
    sim_threshold: float = 0.65,
    disorder_delta: float = 0.06,
    endmember_match: float = 0.9,
    k_neighbours: int = 5,
):
    """``records`` -- list of acquisition dicts from the corpus builder.

    Each needs: ``acq_id``, ``kind``, ``specimen``/``material``/``config`` (codes
    or None), ``features`` (dict of scalars). Optional: ``mean_spectrum`` +
    ``wavenumber``; ``endmembers`` (C, K) + ``endmember_wn``; ``protocol``;
    ``carbon_class``; ``splat`` (dict).
    """
    import networkx as nx

    G = nx.Graph()
    Xz, feat_keys = _feature_matrix(records)
    specs = _resample_all(records)

    for i, r in enumerate(records):
        aid = r["acq_id"]
        G.add_node(aid, ntype="acquisition", kind=r["kind"],
                   carbon_class=r.get("carbon_class", "?"),
                   n_points=int(r.get("n_points", 0)))
        for code, nt, et in [
            (r.get("specimen"), "specimen", "HAS_ACQUISITION"),
            (r.get("material"), "material", "OF_MATERIAL"),
            (r.get("config"), "config", "USED_CONFIG"),
        ]:
            if code:
                if code not in G:
                    G.add_node(code, ntype=nt)
                G.add_edge(code, aid, etype=et)
        if r.get("specimen") and r.get("material"):
            G.add_edge(r["specimen"], r["material"], etype="OF_MATERIAL")

        _ems = r.get("endmembers")
        if _ems is not None:
            for c in range(len(_ems)):
                en = f"{aid}:em{c}"
                G.add_node(en, ntype="endmember", parent=aid)
                G.add_edge(aid, en, etype="HAS_ENDMEMBER")

        sp = r.get("splat")
        if sp:
            sn = f"{aid}:splat"
            G.add_node(sn, ntype="splatmodel", **{k: float(v) for k, v in sp.items()
                                                  if isinstance(v, (int, float))})
            G.add_edge(aid, sn, etype="MODELLED_BY")

    # SIMILAR_TO -- mutual kNN within each basis, never across the two.
    # Spectral and tabular similarities are not on a common scale, so ranking
    # them together would let a curve-fit table outrank a real spectrum.
    n = len(records)
    for idx, basis, conf in (
        ([i for i in range(n) if specs[i] is not None], "spectral", "ok"),
        ([i for i in range(n) if specs[i] is None], "tabular", "low"),
    ):
        if len(idx) < 2:
            continue
        sim = np.eye(len(idx))
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ia, ib = idx[a], idx[b]
                c = (_cos(specs[ia], specs[ib]) if basis == "spectral"
                     else (_cos(Xz[ia], Xz[ib]) if Xz.shape[1] else 0.0))
                sim[a, b] = sim[b, a] = c
        for a, b in _mutual_knn(sim, k_neighbours, sim_threshold):
            G.add_edge(records[idx[a]]["acq_id"], records[idx[b]]["acq_id"],
                       etype="SIMILAR_TO", weight=round(float(sim[a, b]), 4),
                       basis=basis, confidence=conf)

    # DISORDER_NEIGHBOUR -------------------------------------------
    dvals = [(r["acq_id"], r.get("features", {}).get("disorder_median"),
              r.get("protocol", "unknown")) for r in records]
    for i in range(len(dvals)):
        for j in range(i + 1, len(dvals)):
            (ai, di, pi), (aj, dj, pj) = dvals[i], dvals[j]
            if di is None or dj is None or not (np.isfinite(di) and np.isfinite(dj)):
                continue
            if pi != pj:
                continue
            if abs(di - dj) <= disorder_delta:
                G.add_edge(ai, aj, etype="DISORDER_NEIGHBOUR",
                           delta=round(abs(di - dj), 4), protocol=pi)

    # SHARES_ENDMEMBER (specimen <-> specimen) ----------------------
    ems = []
    for i, r in enumerate(records):
        if r.get("endmembers") is not None and r.get("specimen"):
            wn = r.get("endmember_wn", r.get("wavenumber"))
            for em in r["endmembers"]:
                rr = _resample(wn, em) if wn is not None else None
                if rr is not None:
                    ems.append((r["specimen"], rr))
    for i in range(len(ems)):
        for j in range(i + 1, len(ems)):
            if ems[i][0] == ems[j][0]:
                continue
            c = _cos(ems[i][1], ems[j][1])
            if c >= endmember_match:
                G.add_edge(ems[i][0], ems[j][0], etype="SHARES_ENDMEMBER",
                           weight=round(c, 4))

    G.graph["feature_keys"] = feat_keys
    return G


def _resample_all(records):
    out = []
    for r in records:
        ms, wn = r.get("mean_spectrum"), r.get("wavenumber")
        out.append(_resample(wn, ms) if (ms is not None and wn is not None) else None)
    return out


def analyse(G) -> dict:
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities

    sub = G.subgraph([n for n in G if G.nodes[n]["ntype"] == "acquisition"]).copy()
    sub.remove_edges_from([(u, v) for u, v, d in sub.edges(data=True)
                           if d.get("etype") not in ("SIMILAR_TO", "DISORDER_NEIGHBOUR")])
    comms = list(greedy_modularity_communities(sub)) if sub.number_of_edges() else []
    comm_of = {n: k for k, c in enumerate(comms) for n in c}

    purity = None
    labelled = [(comm_of.get(n), G.nodes[n].get("carbon_class"))
                for n in sub if G.nodes[n].get("carbon_class", "?") not in ("?", None)]
    if comms and labelled:
        by_comm: dict = {}
        for c, lab in labelled:
            by_comm.setdefault(c, []).append(lab)
        hits = sum(max(np.unique(v, return_counts=True)[1]) for v in by_comm.values() if v)
        purity = hits / max(len(labelled), 1)

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_acquisitions": sub.number_of_nodes(),
        "n_communities": len(comms),
        "communities": [sorted(c) for c in comms],
        "modularity": (nx.algorithms.community.modularity(sub, comms)
                       if comms else float("nan")),
        "community_purity_vs_carbon_class": purity,
        "component_sizes": sorted((len(c) for c in nx.connected_components(G)), reverse=True),
        "comm_of": comm_of,
    }


NODE_STYLE = {
    "specimen":    {"color": "#c0392b", "shape": "star",     "size": 30},
    "material":    {"color": "#8e44ad", "shape": "triangle", "size": 26},
    "config":      {"color": "#d68910", "shape": "square",   "size": 22},
    "acquisition": {"color": "#2874a6", "shape": "dot",      "size": 16},
    "endmember":   {"color": "#7f8c8d", "shape": "dot",      "size": 8},
    "splatmodel":  {"color": "#148f77", "shape": "diamond",  "size": 20},
}
EDGE_STYLE = {
    "HAS_ACQUISITION":    {"color": "#c0392b", "width": 2.0},
    "OF_MATERIAL":        {"color": "#8e44ad", "width": 2.0},
    "USED_CONFIG":        {"color": "#d68910", "width": 1.6},
    "HAS_ENDMEMBER":      {"color": "#bdc3c7", "width": 0.8},
    "MODELLED_BY":        {"color": "#148f77", "width": 2.0},
    "SIMILAR_TO":         {"color": "#5dade2", "width": 1.2},
    "SHARES_ENDMEMBER":   {"color": "#16a085", "width": 2.2},
    "DISORDER_NEIGHBOUR": {"color": "#f5b041", "width": 1.0},
}


def write_pyvis(G, path: str, analysis_out: dict | None = None,
                height: str = "820px") -> str:
    """Interactive HTML view of the graph (pyvis / vis.js).

    Node shape and colour encode type, edge colour encodes relation, and every
    node and edge carries a tooltip with its attributes, so the graph can be
    explored rather than only looked at. Physics is enabled so communities
    separate visually under the force layout.
    """
    from pyvis.network import Network

    net = Network(height=height, width="100%", bgcolor="#ffffff",
                  font_color="#222222", directed=False, notebook=False,
                  cdn_resources="in_line")
    net.barnes_hut(gravity=-9000, central_gravity=0.25, spring_length=130,
                   spring_strength=0.02, damping=0.55)

    comm = (analysis_out or {}).get("comm_of", {})
    for n, d in G.nodes(data=True):
        nt = d.get("ntype", "?")
        st = NODE_STYLE.get(nt, {"color": "#95a5a6", "shape": "dot", "size": 12})
        tip = "<br>".join(f"<b>{k}</b>: {v}" for k, v in d.items())
        if n in comm:
            tip += f"<br><b>community</b>: {comm[n]}"
        deg = G.degree(n)
        net.add_node(n, label=str(n), title=f"<b>{n}</b><br>{tip}<br>degree: {deg}",
                     color=st["color"], shape=st["shape"],
                     size=st["size"] + min(deg, 12), group=str(comm.get(n, nt)))
    for u, v, d in G.edges(data=True):
        et = d.get("etype", "?")
        st = EDGE_STYLE.get(et, {"color": "#cccccc", "width": 1.0})
        tip = "<br>".join(f"<b>{k}</b>: {val}" for k, val in d.items())
        net.add_edge(u, v, color=st["color"], width=st["width"], title=tip)

    net.set_options("""
    {"interaction": {"hover": true, "tooltipDelay": 120,
                     "navigationButtons": true, "keyboard": true},
     "physics": {"stabilization": {"iterations": 220}}}
    """)
    html = net.generate_html(notebook=False)
    legend = _pyvis_legend(G)
    html = html.replace("<body>", "<body>" + legend, 1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


def _pyvis_legend(G) -> str:
    import collections

    nt = collections.Counter(d.get("ntype", "?") for _n, d in G.nodes(data=True))
    et = collections.Counter(d.get("etype", "?") for _u, _v, d in G.edges(data=True))
    rows = "".join(
        f'<span style="display:inline-block;margin:0 10px 4px 0;font-size:12px">'
        f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;'
        f'background:{NODE_STYLE.get(k, {}).get("color", "#95a5a6")};'
        f'margin-right:5px"></span>{k} ({v})</span>'
        for k, v in sorted(nt.items()))
    erows = "".join(
        f'<span style="display:inline-block;margin:0 10px 4px 0;font-size:12px">'
        f'<span style="display:inline-block;width:16px;height:3px;'
        f'background:{EDGE_STYLE.get(k, {}).get("color", "#ccc")};'
        f'margin-right:5px;vertical-align:middle"></span>{k} ({v})</span>'
        for k, v in sorted(et.items()))
    return (
        '<div style="font-family:system-ui,sans-serif;padding:10px 14px;'
        'border-bottom:1px solid #e2e2e2">'
        '<div style="font-weight:600;font-size:15px;margin-bottom:6px">'
        'Raman cross-sample knowledge graph</div>'
        f'<div style="margin-bottom:4px"><b style="font-size:12px">nodes</b> {rows}</div>'
        f'<div><b style="font-size:12px">edges</b> {erows}</div>'
        '<div style="font-size:11px;color:#666;margin-top:6px">'
        'drag to rearrange, hover for attributes, scroll to zoom</div></div>')


def write_graphml(G, path: str) -> None:
    import networkx as nx

    H = G.copy()
    for k, v in list(H.graph.items()):
        if not isinstance(v, (str, int, float, bool)):
            H.graph[k] = ",".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
    for _n, d in H.nodes(data=True):
        for k, v in list(d.items()):
            if not isinstance(v, (str, int, float, bool)):
                d[k] = str(v)
    for _u, _v, d in H.edges(data=True):
        for k, val in list(d.items()):
            if not isinstance(val, (str, int, float, bool)):
                d[k] = str(val)
    nx.write_graphml(H, path)
