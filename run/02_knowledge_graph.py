"""Stage 2 -- build the cross-sample knowledge graph from the corpus.

Outputs:
    graph/knowledge_graph.graphml
    graph/kg_summary.json
    tables/kg_communities.csv
    figures/kg_graph.png
    figures/kg_similarity_heatmap.png
"""

from __future__ import annotations

import pickle
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import CORPUS, FIGS, GRAPH, TABLES, dump_json  # noqa: E402

from ramansp import knowledge_graph as kg  # noqa: E402
from ramansp import plotting  # noqa: E402
from ramansp._style import acs_figsize, apply_style, save  # noqa: E402


def main():
    records = pickle.loads((CORPUS / "records.pkl").read_bytes())
    # splat metrics, if stage 3 has already run
    for r in records:
        sp = (CORPUS.parent / "splat" / r["acq_id"] / "metrics.json")
        if sp.exists():
            m = __import__("json").loads(sp.read_text())
            r["splat"] = {k: m[k] for k in ("psnr_eval_dB", "compression_ratio",
                                            "n_gaussians") if k in m}

    G = kg.build_graph(records, sim_threshold=0.65, disorder_delta=0.06)
    out = kg.analyse(G)
    kg.write_graphml(G, str(GRAPH / "knowledge_graph.graphml"))
    try:
        kg.write_pyvis(G, str(GRAPH / "knowledge_graph.html"), out)
        print(f"interactive graph -> {GRAPH / 'knowledge_graph.html'}")
    except Exception as e:  # noqa: BLE001
        print(f"  (pyvis view skipped: {type(e).__name__}: {e})")

    dump_json({k: v for k, v in out.items() if k != "comm_of"}, GRAPH / "kg_summary.json")

    rows = []
    for cid, members in enumerate(out["communities"]):
        for m in members:
            nd = G.nodes[m]
            rows.append({"community": cid, "acq_id": m, "kind": nd.get("kind"),
                         "carbon_class": nd.get("carbon_class")})
    pd.DataFrame(rows).to_csv(TABLES / "kg_communities.csv", index=False)

    # --- figure: the graph -----------------------------------------
    apply_style()
    fig, ax = plt.subplots(figsize=acs_figsize("double", 5.9))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.915, bottom=0.005)
    plotting.graph(G, ax=ax, community=out["comm_of"] or None)
    n_em = sum(1 for n in G if G.nodes[n].get("ntype") == "endmember")
    ax.set_title(f"Raman cross-sample knowledge graph: {out['n_nodes']} nodes, "
                 f"{out['n_edges']} edges, {out['n_communities']} communities "
                 f"(modularity Q = {out['modularity']:.2f}); acquisition colour is "
                 f"community" + chr(10) +
                 f"{n_em} endmember leaves and their edges are omitted "
                 f"from the drawing only", fontsize=7.5)
    save(fig, FIGS / "kg_graph.png")

    # --- figure: acquisition similarity heatmap -------------------
    # Two things were wrong with the first version of this panel. The project
    # style sheet draws a grid, and on an imshow that grid lands on top of the
    # data as a white lattice over every cell. And the matrix silently mixes two
    # different quantities: cosine between resampled spectra where both
    # acquisitions have one, and cosine between z-scored tabular features
    # otherwise. Those are not the same measurement, so which one produced each
    # row is now marked rather than left for the reader to guess.
    acq = [r for r in records if r["kind"] in ("raw_map", "raw_scan", "fit_table")]
    specs = kg._resample_all(acq)
    Xz, _ = kg._feature_matrix(acq)
    n = len(acq)
    S = np.zeros((n, n))
    spectral_pair = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(n):
            if specs[i] is not None and specs[j] is not None:
                S[i, j] = kg._cos(specs[i], specs[j])
                spectral_pair[i, j] = True
            else:
                S[i, j] = kg._cos(Xz[i], Xz[j])
    has_spec = np.array([s is not None for s in specs])

    # order by community, then by disorder inside each, so the block structure
    # the graph found is the structure the matrix shows
    comm_of = out["comm_of"] or {}
    dis = np.array([r.get("features", {}).get("disorder_median", np.nan) for r in acq],
                   float)
    key = [(comm_of.get(acq[i]["acq_id"], 10**6),
            dis[i] if np.isfinite(dis[i]) else 10.0, i) for i in range(n)]
    order = [i for _c, _d, i in sorted(key)]
    csorted = [comm_of.get(acq[i]["acq_id"], None) for i in order]
    bounds = [k for k in range(1, n) if csorted[k] != csorted[k - 1]]

    fig, ax = plt.subplots(figsize=acs_figsize("double", 6.5),
                           layout="constrained")
    ax.grid(False)
    im = ax.imshow(S[np.ix_(order, order)], cmap="magma", vmin=0, vmax=1,
                   interpolation="nearest")
    for b in bounds:
        ax.axhline(b - 0.5, color="#6fd0ff", lw=0.7, alpha=0.85)
        ax.axvline(b - 0.5, color="#6fd0ff", lw=0.7, alpha=0.85)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([acq[o]["acq_id"].replace("acq-", "") for o in order],
                       fontsize=6.0, rotation=90)
    ax.set_yticklabels([acq[o]["acq_id"].replace("acq-", "") for o in order],
                       fontsize=6.0)
    for t, o in zip(ax.get_yticklabels(), order):
        t.set_color("#1a1a1a" if has_spec[o] else "#c1651f")
    for t, o in zip(ax.get_xticklabels(), order):
        t.set_color("#1a1a1a" if has_spec[o] else "#c1651f")
    n_spec = int(has_spec.sum())
    ax.set_title("Acquisition similarity, ordered by graph community then by "
                 "disorder index" + chr(10) +
                 f"black labels ({n_spec}): cosine between resampled spectra.  "
                 f"orange labels ({n - n_spec}): cosine between z-scored tabular "
                 f"features" + chr(10) +
                 "blue lines are community boundaries; a pair spanning the two "
                 "label colours is scored on tabular features",
                 fontsize=7.0)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
    cb.ax.tick_params(labelsize=6.5)
    save(fig, FIGS / "kg_similarity_heatmap.png")

    print(f"nodes={out['n_nodes']} edges={out['n_edges']} "
          f"communities={out['n_communities']} modularity={out['modularity']:.3f} "
          f"purity={out['community_purity_vs_carbon_class']}")
    print(f"component sizes: {out['component_sizes'][:8]}")


if __name__ == "__main__":
    main()
