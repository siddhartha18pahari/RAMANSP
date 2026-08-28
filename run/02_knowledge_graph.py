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
    fig, ax = plt.subplots(figsize=(11, 8))
    plotting.graph(G, ax=ax, community=out["comm_of"] or None)
    ax.set_title(f"Raman cross-sample knowledge graph  "
                 f"({out['n_nodes']} nodes, {out['n_edges']} edges, "
                 f"{out['n_communities']} communities, Q={out['modularity']:.2f})",
                 fontsize=10)
    fig.savefig(FIGS / "kg_graph.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    # --- figure: acquisition similarity heatmap -------------------
    acq = [r for r in records if r["kind"] in ("raw_map", "raw_scan", "fit_table")]
    specs = kg._resample_all(acq)
    Xz, _ = kg._feature_matrix(acq)
    n = len(acq)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if specs[i] is not None and specs[j] is not None:
                S[i, j] = kg._cos(specs[i], specs[j])
            else:
                S[i, j] = kg._cos(Xz[i], Xz[j])
    order = np.argsort([r.get("features", {}).get("disorder_median", np.nan) for r in acq])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(S[np.ix_(order, order)], cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([acq[o]["acq_id"].replace("acq-", "") for o in order], fontsize=5, rotation=90)
    ax.set_yticklabels([acq[o]["acq_id"].replace("acq-", "") for o in order], fontsize=5)
    ax.set_title("acquisition similarity (spectral where available, else tabular)\n"
                 "ordered by disorder index", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIGS / "kg_similarity_heatmap.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    print(f"nodes={out['n_nodes']} edges={out['n_edges']} "
          f"communities={out['n_communities']} modularity={out['modularity']:.3f} "
          f"purity={out['community_purity_vs_carbon_class']}")
    print(f"component sizes: {out['component_sizes'][:8]}")


if __name__ == "__main__":
    main()
