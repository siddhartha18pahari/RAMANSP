import numpy as np

from ramansp import knowledge_graph as kg


def _records():
    wn = np.arange(1000.0, 1800.0, 2.0)
    d_band = np.exp(-((wn - 1350) / 40) ** 2)
    g_band = np.exp(-((wn - 1585) / 30) ** 2)
    recs = []
    for i in range(4):
        mix = 0.2 + 0.2 * i
        spec = mix * d_band + (1 - mix) * g_band
        recs.append(dict(
            acq_id=f"acq-{i:03d}", kind="raw_map",
            specimen=f"S{i // 2 + 1:02d}", material="MAT-A" if i < 2 else "MAT-B",
            config="CFG-01", protocol="carbon_dg", carbon_class="A" if i < 2 else "B",
            n_points=400, mean_spectrum=spec, wavenumber=wn,
            endmembers=np.vstack([d_band, g_band]), endmember_wn=wn,
            features={"disorder_median": float(mix), "g_width_median": 30.0 + i},
        ))
    return recs


def test_graph_builds_and_has_expected_edges():
    G = kg.build_graph(_records(), sim_threshold=0.6, disorder_delta=0.25)
    assert G.number_of_nodes() > 4
    etypes = {d["etype"] for _, _, d in G.edges(data=True)}
    assert "HAS_ACQUISITION" in etypes
    assert "SIMILAR_TO" in etypes
    assert any(d["etype"] == "SIMILAR_TO" and d["basis"] == "spectral"
               for _, _, d in G.edges(data=True))


def test_analyse_returns_communities_and_purity():
    G = kg.build_graph(_records(), sim_threshold=0.5, disorder_delta=0.25)
    out = kg.analyse(G)
    assert out["n_acquisitions"] == 4
    assert out["community_purity_vs_carbon_class"] is None or 0.0 <= out[
        "community_purity_vs_carbon_class"] <= 1.0


def test_mutual_knn_keeps_the_graph_sparse():
    """An absolute cosine cut links most of a carbon corpus; kNN must not."""
    rng = np.random.default_rng(0)
    n = 30
    wn = np.arange(1000.0, 1800.0, 2.0)
    base = np.exp(-((wn - 1350) / 40) ** 2) + np.exp(-((wn - 1585) / 30) ** 2)
    recs = []
    for i in range(n):
        spec = base + 0.05 * rng.standard_normal(wn.size)
        recs.append(dict(acq_id=f"acq-{i:03d}", kind="raw_map", specimen=None,
                         material=None, config=None, protocol="carbon_dg",
                         carbon_class="?", n_points=10, mean_spectrum=spec,
                         wavenumber=wn, features={}))
    G = kg.build_graph(recs, sim_threshold=0.5, k_neighbours=4)
    sim_edges = [e for e in G.edges(data=True) if e[2]["etype"] == "SIMILAR_TO"]
    # every pair here has cosine ~1.0; a threshold rule would link all 435
    assert len(sim_edges) < 0.25 * (n * (n - 1) / 2), len(sim_edges)
    deg = {}
    for u, v, _d in sim_edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    assert max(deg.values()) <= 4


def test_graphml_roundtrip(tmp_path):
    G = kg.build_graph(_records())
    p = tmp_path / "g.graphml"
    kg.write_graphml(G, str(p))
    import networkx as nx
    H = nx.read_graphml(str(p))
    assert H.number_of_nodes() == G.number_of_nodes()
