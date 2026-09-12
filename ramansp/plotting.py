"""Small matplotlib helpers shared across the run scripts."""

from __future__ import annotations

import numpy as np

from .containers import SpectralImage, Spectrum


def spectrum(sp: Spectrum, ax=None, **kw):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    ax.plot(sp.wavenumber, sp.intensities, **kw)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("intensity (a.u.)")
    return ax


def image_map(img2d: np.ndarray, ax=None, cmap: str = "magma", title: str | None = None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    im = ax.imshow(img2d, origin="lower", cmap=cmap)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return ax, im


def overview(image: SpectralImage, path: str | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(image.wavenumber, image.mean_spectrum().intensities, color="0.2")
    ax[0].set_title("mean spectrum", fontsize=9)
    ax[0].set_xlabel("Raman shift (cm$^{-1}$)")
    if image.is_gridded:
        image_map(image.flat().mean(1).reshape(image.band_shape), ax=ax[1],
                  title="total counts")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=320, bbox_inches="tight")
        plt.close(fig)
    return fig


def _community_layout(H, comm_of, seed=0, spread=2.6, iterations=90):
    """Spring layout seeded by community, with components kept apart.

    A plain spring layout on this graph is unusable: the eight small
    components repel the 218-node component until it collapses to a dot in
    the middle of an empty canvas. Here the giant component is laid out on
    its own from a community-seeded start, so communities land in
    distinguishable regions, and the small components are parked in a strip
    below it rather than being allowed to push it around.
    """
    import networkx as nx
    import numpy as np

    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    giant = H.subgraph(comps[0])

    # every node needs a community for the seed; the detector only labels
    # acquisitions, so provenance nodes inherit from their neighbours
    comm = {n: comm_of[n] for n in giant if comm_of and n in comm_of}
    for _ in range(3):
        for n in giant:
            if n in comm:
                continue
            votes = [comm[m] for m in giant.neighbors(n) if m in comm]
            if votes:
                comm[n] = max(set(votes), key=votes.count)
    for n in giant:
        comm.setdefault(n, -1)

    sizes = {c: sum(v == c for v in comm.values()) for c in set(comm.values())}
    order = sorted(sizes, key=lambda c: -sizes[c])
    rng = np.random.default_rng(seed)
    ang = {c: 2 * np.pi * i / max(len(order), 1) for i, c in enumerate(order)}
    init = {n: np.array([np.cos(ang[comm[n]]), np.sin(ang[comm[n]])])
            + rng.normal(0, 0.09, 2) for n in giant}

    pos = nx.spring_layout(giant, pos=init, seed=seed, iterations=iterations,
                           k=spread / max(len(giant) ** 0.5, 1))
    P = np.array(list(pos.values()))
    lo, hi = P.min(0), P.max(0)
    rngxy = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    pos = {n: (2 * (p - lo) / rngxy - 1) for n, p in pos.items()}

    # strip of orphans underneath, on their own row
    rest = [n for c in comps[1:] for n in sorted(c, key=str)]
    for i, n in enumerate(rest):
        x = -1.0 + 2.0 * (i + 0.5) / max(len(rest), 1)
        pos[n] = np.array([x, -1.34])
    return pos, comm, len(comps) - 1, rest


def graph(G, ax=None, seed: int = 0, community: dict | None = None,
          path: str | None = None, show_endmembers: bool = False,
          label_kinds=("specimen", "material"), label_top_hubs: int = 6):
    """Draw the knowledge graph so that its structure is readable.

    Two decisions make the difference. Endmember leaves are hidden by default:
    there are four per map, each attached to exactly one acquisition, so they
    add 100 nodes and 100 edges that carry no connectivity information, and the
    cross-map signal they do carry is already on the specimen-level
    ``SHARES_ENDMEMBER`` edges. And node type is encoded by marker while only
    provenance nodes and the busiest acquisitions are named, because labelling
    all 108 acquisitions produced an unreadable pile in the core.
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import networkx as nx

    from ._style import INK, OKABE, apply_style
    apply_style()

    ax = ax or plt.gca()
    H = G if show_endmembers else G.subgraph(
        [n for n in G if G.nodes[n].get("ntype") != "endmember"]).copy()
    kind = {n: H.nodes[n].get("ntype", "?") for n in H}
    deg = dict(H.degree())
    pos, comm, n_orphan_comp, orphans = _community_layout(H, community or {}, seed)

    marker = {"specimen": "*", "material": "^", "config": "s",
              "acquisition": "o", "endmember": ".", "splatmodel": "D"}
    # marker areas are in points squared and so do not follow the canvas: sizes
    # tuned on a large exploratory canvas read as bloated once the figure is set
    # at a journal column width, so they are scaled by the axes area in inches
    bb = ax.get_window_extent().transformed(
        ax.figure.dpi_scale_trans.inverted())
    scale = max(min((bb.width * bb.height) / 60.0, 1.6), 0.30)
    base = {k: v * scale for k, v in
            {"specimen": 620, "material": 330, "config": 150,
             "acquisition": 60, "endmember": 12, "splatmodel": 130}.items()}
    kcol = {"specimen": OKABE["red"], "material": OKABE["purple"],
            "config": OKABE["orange"], "acquisition": OKABE["blue"],
            "endmember": "#cccccc", "splatmodel": "#333333"}

    et = {e: H.edges[e].get("etype", "?") for e in H.edges}
    groups = [
        ([e for e in H.edges if et[e] == "HAS_ENDMEMBER"], "#dddddd", 0.4, 0.25, None),
        ([e for e in H.edges if et[e] == "SIMILAR_TO"], OKABE["sky"], 0.7, 0.35,
         "spectral similarity"),
        ([e for e in H.edges if et[e] == "DISORDER_NEIGHBOUR"], OKABE["green"], 0.7, 0.30,
         "disorder proximity"),
        ([e for e in H.edges if et[e] == "SHARES_ENDMEMBER"], "#B8860B", 2.4, 0.95,
         "shared endmember"),
        ([e for e in H.edges if et[e] in ("HAS_ACQUISITION", "OF_MATERIAL",
                                          "USED_CONFIG", "MODELLED_BY")],
         "#707070", 1.1, 0.55, "provenance"),
    ]
    for edges, col, w, al, _lab in groups:
        if edges:
            nx.draw_networkx_edges(H, pos, ax=ax, edgelist=edges, edge_color=col,
                                   width=w, alpha=al)

    for k in ("acquisition", "splatmodel", "config", "material", "specimen"):
        nodes = [n for n in H if kind[n] == k]
        if not nodes:
            continue
        if community and k == "acquisition":
            cols = [plt.cm.tab20(comm[n] % 20) if comm.get(n, -1) >= 0 else "#c0c0c0"
                    for n in nodes]
        else:
            cols = kcol.get(k, "#999999")
        nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=nodes, node_color=cols,
                               node_shape=marker.get(k, "o"),
                               node_size=[base[k] + (6 * deg[n] if k == "acquisition" else 0)
                                          for n in nodes],
                               linewidths=0.6, edgecolors="white")

    hubs = sorted((n for n in H if kind[n] == "acquisition"),
                  key=lambda n: -deg[n])[:label_top_hubs]
    show = {n for n in H if kind[n] in label_kinds} | set(hubs)
    # hub labels land on top of each other inside the dense community, so the
    # label anchors are relaxed apart before drawing, and a leader line keeps
    # each one attached to the node it names
    import numpy as np
    show = sorted(show, key=str)
    centre = np.mean([pos[n] for n in H], axis=0)
    anchor = {}
    for n in show:
        if pos[n][1] < -1.2:          # orphan strip: the radial rule aims at the
            anchor[n] = pos[n] + [0, 0.11]    # marker itself, so lift it clear
            continue
        d = pos[n] - centre
        anchor[n] = pos[n] + (d / (np.linalg.norm(d) + 1e-9)) * 0.05
    keys = list(anchor)
    for _ in range(160):
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                v = anchor[a] - anchor[b]
                dist = np.linalg.norm(v)
                if dist < 0.13:
                    push = (v / (dist + 1e-9)) * (0.13 - dist) * 0.5
                    anchor[a] = anchor[a] + push
                    anchor[b] = anchor[b] - push
            anchor[a] = pos[a] + np.clip(anchor[a] - pos[a], -0.22, 0.22)
    for n in show:
        big = kind[n] in ("specimen", "material")
        if np.linalg.norm(anchor[n] - pos[n]) > 0.07:
            ax.plot(*zip(pos[n], anchor[n]), color="#999999", lw=0.5, zorder=5)
        t = ax.text(*anchor[n], str(n).replace("acq-", ""),
                    fontsize=7.0 if big else 6.2, color=INK, ha="center",
                    va="center", fontweight="bold" if big else "normal", zorder=6)
        t.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])

    if orphans:
        n_acq = sum(kind.get(n) == "acquisition" for n in orphans)
        ax.text(0, -1.50, f"{n_acq} acquisitions in {n_orphan_comp} components that "
                          f"never link to the main component", ha="center",
                fontsize=6.5, color="#777777")

    counts = {k: sum(v == k for v in kind.values()) for k in set(kind.values())}
    ms_scale = max(min(scale, 1.0), 0.62)
    handles = [plt.Line2D([], [], marker=marker[k], color="none", markerfacecolor=kcol[k],
                          markeredgecolor="white", markersize=ms * ms_scale,
                          label=f"{k} ({counts[k]})")
               for k, ms in (("specimen", 13), ("material", 10), ("config", 7.5),
                             ("acquisition", 6.5), ("splatmodel", 7.5))
               if counts.get(k)]
    handles += [plt.Line2D([], [], color=c, lw=max(w, 1.0), label=f"{lab} ({len(e)})")
                for e, c, w, _a, lab in groups if lab and e]
    ax.legend(handles=handles, loc="upper left", fontsize=6.0, ncol=2,
              frameon=True, framealpha=0.92, borderpad=0.4, labelspacing=0.35,
              handlelength=1.3, handletextpad=0.45, columnspacing=0.9,
              bbox_to_anchor=(-0.015, 1.015))
    ax.set_axis_off()
    ax.margins(0.05)
    if path:
        ax.figure.savefig(path, dpi=320, bbox_inches="tight")
        plt.close(ax.figure)
    return ax
