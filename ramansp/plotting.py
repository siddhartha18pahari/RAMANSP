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


def graph(G, ax=None, seed: int = 0, community: dict | None = None, path: str | None = None):
    import matplotlib.pyplot as plt
    import networkx as nx

    ax = ax or plt.gca()
    pos = nx.spring_layout(G, seed=seed, k=1.6 / max(len(G) ** 0.5, 1))
    kinds = [G.nodes[n].get("ntype", "?") for n in G]
    palette = {k: c for k, c in zip(sorted(set(kinds)), plt.cm.tab10.colors)}
    if community:
        colors = [plt.cm.tab20(community[n] % 20) if n in community else (0.8, 0.8, 0.8, 1.0)
                  for n in G]
    else:
        colors = [palette[k] for k in kinds]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=90, linewidths=0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
    ax.axis("off")
    if path:
        ax.figure.savefig(path, dpi=320, bbox_inches="tight")
        plt.close(ax.figure)
    return ax
