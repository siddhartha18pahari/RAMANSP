"""Stage 7 -- machine-learning benchmarks on the anonymised corpus.

Two experiments, both run on this corpus rather than on published data, since
the reference datasets used by comparable studies are not distributed with this
work. The *designs* follow that literature; the numbers are our own.

(a) Denoising. A high-SNR map is contaminated with additive noise to form the
    input, and the original serves as the target, which is the paired design
    used for held-out denoiser evaluation in the RamanSPy study. We compare six
    Savitzky-Golay filters, a PCA rank-truncation denoiser, and the fitted
    Spectral 3D Gaussian field, on MSE, spectral angle distance (SAD) and
    spectral information divergence (SID). Significance is a two-sided Wilcoxon
    signed-rank test over spectra with Benjamini-Hochberg correction.

(b) Classification. Twenty-eight scikit-learn models are benchmarked on two
    tasks with genuine labels: particle against substrate, and which
    acquisition configuration produced a spectrum. Splits are by *map*, never by
    spectrum, because spectra from one map are not independent samples.

Outputs:
    figures/fig_ml_denoise.png
    figures/fig_ml_benchmark.png
    tables/ml_denoise.csv, tables/ml_models.csv
    OUTPUT FILES/ml/summary.json
"""

from __future__ import annotations

import json
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.simplefilter("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import CORPUS, FIGS, OUTPUT, TABLES, dump_json  # noqa: E402

from ramansp.containers import SpectralImage  # noqa: E402

ML = OUTPUT / "ml"
ML.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)


# ================================================================
#  metrics
# ================================================================
def mse(a, b):
    return np.mean((a - b) ** 2, axis=-1)


def sad(a, b, eps=1e-12):
    """Spectral angle distance, radians (Kruse et al.)."""
    na = np.linalg.norm(a, axis=-1) + eps
    nb = np.linalg.norm(b, axis=-1) + eps
    cos = np.sum(a * b, axis=-1) / (na * nb)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def sid(a, b, eps=1e-10):
    """Spectral information divergence (Chang). Needs non-negative spectra."""
    p = np.clip(a, 0, None) + eps
    q = np.clip(b, 0, None) + eps
    p = p / p.sum(-1, keepdims=True)
    q = q / q.sum(-1, keepdims=True)
    return np.sum(p * np.log(p / q), -1) + np.sum(q * np.log(q / p), -1)


def benjamini_hochberg(pvals):
    try:
        from scipy.stats import false_discovery_control
        return false_discovery_control(pvals, method="bh")
    except Exception:
        p = np.asarray(pvals, float)
        order = np.argsort(p)
        n = p.size
        adj = np.empty(n)
        prev = 1.0
        for rank, i in enumerate(order[::-1]):
            k = n - rank
            prev = min(prev, p[i] * n / k)
            adj[i] = prev
        return adj


def stars(p):
    return ("****" if p < 1e-4 else "***" if p < 1e-3
            else "**" if p < 1e-2 else "*" if p < 0.05 else "ns")


# ================================================================
#  (a) denoising benchmark
# ================================================================
def pick_cleanest_map(man):
    """Highest contrast-to-noise raw map: the one worth corrupting on purpose."""
    best, best_cnr = None, -np.inf
    for aid in man[man.kind == "raw_map"].acq_id:
        d = np.load(CORPUS / "arrays" / f"{aid}.npz")
        if "particle_mask" not in d:
            continue
        cube = d["cube"].astype(float)
        flat = cube.reshape(-1, cube.shape[-1])[d["particle_mask"].ravel()]
        if flat.shape[0] < 200:
            continue
        noise = np.median(np.abs(np.diff(flat, 2, axis=1))) / np.sqrt(6)
        cnr = float(np.ptp(flat, axis=1).mean() / max(noise, 1e-12))
        if cnr > best_cnr:
            best, best_cnr = aid, cnr
    return best, best_cnr


def denoise_benchmark(man, n_spectra=1200, noise_mult=1.5):
    from scipy.signal import savgol_filter
    from scipy.stats import wilcoxon

    aid, cnr = pick_cleanest_map(man)
    if aid is None:
        return None
    d = np.load(CORPUS / "arrays" / f"{aid}.npz")
    cube = d["cube"].astype(float)
    wn = d["wavenumber"].astype(float)
    pmask2d = d["particle_mask"]
    mask = pmask2d.ravel()
    flat = cube.reshape(-1, wn.size)

    # Contaminate *every* masked spectrum, not only those scored. A method that
    # pools information across spectra must see the same corrupted map that a
    # per-spectrum filter does, or the comparison is rigged in its favour.
    sigma = noise_mult * np.median(
        np.std(np.diff(flat[mask], 2, axis=1), axis=1) / np.sqrt(6))
    noisy_full = flat.copy()
    noisy_full[mask] = flat[mask] + RNG.normal(0, sigma, flat[mask].shape)

    idx_all = np.where(mask)[0]
    score_idx = (RNG.choice(idx_all, n_spectra, replace=False)
                 if idx_all.size > n_spectra else idx_all)
    target = flat[score_idx]
    noisy = noisy_full[score_idx]

    methods = {}
    for order, win in [(2, 5), (3, 9), (3, 15), (4, 15), (3, 21), (5, 25)]:
        methods[f"SG({order},{win})"] = savgol_filter(noisy, win, order, axis=1)

    # PCA rank truncation at the Marchenko-Pastur edge, fitted on the whole
    # corrupted map and then read back at the scored spectra
    sub = noisy_full[mask]
    mu = sub.mean(0)
    U, S, Vt = np.linalg.svd(sub - mu, full_matrices=False)
    edge = sigma * (np.sqrt(sub.shape[0]) + np.sqrt(sub.shape[1]))
    rank = max(int((S > edge).sum()), 1)
    basis = Vt[:rank]
    proj = (noisy - mu) @ basis.T @ basis + mu
    methods[f"PCA (rank {rank})"] = proj

    # the fitted Gaussian field, used as a denoiser, at the same primitive
    # density the paper's own fits use
    splat_pred = splat_denoise(noisy_full, cube.shape, wn, pmask2d, score_idx)
    if splat_pred is not None:
        methods["S3DGS (ours)"] = splat_pred

    rows = []
    per_spec = {}
    for name, pred in methods.items():
        m = {"MSE": mse(pred, target), "SAD": sad(pred, target),
             "SID": sid(pred, target)}
        per_spec[name] = m
        rows.append({"method": name, **{k: float(np.mean(v)) for k, v in m.items()},
                     **{k + "_sd": float(np.std(v)) for k, v in m.items()}})
    base = {"MSE": mse(noisy, target), "SAD": sad(noisy, target),
            "SID": sid(noisy, target)}
    rows.insert(0, {"method": "input (noisy)",
                    **{k: float(np.mean(v)) for k, v in base.items()},
                    **{k + "_sd": float(np.std(v)) for k, v in base.items()}})

    # Wilcoxon of every method against the best Savitzky-Golay filter
    sg_names = [n for n in methods if n.startswith("SG")]
    ref = min(sg_names, key=lambda n: per_spec[n]["MSE"].mean())
    tests, raw_p = [], []
    for name in methods:
        if name == ref:
            continue
        for metric in ("MSE", "SAD", "SID"):
            st = wilcoxon(per_spec[name][metric], per_spec[ref][metric],
                          alternative="two-sided")
            tests.append({"method": name, "metric": metric,
                          "median_delta": float(np.median(per_spec[name][metric]
                                                          - per_spec[ref][metric]))})
            raw_p.append(st.pvalue)
    adj = benjamini_hochberg(raw_p)
    for t, p0, p1 in zip(tests, raw_p, adj):
        t["p_raw"], t["p_bh"], t["stars"] = float(p0), float(p1), stars(p1)

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "ml_denoise.csv", index=False)
    fig_denoise(wn, target, noisy, methods, per_spec, base, ref, tests, aid,
                noise_mult_label=f"{noise_mult:g}")
    return {"map": aid, "cnr": cnr, "reference_filter": ref,
            "added_noise_sd": float(sigma), "n_spectra": int(target.shape[0]),
            "splat_rho": getattr(splat_denoise, "rho", None),
            "splat_n": getattr(splat_denoise, "n_primitives", None),
            "table": rows, "tests": tests}


def splat_denoise(noisy_flat, shape, wn, pmask, score_idx, per_1k_voxels=30.0):
    """Fit a field to the corrupted cube and read it back at the scored points.

    The primitive budget is set by the same density rule the rest of the study
    uses, rather than a fixed count, so this is the method as deployed and not
    a starved version of it.
    """
    from ramansp.splatting import fit_image
    from ramansp.splatting.fit import SplatConfig

    try:
        H, W, K = shape
        img = SpectralImage(noisy_flat.reshape(H, W, K), wn, {})
        vox = int(pmask.sum() * K)
        n_g = int(np.clip(round(per_1k_voxels * vox / 1000), 800, 3400))
        cfg = SplatConfig(iters=110, channel_bin=1, hold_out=0,
                          auto_gaussians=False, n_gaussians=n_g,
                          max_gaussians=int(n_g * 1.4), log_every=0)
        print(f"  S3DGS denoiser: {n_g} primitives over {vox} voxels "
              f"({per_1k_voxels:.0f} per 1000) ...")
        field, vol, _ = fit_image(img, spatial_mask=pmask, config=cfg)
        splat_denoise.rho = round(1000 * field.n / max(vox, 1), 1)
        splat_denoise.n_primitives = int(field.n)
        xa, ya, _za = field.world_axes
        rec = field.render_grid((xa, ya, vol.z_train)) * vol.scale
        i0, i1, j0, j1 = vol.bbox
        full = np.full((H, W, vol.z_train.size), np.nan)
        full[i0:i1, j0:j1] = rec
        out = full.reshape(-1, vol.z_train.size)[score_idx]
        if out.shape[1] != K or not np.isfinite(out).all():
            print(f"  (S3DGS denoiser: shape {out.shape} unusable)")
            return None
        return out
    except Exception as e:  # noqa: BLE001
        print(f"  (S3DGS denoiser skipped: {type(e).__name__}: {e})")
        return None


def fig_denoise(wn, target, noisy, methods, per_spec, base, ref, tests, aid,
                noise_mult_label="1.5"):
    """Denoising benchmark, with every bar read against the noisy input.

    Colour previously encoded only which family a method belonged to, so a bar
    that was worse than doing nothing looked the same as one that helped. The
    dashed line is the input, and a method is only useful if its bar ends to
    the left of it, so that is what the shading now says.
    """
    from ramansp._style import OKABE, acs_figsize, apply_style, save
    apply_style()

    fig = plt.figure(figsize=acs_figsize("double", 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.85, 1.25], hspace=0.42,
                          wspace=0.10, left=0.135, right=0.985, top=0.945,
                          bottom=0.085)

    ax = fig.add_subplot(gs[0, :])
    k = 0
    ax.plot(wn, noisy[k], lw=0.9, color="#b9a7d0", label="input (noise added)")
    ax.plot(wn, target[k], lw=1.5, color=OKABE["green"], label="target (original)")
    show = [n for n in methods if n.startswith("SG")][:1] +            [n for n in methods if not n.startswith("SG")]
    for name, col in zip(show, [OKABE["blue"], OKABE["orange"], OKABE["red"]]):
        ax.plot(wn, methods[name][k], lw=1.3, color=col, label=name)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("intensity (a.u.)")
    ax.set_title(f"(a) one contaminated spectrum from map {aid}, and what each "
                 f"method returns", fontsize=8, loc="left")
    ax.legend(ncol=5, loc="upper left", handlelength=1.3, handletextpad=0.5,
              columnspacing=1.0, borderaxespad=0.2)
    ax.text(0.995, 0.015, f"input = target + Gaussian noise at "
                         f"{noise_mult_label}x the map's own noise level",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color="#777777")

    names = list(methods)
    for c, metric in enumerate(("MSE", "SAD", "SID")):
        ax = fig.add_subplot(gs[1, c])
        vals = np.array([np.mean(per_spec[n][metric]) for n in names])
        errs = np.array([np.std(per_spec[n][metric]) / np.sqrt(len(per_spec[n][metric]))
                         for n in names])
        b = float(np.mean(base[metric]))
        cols = [OKABE["blue"] if v < b else "#c9c9c9" for v in vals]
        y = np.arange(len(names))
        ax.barh(y, vals, xerr=errs, color=cols, capsize=2.5,
                error_kw={"lw": 0.9, "ecolor": "0.3"})
        ax.axvline(b, color="#7b5ea7", ls="--", lw=1.3, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(names if c == 0 else [], fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel(metric + "  (lower is better)", fontsize=7.5)
        ax.grid(axis="x", alpha=0.35); ax.set_axisbelow(True)
        n_better = int((vals < b).sum())
        ax.set_title(f"(b{c + 1}) {n_better} of {len(names)} beat the input",
                     fontsize=7.5, loc="left")
        for i, n in enumerate(names):
            t = next((t for t in tests if t["method"] == n and t["metric"] == metric),
                     None)
            if t:
                ax.text(vals[i] + errs[i], i, " " + t["stars"], va="center",
                        fontsize=6.5, color="0.25")
    save(fig, FIGS / "fig_ml_denoise.png")


# ================================================================
#  (b) model benchmark
# ================================================================
def model_zoo(seed=0):
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                               QuadraticDiscriminantAnalysis)
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                                  ExtraTreesClassifier, GradientBoostingClassifier,
                                  HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.linear_model import (LogisticRegression, PassiveAggressiveClassifier,
                                      Perceptron, RidgeClassifier, SGDClassifier)
    from sklearn.naive_bayes import BernoulliNB, GaussianNB
    from sklearn.neighbors import (KNeighborsClassifier, NearestCentroid,
                                   RadiusNeighborsClassifier)
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC, LinearSVC, NuSVC
    from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.semi_supervised import LabelPropagation, LabelSpreading

    s = dict(random_state=seed)
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, **s),
        "RidgeClassifier": RidgeClassifier(**s),
        "LinearSVC": LinearSVC(max_iter=4000, **s),
        "SVC (RBF)": SVC(**s),
        "NuSVC": NuSVC(**s),
        "SGDClassifier": SGDClassifier(max_iter=2000, **s),
        "PassiveAggressive": PassiveAggressiveClassifier(max_iter=2000, **s),
        "Perceptron": Perceptron(max_iter=2000, **s),
        "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, **s),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, **s),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, n_jobs=-1, **s),
        "HistGradientBoosting": HistGradientBoostingClassifier(**s),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=60, **s),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, **s),
        "Bagging": BaggingClassifier(n_estimators=50, n_jobs=-1, **s),
        "DecisionTree": DecisionTreeClassifier(**s),
        "ExtraTree": ExtraTreeClassifier(**s),
        "KNN (k=5)": KNeighborsClassifier(5, n_jobs=-1),
        "KNN (k=15)": KNeighborsClassifier(15, n_jobs=-1),
        "NearestCentroid": NearestCentroid(),
        "RadiusNeighbors": RadiusNeighborsClassifier(radius=12.0, outlier_label="most_frequent"),
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "GaussianProcess": GaussianProcessClassifier(max_iter_predict=40, **s),
        "LabelPropagation": LabelPropagation(),
        "LabelSpreading": LabelSpreading(),
        "CalibratedLinearSVC": CalibratedClassifierCV(LinearSVC(max_iter=3000, **s)),
        "Dummy (majority)": DummyClassifier(strategy="most_frequent"),
    }


def build_tasks(man, per_map=700, n_components=30):
    """Feature matrices with map-level grouping, for leakage-free splits."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X, y_pm, y_cfg, grp = [], [], [], []
    common = np.arange(1000.0, 1800.0, 3.0)
    for _, row in man[man.kind == "raw_map"].iterrows():
        aid = row.acq_id
        p = CORPUS / "arrays" / f"{aid}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        if "particle_mask" not in d:
            continue
        cube = d["cube"].astype(float)
        wn = d["wavenumber"].astype(float)
        flat = cube.reshape(-1, wn.size)
        m = d["particle_mask"].ravel()
        if m.sum() < 60 or (~m).sum() < 60:
            continue
        idx = RNG.permutation(flat.shape[0])[:per_map]
        F = np.vstack([np.interp(common, wn, flat[i]) for i in idx])
        n = np.linalg.norm(F, axis=1, keepdims=True)
        X.append(F / np.clip(n, 1e-12, None))
        y_pm.append(np.where(m[idx], "particle", "substrate"))
        y_cfg.append(np.repeat(str(row.get("config")), len(idx)))
        grp.append(np.repeat(aid, len(idx)))
    if not X:
        return None
    X = np.vstack(X)
    Z = PCA(n_components=n_components, random_state=0).fit_transform(
        StandardScaler().fit_transform(X))
    return dict(X=Z, y_particle=np.concatenate(y_pm),
                y_config=np.concatenate(y_cfg), group=np.concatenate(grp))


# Models whose cost is quadratic or cubic in the training set. They stay in the
# benchmark, but on a capped subsample, and the cap is recorded alongside the
# score so the comparison is not silently unequal.
SLOW_MODELS = {"GaussianProcess", "LabelPropagation", "LabelSpreading",
               "NuSVC", "SVC (RBF)", "RadiusNeighbors", "CalibratedLinearSVC"}
SLOW_CAP = 1200


def run_task(data, y, name, max_train=2500, max_test=2500):
    """Split by map, never by spectrum."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    groups = data["group"]
    keep = pd.Series(y).astype(str).values != "nan"
    X, y, groups = data["X"][keep], np.asarray(y)[keep].astype(str), groups[keep]
    uniq_lab = np.unique(y)
    if uniq_lab.size < 2:
        return None
    # hold out whole maps, ensuring the test set still contains >1 class
    maps = np.unique(groups)
    rng = np.random.default_rng(1)
    for _ in range(60):
        rng.shuffle(maps)
        n_test = max(1, int(round(0.3 * len(maps))))
        te_maps = set(maps[:n_test])
        te = np.isin(groups, list(te_maps))
        tr = ~te
        if (np.unique(y[tr]).size >= 2 and np.unique(y[te]).size >= 2
                and set(np.unique(y[te])) <= set(np.unique(y[tr]))):
            break
    else:
        return None
    if tr.sum() > max_train:
        sel = rng.choice(np.where(tr)[0], max_train, replace=False)
        tr = np.zeros_like(tr); tr[sel] = True
    if te.sum() > max_test:
        sel = rng.choice(np.where(te)[0], max_test, replace=False)
        te = np.zeros_like(te); te[sel] = True

    tr_idx = np.where(tr)[0]

    def fit_idx(mname):
        if mname in SLOW_MODELS and tr_idx.size > SLOW_CAP:
            return rng.choice(tr_idx, SLOW_CAP, replace=False)
        return tr_idx

    rows = []
    for mname, clf in model_zoo().items():
        ti = fit_idx(mname)
        try:
            clf.fit(X[ti], y[ti])
            pred = clf.predict(X[te])
            rows.append({"task": name, "model": mname,
                         "accuracy": float(accuracy_score(y[te], pred)),
                         "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
                         "f1_weighted": float(f1_score(y[te], pred, average="weighted")),
                         "n_fit": int(ti.size)})
        except Exception:
            continue
    rows.sort(key=lambda r: -r["accuracy"])
    best = rows[0]["model"] if rows else None
    cm, labels = None, None
    if best:
        from sklearn.metrics import confusion_matrix
        clf = model_zoo()[best]
        ti = fit_idx(best)
        clf.fit(X[ti], y[ti])
        pred = clf.predict(X[te])
        labels = sorted(np.unique(np.concatenate([y[te], pred])))
        cm = confusion_matrix(y[te], pred, labels=labels).tolist()
    return {"task": name, "rows": rows, "best": best, "cm": cm, "labels": labels,
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "n_train_maps": int(len(np.unique(groups[tr]))),
            "n_test_maps": int(len(np.unique(groups[te])))}


def fig_benchmark(res):
    """Both tasks, every model, each read against its own majority baseline.

    The earlier version plotted only the first task and coloured every bar the
    same, so a model scoring 0.72 looked strong without the reader being told
    that always predicting the majority class already scores 0.65. Accuracy on
    an unbalanced task is meaningless without that reference, so the baseline
    is drawn, annotated, and used to colour the bars.
    """
    from ramansp._style import (ACS_ART_DEPTH_IN, OKABE, acs_figsize,
                                apply_style, panel_tag, save)
    apply_style()

    ok = [r for r in res if r]
    if not ok:
        return {}
    n_rows = max(len(r["rows"]) for r in ok)
    # 30 named bars need roughly 0.115 in each to stay legible at 6 pt, and the
    # confusion matrices take what the journal's depth limit leaves
    bar_h = min(0.115 * n_rows, ACS_ART_DEPTH_IN - 2.9)
    fig = plt.figure(figsize=acs_figsize("double", bar_h + 2.9))
    gs = fig.add_gridspec(2, len(ok), height_ratios=[bar_h, 2.5],
                          hspace=0.42, wspace=0.52, left=0.175, right=0.965,
                          top=0.935, bottom=0.125)

    summary = {}
    for c, r in enumerate(ok):
        ax = fig.add_subplot(gs[0, c])
        rows = sorted(r["rows"], key=lambda x: x["accuracy"])
        base = next((x["accuracy"] for x in rows if x["model"].startswith("Dummy")), np.nan)
        names = [x["model"] for x in rows]
        vals = np.array([x["accuracy"] for x in rows], float)
        beat = vals > base + 1e-12
        cols = [OKABE["blue"] if b else "#c4c4c4" for b in beat]
        for i, x in enumerate(rows):
            if x["model"].startswith("Dummy"):
                cols[i] = OKABE["red"]
        ax.barh(np.arange(len(names)), vals, color=cols, height=0.74)
        ax.axvline(base, color=OKABE["red"], ls="--", lw=1.3, zorder=4)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=6.0)
        ax.set_xlim(0, 1)
        ax.set_xlabel("accuracy on held-out maps", fontsize=7.5)
        # below the axis this label sat on top of the tick labels; it belongs
        # next to the bar it describes
        di = next((i for i, x in enumerate(rows) if x["model"].startswith("Dummy")), 0)
        ax.text(base + 0.02, di, f"majority baseline {base:.3f}",
                color=OKABE["red"], fontsize=6.5, va="center")
        # label the bar belonging to the model named in the confusion matrix,
        # not merely the topmost bar: ties are common here and labelling the
        # wrong one of two equal bars contradicts the panel below
        best = max(vals)
        bi = next((i for i, nm in enumerate(names) if nm == r["best"]),
                  int(np.argmax(vals)))
        ax.text(vals[bi] + 0.015, bi, f"{vals[bi]:.3f}", fontsize=6.5,
                va="center", fontweight="bold", color=OKABE["blue"])
        # with the canvas fixed at the column width a long left-aligned title
        # runs off the panel, so the counts go on their own shorter line
        ax.set_title(f"{r['task']}" + chr(10) +
                     f"{int(beat.sum())}/{len(names)} beat the baseline, "
                     f"{r['n_train_maps']} train and {r['n_test_maps']} test maps",
                     fontsize=7.0, loc="left")
        panel_tag(ax, "ab"[c] if c < 2 else "c", dx=-0.27, dy=1.12, size=9)
        ax.grid(axis="x", alpha=0.35)
        ax.set_axisbelow(True)
        cmr = np.array(r["cm"], float)
        summary[r["task"]] = {"baseline": float(base), "best": float(best),
                              "best_model": r["best"], "n_beat": int(beat.sum()),
                              "n_models": len(names),
                              "n_classes": int(cmr.shape[0]),
                              "n_classes_tested": int((cmr.sum(1) > 0).sum())}

        axc = fig.add_subplot(gs[1, c])
        axc.grid(False)
        cm = np.array(r["cm"], float)
        # a square matrix over all classes leaves an all-zero row for every
        # class the held-out maps happen not to contain, and a blank row reads
        # as a model failure rather than as an absent class. Rows are therefore
        # the classes actually present in the truth, columns stay the full label
        # set so predictions into absent classes remain visible, and each row
        # carries the count it is normalised by
        support = cm.sum(1)
        rows_keep = np.flatnonzero(support > 0)
        n_drop = cm.shape[0] - rows_keep.size
        cm = cm[rows_keep, :]
        cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        im = axc.imshow(cmn, cmap="Blues", vmin=0, vmax=1, interpolation="nearest",
                        aspect="auto")
        xlab = [str(x)[:10] for x in r["labels"]]
        ylab = [f"{str(r['labels'][i])[:10]}  (n={int(support[i])})" for i in rows_keep]
        lab = ylab
        axc.set_xticks(range(len(xlab))); axc.set_yticks(range(len(ylab)))
        axc.set_xticklabels(xlab, rotation=90, fontsize=6.2)
        axc.set_yticklabels(ylab, fontsize=6.2)
        if n_drop:
            axc.set_xlabel(f"predicted   ({n_drop} of {len(r['labels'])} classes "
                           f"have no example in the held-out maps)", fontsize=7.0)
        else:
            axc.set_xlabel("predicted", fontsize=7.5)
        acc = max(x["accuracy"] for x in r["rows"])
        axc.set_title(f"{r['best']}, {100 * acc:.2f}%" + chr(10) +
                      f"(baseline {100 * base:.2f}%)", fontsize=7.5)
        axc.set_ylabel("true", fontsize=7.5)
        panel_tag(axc, "cd"[c] if c < 2 else "e", dx=-0.24, dy=1.16, size=9)
        if cmn.size <= 144:
            for i in range(cmn.shape[0]):
                for j in range(cmn.shape[1]):
                    # a wide matrix at this width cannot carry a number in
                    # every cell without them running together, so only the
                    # cells that matter are labelled
                    floor = 0.05 if cmn.shape[1] > 6 else 0.004
                    if cmn[i, j] > floor:
                        axc.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                                 fontsize=6.0,
                                 color="white" if cmn[i, j] > 0.55 else "0.25")
        cb = fig.colorbar(im, ax=axc, shrink=0.78)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("row-normalised", fontsize=6.5)

    save(fig, FIGS / "fig_ml_benchmark.png")
    return summary


def main():
    man = pd.read_csv(CORPUS / "manifest.csv")
    print("=== (a) denoising benchmark ===")
    den = denoise_benchmark(man)
    if den:
        print(f"  map {den['map']} (CNR {den['cnr']:.0f}), reference {den['reference_filter']}")
        for r in den["table"]:
            print(f"   {r['method']:<18} MSE {r['MSE']:.3e}  SAD {r['SAD']:.4f}  SID {r['SID']:.4e}")

    print("=== (b) model benchmark ===")
    data = build_tasks(man)
    res, bench = [], {}
    if data is not None:
        for key, label in (("y_particle", "particle vs substrate"),
                           ("y_config", "acquisition configuration")):
            r = run_task(data, data[key], label)
            if r:
                res.append(r)
                print(f"  {label}: best {r['best']} "
                      f"{100*r['rows'][0]['accuracy']:.2f}%  "
                      f"(train {r['n_train_maps']} maps, test {r['n_test_maps']} maps)")
        if res:
            pd.DataFrame([x for r in res for x in r["rows"]]).to_csv(
                TABLES / "ml_models.csv", index=False)
            bench = fig_benchmark(res)
            for r in res:
                print(f"    baseline {bench[r['task']]['baseline']:.4f} -> best "
                      f"{bench[r['task']]['best']:.4f} "
                      f"({bench[r['task']]['n_beat']}/"
                      f"{bench[r['task']]['n_models']} beat it)")
    dump_json({"denoise": den, "baselines": bench,
               "classification": [{k: v for k, v in r.items() if k != "rows"} |
                                  {"top5": r["rows"][:5]} for r in res]},
              ML / "summary.json")
    print(f"wrote {ML / 'summary.json'}")


if __name__ == "__main__":
    main()
