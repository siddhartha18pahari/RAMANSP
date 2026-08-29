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
    fig_denoise(wn, target, noisy, methods, per_spec, base, ref, tests, aid)
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


def fig_denoise(wn, target, noisy, methods, per_spec, base, ref, tests, aid):
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], hspace=0.42, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])
    k = 0
    ax.plot(wn, noisy[k], lw=0.9, color="#8f6fb0", label="input (noise added)")
    ax.plot(wn, target[k], lw=1.6, color="#2f8f4e", label="target (original)")
    show = [n for n in methods if n.startswith("SG")][:1] + \
           [n for n in methods if not n.startswith("SG")]
    for name, col in zip(show, ["#3b76af", "#d8a13a", "#c0392b"]):
        ax.plot(wn, methods[name][k], lw=1.2, color=col, label=name)
    ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("intensity (a.u.)")
    ax.set_title(f"(a) denoising a contaminated spectrum, map {aid}", fontsize=10)
    ax.legend(fontsize=7.5, ncol=3)

    names = list(methods)
    for c, metric in enumerate(("MSE", "SAD", "SID")):
        ax = fig.add_subplot(gs[1, c])
        vals = [np.mean(per_spec[n][metric]) for n in names]
        errs = [np.std(per_spec[n][metric]) / np.sqrt(len(per_spec[n][metric]))
                for n in names]
        cols = ["#c0392b" if "S3DGS" in n else "#d8a13a" if n.startswith("PCA")
                else "#3b76af" for n in names]
        y = np.arange(len(names))
        ax.barh(y, vals, xerr=errs, color=cols, capsize=2.5)
        ax.axvline(np.mean(base[metric]), color="#8f6fb0", ls="--", lw=1.2)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(metric + "  (lower is better)", fontsize=8)
        if c == 0:
            ax.set_title("(b) six SG filters, PCA truncation and the "
                         "Gaussian field\ndashed line = the noisy input",
                         fontsize=9, loc="left")
        for i, n in enumerate(names):
            t = next((t for t in tests if t["method"] == n and t["metric"] == metric),
                     None)
            if t:
                ax.text(vals[i] + errs[i], i, " " + t["stars"], va="center",
                        fontsize=6.5, color="0.25")
    fig.savefig(FIGS / "fig_ml_denoise.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


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
        y_pm.append(m[idx].astype(int))
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
    ok = [r for r in res if r]
    if not ok:
        return
    fig = plt.figure(figsize=(12, 4.6 + 3.4))
    gs = fig.add_gridspec(2, len(ok) + 1, height_ratios=[1.35, 1.0], hspace=0.55,
                          wspace=0.45)

    ax = fig.add_subplot(gs[0, :])
    first = ok[0]
    names = [r["model"] for r in first["rows"]]
    vals = [r["accuracy"] for r in first["rows"]]
    ax.bar(np.arange(len(names)), vals, color="#3b76af")
    if ok[0]["rows"]:
        ax.axhline(next((r["accuracy"] for r in first["rows"]
                         if r["model"].startswith("Dummy")), np.nan),
                   color="crimson", ls="--", lw=1.2, label="majority-class baseline")
        ax.legend(fontsize=8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=68, ha="right", fontsize=6.6)
    ax.set_ylabel("accuracy on held-out maps")
    ax.set_title(f"(c) benchmark of {len(names)} models, task: {first['task']}",
                 fontsize=10)
    ax.set_ylim(0, 1)

    for c, r in enumerate(ok):
        ax = fig.add_subplot(gs[1, c])
        cm = np.array(r["cm"], float)
        cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(r["labels"])))
        ax.set_yticks(range(len(r["labels"])))
        lab = [str(x)[:9] for x in r["labels"]]
        ax.set_xticklabels(lab, rotation=90, fontsize=6)
        ax.set_yticklabels(lab, fontsize=6)
        acc = max(x["accuracy"] for x in r["rows"])
        ax.set_title(f"({'de'[c] if c < 2 else 'f'}) {r['task']}\n{r['best']}, "
                     f"{100*acc:.2f}%", fontsize=8)
        ax.set_xlabel("predicted", fontsize=7); ax.set_ylabel("true", fontsize=7)
        if len(r["labels"]) <= 6:
            for i in range(cmn.shape[0]):
                for j in range(cmn.shape[1]):
                    ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if cmn[i, j] > 0.55 else "0.2")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(FIGS / "fig_ml_benchmark.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


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
    res = []
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
            fig_benchmark(res)
    dump_json({"denoise": den,
               "classification": [{k: v for k, v in r.items() if k != "rows"} |
                                  {"top5": r["rows"][:5]} for r in res]},
              ML / "summary.json")
    print(f"wrote {ML / 'summary.json'}")


if __name__ == "__main__":
    main()
