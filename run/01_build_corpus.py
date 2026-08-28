"""Stage 1 -- read every machine-readable acquisition, anonymise it, and write
a compact corpus.

Outputs (under ``OUTPUT FILES/``):
    corpus/manifest.csv      one row per acquisition, identifiers already scrubbed
    corpus/records.pkl       full records (incl. mean spectra / endmembers) for
                             the later stages
    corpus/arrays/<id>.npz   preprocessed cube / mean spectrum per acquisition
    _private/sample_key.csv  code -> original token   (NEVER distribute; git-ignored)

Nothing outside ``_private/`` contains a real specimen id, material name,
person name or dated-session folder.
"""

from __future__ import annotations

import collections
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import ARRAYS, CORPUS, INPUT, PRIVATE, dump_json, rel_to_input  # noqa: E402

from ramansp import analysis, io, preprocessing  # noqa: E402
from ramansp._anonymize import IdentityMap  # noqa: E402
from ramansp.containers import Spectrum  # noqa: E402
from ramansp.metrics import effective_n  # noqa: E402

SKIP_EXT = {".l6pf", ".h5", ".png", ".jpg", ".jpeg",
            ".tif", ".tiff", ".bmp", ".docx", ".doc", ".pdf", ".lst", ".ngv", ".ngs",
            ".ngc", ".xlsx"}
# LabSpec 6 binaries, read directly by ramansp.io.read_l6
L6_EXT = {".l6m", ".l6s", ".l6v", ".l6i"}
# cache the un-baselined cube (for the preprocessing ablation) only for maps
# small enough that a second copy is cheap
RAW_CACHE_MAX = 2_000_000

# Material class carried into published outputs.
#
# Deliberate split, not an oversight. Glassy carbon and graphite are *public
# reference standards* -- naming them is scientifically useful and discloses
# nothing about the study, so they keep their names. The remaining entries are
# the originating lab's own material designations, so they are opaque here and
# resolvable only through _private/sample_key.csv.
CLASS_OF = {
    "REF-01": "glassy_carbon",     # public reference standard
    "REF-02": "graphite",          # public reference standard
    "MAT-A": "MAT-A",
    "MAT-B": "MAT-B",
    "METH-A": "fast_screen",
}
PUBLIC_CLASS_NAMES = {"glassy_carbon", "graphite"}


def classify_txt(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
        second = fh.readline()
    low = first.lower()
    if "g intensity" in low and "d3" in low:
        return "fit_table"
    if "note" in path.name.lower() or first.strip().lower().startswith(("this is", "from ", "note ")):
        return "note"
    header_nums = [t for t in first.split("\t") if t.strip()]
    try:
        vals = [float(t) for t in header_nums]
        many = len(vals) >= 16
    except ValueError:
        many = False
    if many:
        return "matrix_like"
    if len(first.split("\t")) == 2 or len(first.split()) == 2:
        return "two_column"
    return "unknown"


def spectrum_features(wn, y):
    d = {"wn_min": float(wn.min()), "wn_max": float(wn.max()), "n_channels": int(wn.size)}
    for lab, (lo, hi) in {"D": (1330, 1370), "G": (1560, 1610)}.items():
        k = (wn >= lo) & (wn <= hi)
        d[f"{lab}_peak"] = float(np.nanmax(y[k])) if k.any() else np.nan
    if d["G_peak"] and np.isfinite(d["G_peak"]) and d["G_peak"] != 0:
        d["D_over_G_peak"] = d["D_peak"] / d["G_peak"]
    return d


def process_fit_table(path, codes):
    t = io.read_fit_table(str(path))
    P = t["params"]
    col = {c: P[:, i] for i, c in enumerate(t["columns"])}
    qc = io.fit_table_qc(t)
    both = (col["G_A"] > 0) & (col["D3_A"] > 0)
    disorder = (col["D3_A"][both] / (col["G_A"][both] + col["D3_A"][both])) if both.any() else np.array([])
    feats = {
        "n_points": t["n_points"],
        "frac_g_ok": float(qc["g_ok"].mean()),
        "frac_pinned": float(qc["pinned"].mean()),
        "g_width_median": float(np.median(col["G_W"][col["G_I"] > 0])) if (col["G_I"] > 0).any() else np.nan,
        "g_area_median": float(np.median(col["G_A"][col["G_A"] > 0])) if (col["G_A"] > 0).any() else np.nan,
        "d3_area_median": float(np.median(col["D3_A"][col["D3_A"] > 0])) if (col["D3_A"] > 0).any() else np.nan,
        "disorder_median": float(np.median(disorder)) if disorder.size else np.nan,
    }
    import hashlib
    rec = dict(kind="fit_table", n_points=t["n_points"], protocol="instrument_5band_D3vsG",
               features=feats, arrays={},
               content_sha1=hashlib.sha1(np.ascontiguousarray(P).tobytes()).hexdigest()[:12])
    return rec


def process_matrix(path, codes):
    try:
        img = io.read_matrix_map(str(path))
    except ValueError:
        qm = io.read_quickmap(str(path))
        return dict(kind="quickmap", n_points=int(qm.size), protocol=None,
                    features={"n_points": int(qm.size)}, arrays={"quickmap": qm.astype(np.float32)})
    return process_image(img, source_format="matrix_text")


def process_l6(path, codes):
    """A LabSpec 6 binary, read directly -- no vendor text export needed."""
    obj = io.read_l6(str(path))
    if isinstance(obj, Spectrum):
        return dict(kind="spectrum", n_points=int(obj.wavenumber.size), protocol=None,
                    features=spectrum_features(obj.wavenumber, obj.intensities),
                    arrays={"mean_spectrum": obj.intensities.astype(np.float32),
                            "wavenumber": obj.wavenumber.astype(np.float32)},
                    source_format="labspec6")
    return process_image(obj, source_format="labspec6")


def process_image(img, source_format="matrix_text"):
    """Shared analysis for any SpectralImage, however it was loaded."""
    lo = max(1000.0, float(img.wavenumber[0]) + 1)
    hi = min(1800.0, float(img.wavenumber[-1]) - 1)
    if hi - lo < 200:
        raise ValueError(f"spectral range {img.wavenumber[0]:.0f}-{img.wavenumber[-1]:.0f} "
                         "does not cover the carbon fingerprint window")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = preprocessing.Pipeline(
            preprocessing.Crop(lo, hi), preprocessing.Despike(),
            preprocessing.BaselineARPLS(lam=1e5), preprocessing.SavGol(7, 3),
            preprocessing.Normalize("area"), name="carbon_dg").apply(img)
        # crop+despike only -- keeps the fluorescence baseline for the
        # preprocessing-ablation figure in stage 4
        raw_crop = preprocessing.Pipeline(
            preprocessing.Crop(lo, hi), preprocessing.Despike()).apply(img)

    gridded = clean.is_gridded
    if gridded:
        det = analysis.detect_particle(img)
        H, W = clean.band_shape
        pmask = det["mask"].reshape(H, W)
        use_mask = 0.02 < pmask.mean() < 0.98
        flat_mask = pmask.ravel() if use_mask else np.ones(H * W, bool)
    else:
        use_mask = False
        flat_mask = np.ones(clean.flat().shape[0], bool)

    sub = clean.flat()[flat_mask]
    mean_spec = sub.mean(0)
    di = analysis.disorder_index(clean)[flat_mask]
    ba = analysis.band_areas(clean)
    try:
        um = analysis.unmix(clean, n_endmembers=4)
        endm = um["endmembers"]
    except Exception:
        endm = None

    feats = {
        "n_points": int(flat_mask.sum()),
        "disorder_median": float(np.nanmedian(di)) if np.isfinite(di).any() else np.nan,
        "d1_area_median": float(np.nanmedian(ba["D1"][flat_mask])),
        "g_area_median": float(np.nanmedian(ba["G+D2"][flat_mask])),
        "total_counts_median": float(np.median(img.flat().mean(1))),
    }
    if gridded and use_mask:
        feats["n_eff"] = float(effective_n(ba["G+D2"].reshape(H, W), pmask)["n_eff"])

    arrays = {"mean_spectrum": mean_spec.astype(np.float32),
              "wavenumber": clean.wavenumber.astype(np.float32)}
    if endm is not None:
        arrays["endmembers"] = endm.astype(np.float32)
        arrays["endmember_wn"] = clean.wavenumber.astype(np.float32)
    if gridded:
        arrays["cube"] = clean.intensities.astype(np.float32)
        if clean.intensities.size <= RAW_CACHE_MAX:
            arrays["cube_raw"] = raw_crop.intensities.astype(np.float32)
        arrays["particle_mask"] = (pmask if use_mask else np.ones((H, W), bool))

    feats["n_channels"] = int(clean.wavenumber.size)
    if gridded:
        feats["nx"], feats["ny"] = int(W), int(H)
    return dict(kind="raw_map" if gridded else "raw_scan",
                n_points=int(flat_mask.sum()), protocol="carbon_dg",
                features=feats, arrays=arrays, source_format=source_format)


def process_two_column(path, codes, autocal=False):
    sp = io.read_autocal(str(path)) if autocal else io.read_single(str(path))
    feats = spectrum_features(sp.wavenumber, sp.intensities)
    return dict(kind="autocal" if autocal else "spectrum",
                n_points=int(sp.wavenumber.size), protocol=None, features=feats,
                arrays={"mean_spectrum": sp.intensities.astype(np.float32),
                        "wavenumber": sp.wavenumber.astype(np.float32)})


def main():
    print(f"scanning {INPUT}")
    idmap = IdentityMap.from_tree(str(INPUT))

    banner = ("# NOT FOR DISTRIBUTION -- maps anonymised codes back to real "
              "specimen ids / names.\n")
    key = pd.DataFrame(idmap.key_rows())
    (PRIVATE / "sample_key.csv").write_text(banner + key.to_csv(index=False))
    print(f"  identity map: {len(idmap.specimen)} specimens, {len(idmap.config)} configs, "
          f"{len(idmap.session)} sessions -> _private/sample_key.csv")

    files = sorted(p for p in INPUT.rglob("*") if p.is_file())
    records: list[dict] = []
    n_skip = 0
    n_l6_fail = 0
    l6_reasons: collections.Counter = collections.Counter()
    for p in files:
        ext = p.suffix.lower()
        if ext in SKIP_EXT:
            n_skip += 1
            continue
        codes = idmap.code_for_path(rel_to_input(p))
        try:
            if ext == ".spc":
                try:
                    sp = io.read_spc(str(p))
                except (NotImplementedError, ValueError) as e:
                    print(f"  skip {rel_to_input(p)}: {e}")
                    n_skip += 1
                    continue
                rec = dict(kind="spectrum", n_points=int(sp.wavenumber.size),
                           protocol=None,
                           features=spectrum_features(sp.wavenumber, sp.intensities),
                           arrays={"mean_spectrum": sp.intensities.astype(np.float32),
                                   "wavenumber": sp.wavenumber.astype(np.float32)})
            elif ext in L6_EXT:
                try:
                    rec = process_l6(p, codes)
                except (ValueError, NotImplementedError, MemoryError) as e:
                    n_l6_fail += 1
                    l6_reasons[str(e).split(": ", 1)[-1][:52]] += 1
                    continue
            elif ext == ".txt":
                cls = classify_txt(p)
                if cls == "note":
                    n_skip += 1
                    continue
                if cls == "fit_table":
                    rec = process_fit_table(p, codes)
                elif cls == "matrix_like":
                    rec = process_matrix(p, codes)
                elif cls == "two_column":
                    rec = process_two_column(p, codes, autocal="autocalibration" in p.name.lower())
                else:
                    n_skip += 1
                    continue
            else:
                n_skip += 1
                continue
        except Exception as e:  # noqa: BLE001  -- corpus scan must not die on one file
            print(f"  FAILED {rel_to_input(p)}: {type(e).__name__}: {e}")
            n_skip += 1
            continue

        acq_id = f"acq-{len(records):04d}"
        material = codes["material"]
        rec.update(
            acq_id=acq_id,
            specimen=codes["specimen"], material=material,
            config=codes["config"], session=codes["session"],
            carbon_class=CLASS_OF.get(material, "particle" if codes["specimen"] else "?"),
            source_rel=idmap.scrub(rel_to_input(p)),
        )
        arrays = rec.pop("arrays", {})
        if arrays:
            np.savez_compressed(ARRAYS / f"{acq_id}.npz", **arrays)
            for k in ("mean_spectrum", "wavenumber", "endmembers", "endmember_wn"):
                if k in arrays:
                    rec[k] = arrays[k]
        records.append(rec)
        print(f"  {acq_id}  {rec['kind']:10s}  {rec.get('source_rel')}")

    with open(CORPUS / "records.pkl", "wb") as fh:
        pickle.dump(records, fh)

    man = pd.DataFrame([{
        k: r.get(k) for k in ("acq_id", "kind", "specimen", "material", "config",
                              "session", "carbon_class", "n_points", "protocol",
                              "source_format", "content_sha1", "source_rel")
    } | {f"feat_{fk}": fv for fk, fv in (r.get("features", {}) or {}).items()
         if isinstance(fv, (int, float))} for r in records])
    man.to_csv(CORPUS / "manifest.csv", index=False)
    dup = man.dropna(subset=["content_sha1"]).groupby("content_sha1").acq_id.apply(list)
    dup = {h: v for h, v in dup.items() if len(v) > 1}
    if dup:
        print(f"\n  {sum(len(v) for v in dup.values())} fit-table exports are "
              f"byte-identical duplicates in {len(dup)} groups: {list(dup.values())}")

    summary = {
        "n_acquisitions": len(records),
        "by_kind": man.kind.value_counts().to_dict(),
        "n_specimens": int(man.specimen.nunique(dropna=True)),
        "n_configs": int(man.config.nunique(dropna=True)),
        "files_skipped": n_skip,
        "l6_binaries_read": int((man.source_format == "labspec6").sum()),
        "l6_binaries_unreadable": n_l6_fail,
        "l6_failure_reasons": dict(l6_reasons.most_common(6)),
        "duplicate_fit_table_groups": [v for v in dup.values()] if dup else [],
    }
    dump_json(summary, CORPUS / "summary.json")
    print("\n" + "\n".join(f"{k}: {v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
