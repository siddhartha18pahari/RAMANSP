"""The differentiable Gaussian field and its analytic gradients.

Field
-----
``V_hat(p) = sum_i a_i * exp(-0.5 (p - mu_i)^T P_i (p - mu_i))``

with, for Gaussian ``i``:
    ``mu_i``      centre in the unit cube               (3,)
    ``s_i``       per-axis std dev, ``s = exp(log_s)``     (3,)
    ``q_i``       rotation quaternion (w, x, y, z), normalised on use   (4,)
    ``a_i``       amplitude, ``a = exp(log_a)``          ()
    ``P_i``       = R(q_i) diag(1/s_i^2) R(q_i)^T        (precision)

Loss
----
``L = 0.5 * sum_{p in mask} (V_hat(p) - V(p))^2
      +  lam_a * sum_i a_i
      +  lam_s * sum_i ||log_s_i||^2``

``loss_and_grad`` returns closed-form gradients for every parameter;
``tests/test_splatting.py`` checks them against central finite differences.
The forward and backward passes are dense but **chunked over voxels** and
vectorised over Gaussians with BLAS ``matmul`` -- fast enough for ~1e3-1e4
Gaussians on a CPU. The GPU/tiled path for millions is in ``torch_backend``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .volume import Volume


def _dR_dq(qn: np.ndarray) -> np.ndarray:
    """d R / d q_n for a normalised quaternion q_n = (w, x, y, z). Returns (4,3,3)."""
    w, x, y, z = qn
    dRdw = np.array([[0, -2 * z, 2 * y], [2 * z, 0, -2 * x], [-2 * y, 2 * x, 0]], float)
    dRdx = np.array([[0, 2 * y, 2 * z], [2 * y, -4 * x, -2 * w], [2 * z, 2 * w, -4 * x]], float)
    dRdy = np.array([[-4 * y, 2 * x, 2 * w], [2 * x, 0, 2 * z], [-2 * w, 2 * z, -4 * y]], float)
    dRdz = np.array([[-4 * z, -2 * w, 2 * x], [2 * w, -4 * z, 2 * y], [2 * x, 2 * y, 0]], float)
    return np.stack([dRdw, dRdx, dRdy, dRdz])


def _dR_dq_batch(qn: np.ndarray) -> np.ndarray:
    """Batched version of :func:`_dR_dq`. ``qn`` (N,4) -> (N,4,3,3)."""
    w, x, y, z = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    O = np.zeros_like(w)
    z2, x2, y2, w2 = 2 * z, 2 * x, 2 * y, 2 * w
    dRdw = np.stack([np.stack([O, -z2, y2], -1),
                     np.stack([z2, O, -x2], -1),
                     np.stack([-y2, x2, O], -1)], -2)
    dRdx = np.stack([np.stack([O, y2, z2], -1),
                     np.stack([y2, -2 * x2, -w2], -1),
                     np.stack([z2, w2, -2 * x2], -1)], -2)
    dRdy = np.stack([np.stack([-2 * y2, x2, w2], -1),
                     np.stack([x2, O, z2], -1),
                     np.stack([-w2, z2, -2 * y2], -1)], -2)
    dRdz = np.stack([np.stack([-2 * z2, -w2, x2], -1),
                     np.stack([w2, -2 * z2, y2], -1),
                     np.stack([x2, y2, O], -1)], -2)
    return np.stack([dRdw, dRdx, dRdy, dRdz], axis=1)


def _rotmat(qn: np.ndarray) -> np.ndarray:
    w, x, y, z = qn
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], float)


def _rotmat_batch(qn: np.ndarray) -> np.ndarray:
    w, x, y, z = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    R = np.empty((qn.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@dataclass
class Grads:
    mu: np.ndarray
    log_s: np.ndarray
    quat: np.ndarray
    log_a: np.ndarray

    def flat_norm(self) -> float:
        return float(np.sqrt(sum(np.sum(g * g) for g in
                                 (self.mu, self.log_s, self.quat, self.log_a))))


class GaussianField:
    def __init__(self, mu, log_s, quat, log_a, *, world_axes=None,
                 lam_a: float = 2e-4, lam_s: float = 1e-4, trunc: float = 3.0,
                 chunk: int = 8192):
        self.mu = np.asarray(mu, float).reshape(-1, 3).copy()
        self.log_s = np.asarray(log_s, float).reshape(-1, 3).copy()
        self.quat = np.asarray(quat, float).reshape(-1, 4).copy()
        self.log_a = np.asarray(log_a, float).reshape(-1).copy()
        self.lam_a = float(lam_a)
        self.lam_s = float(lam_s)
        self.trunc = float(trunc)
        self.chunk = int(chunk)
        self.world_axes = world_axes

    # -- properties -------------------------------------------------
    @property
    def n(self) -> int:
        return self.mu.shape[0]

    @property
    def amplitude(self) -> np.ndarray:
        return np.exp(self.log_a)

    @property
    def sigma(self) -> np.ndarray:
        return np.exp(self.log_s)

    def _qn(self) -> np.ndarray:
        return self.quat / np.linalg.norm(self.quat, axis=1, keepdims=True).clip(1e-12)

    def _precisions(self):
        qn = self._qn()
        R = _rotmat_batch(qn)
        inv_s2 = 1.0 / self.sigma ** 2
        P = np.einsum("nij,nj,nkj->nik", R, inv_s2, R)
        return qn, R, inv_s2, P

    def covariances(self) -> np.ndarray:
        qn = self._qn()
        R = _rotmat_batch(qn)
        s2 = self.sigma ** 2
        return np.einsum("nij,nj,nkj->nik", R, s2, R)

    # -- constructor ---------------------------------------------
    @classmethod
    def from_volume(cls, vol: Volume, n_gaussians: int, seed: int = 0,
                    init_sigma_vox=(1.1, 1.1, 1.6), **kw) -> "GaussianField":
        rng = np.random.default_rng(seed)
        H, W, K = vol.V.shape
        x_axis = np.linspace(0.0, 1.0, W) if W > 1 else np.zeros(1)
        y_axis = np.linspace(0.0, 1.0, H) if H > 1 else np.zeros(1)
        z_axis = vol.z_train.astype(float)

        val = np.where(vol.mask3d, np.abs(np.nan_to_num(vol.V)), 0.0).ravel()
        if val.sum() <= 0:
            val = vol.mask3d.astype(float).ravel()
        pick = rng.choice(val.size, size=n_gaussians, replace=True, p=val / val.sum())
        ii, jj, kk = np.unravel_index(pick, (H, W, K))

        dx = 1.0 / max(W - 1, 1); dy = 1.0 / max(H - 1, 1)
        dz = float(np.median(np.diff(z_axis))) if z_axis.size > 1 else 1.0
        jit = rng.normal(0, 0.35, size=(n_gaussians, 3)) * np.array([dx, dy, dz])
        mu = np.clip(np.stack([x_axis[jj], y_axis[ii], z_axis[kk]], 1) + jit, 0.0, 1.0)

        base = np.array([init_sigma_vox[0] * dx, init_sigma_vox[1] * dy, init_sigma_vox[2] * dz])
        log_s = np.log(base[None] * rng.uniform(0.7, 1.4, size=(n_gaussians, 3)))
        quat = np.zeros((n_gaussians, 4)); quat[:, 0] = 1.0
        quat += rng.normal(0, 0.05, size=quat.shape)

        overlap = max(n_gaussians * np.prod(init_sigma_vox) * 8.0 / max(vol.mask3d.sum(), 1), 1.0)
        local = np.abs(np.nan_to_num(vol.V))[ii, jj, kk]
        log_a = np.log(np.clip(local / overlap, 1e-4, None))
        return cls(mu, log_s, quat, log_a, world_axes=(x_axis, y_axis, z_axis), **kw)

    # -- dense evaluation (exact) --------------------------------
    def render_at(self, coords: np.ndarray, chunk: int | None = None) -> np.ndarray:
        shp = coords.shape[:-1]
        pts = coords.reshape(-1, 3)
        _, _, _, P = self._precisions()
        a = self.amplitude
        out = np.zeros(pts.shape[0])
        c = chunk or self.chunk
        for c0 in range(0, pts.shape[0], c):
            d = pts[c0:c0 + c][None, :, :] - self.mu[:, None, :]      # (N, m, 3)
            Pd = np.matmul(d, np.transpose(P, (0, 2, 1)))
            quad = np.einsum("nmi,nmi->nm", d, Pd)
            out[c0:c0 + c] = (a[:, None] * np.exp(-0.5 * quad)).sum(0)
        return out.reshape(shp)

    def render_grid(self, axes) -> np.ndarray:
        """Evaluate on a regular (x, y, z) grid using the truncated support.

        Same result as ``render_at`` on the same points, but O(N * box) instead
        of O(N * voxels) -- the difference between seconds and minutes once a
        map has a few hundred thousand voxels.
        """
        xa, ya, za = axes
        _qn, _R, _inv, P = self._precisions()
        L = self._local_grid(axes=axes)
        X = L["dxg"][:, None, :, None]
        Y = L["dyg"][:, :, None, None]
        Z = L["dzg"][:, None, None, :]
        p = {(i, j): P[:, i, j][:, None, None, None] for i in range(3) for j in range(3)}
        quad = (p[0, 0] * X * X + p[1, 1] * Y * Y + p[2, 2] * Z * Z
                + 2 * p[0, 1] * X * Y + 2 * p[0, 2] * X * Z + 2 * p[1, 2] * Y * Z)
        valid = (L["vy"][:, :, None, None] & L["vx"][:, None, :, None]
                 & L["vz"][:, None, None, :])
        phi = np.where(valid, np.exp(-0.5 * quad), 0.0)
        del quad
        lin = ((L["iyc"][:, :, None, None] * xa.size + L["jxc"][:, None, :, None]) * za.size
               + L["kzc"][:, None, None, :])
        out = np.bincount(lin.ravel(),
                          weights=(self.amplitude[:, None, None, None] * phi).ravel(),
                          minlength=ya.size * xa.size * za.size)
        return out.reshape(ya.size, xa.size, za.size)

    def render_volume(self, vol: Volume, which: str = "train") -> np.ndarray:
        xa, ya, _z = self.world_axes
        z = vol.z_train if which == "train" else vol.z_eval_axis()
        if z.size == 0:
            return np.zeros((ya.size, xa.size, 0))
        return self.render_grid((xa, ya, z))

    # -- local truncated grid (batched over all Gaussians) --------
    def _local_grid(self, vol: Volume = None, axes=None):
        """Return per-Gaussian local index/offset tensors for the 3-sigma box.

        Uniform half-widths across Gaussians so the whole thing stays a single
        vectorised block; entries outside the grid are flagged in ``valid``.
        """
        xa, ya, za = axes if axes is not None else self.world_axes
        H, W, K = ya.size, xa.size, za.size
        # per-axis worst-case marginal std over all Gaussians
        marg = np.sqrt(np.clip(np.einsum("nii->ni", self.covariances()), 1e-12, None))
        mmax = marg.max(axis=0)

        def half(ax, m):
            """Index half-width that certainly covers `trunc` sigma.

            Sized from the *smallest* gap, because a held-out-channel axis is
            not uniformly spaced (dropping every 5th channel leaves gaps of
            1,1,1,2,...). Assuming uniform spacing here silently walks the box
            off the Gaussian it belongs to.
            """
            if ax.size < 2:
                return 0
            step = float(np.min(np.diff(ax)))
            return int(min(np.ceil(self.trunc * m / max(step, 1e-12)) + 1, ax.size - 1))

        bx, by, bz = half(xa, mmax[0]), half(ya, mmax[1]), half(za, mmax[2])

        # nearest index by search, not by division -- correct for any spacing
        def centre(ax, v):
            j = np.searchsorted(ax, v)
            j = np.clip(j, 1, max(ax.size - 1, 1))
            lo = np.clip(j - 1, 0, ax.size - 1)
            take_lo = np.abs(v - ax[lo]) <= np.abs(ax[np.clip(j, 0, ax.size - 1)] - v)
            return np.where(take_lo, lo, np.clip(j, 0, ax.size - 1))

        cj = centre(xa, self.mu[:, 0])
        ci = centre(ya, self.mu[:, 1])
        ck = centre(za, self.mu[:, 2])

        ox, oy, oz = np.arange(-bx, bx + 1), np.arange(-by, by + 1), np.arange(-bz, bz + 1)
        jx = cj[:, None] + ox[None, :]                       # (N, Bx)
        iy = ci[:, None] + oy[None, :]
        kz = ck[:, None] + oz[None, :]
        vx, vy, vz = (jx >= 0) & (jx < W), (iy >= 0) & (iy < H), (kz >= 0) & (kz < K)
        jxc, iyc, kzc = np.clip(jx, 0, W - 1), np.clip(iy, 0, H - 1), np.clip(kz, 0, K - 1)
        dxg = xa[jxc] - self.mu[:, 0, None]                  # (N, Bx)
        dyg = ya[iyc] - self.mu[:, 1, None]
        dzg = za[kzc] - self.mu[:, 2, None]
        return dict(H=H, W=W, K=K, jxc=jxc, iyc=iyc, kzc=kzc, vx=vx, vy=vy, vz=vz,
                    dxg=dxg, dyg=dyg, dzg=dzg, shape=(iy.shape[1], jx.shape[1], kz.shape[1]))


    # -- loss + analytic gradient (truncated, separable) ---------
    def loss_and_grad(self, vol: Volume):
        """Loss and closed-form gradients on the truncated support.

        The quadratic form is *separable*: with offsets that depend on only one
        box axis each, ``d^T P d`` expands into six outer products of 1-D
        per-Gaussian vectors. Nothing here ever materialises the
        ``(N, By, Bx, Bz, 3)`` offset tensor -- which for a few thousand
        primitives on a large map is hundreds of megabytes per call, and was
        the real cost of this routine. Every gradient contraction likewise
        reduces to nine scalars per Gaussian.
        """
        qn, R, inv_s2, P = self._precisions()
        a = self.amplitude
        N = self.n
        L = self._local_grid(vol)
        H, W, K = L["H"], L["W"], L["K"]
        By, Bx, Bz = L["shape"]

        X = L["dxg"][:, None, :, None]          # (N,1,Bx,1)
        Y = L["dyg"][:, :, None, None]          # (N,By,1,1)
        Z = L["dzg"][:, None, None, :]          # (N,1,1,Bz)
        p = {(i, j): P[:, i, j][:, None, None, None] for i in range(3) for j in range(3)}
        quad = (p[0, 0] * X * X + p[1, 1] * Y * Y + p[2, 2] * Z * Z
                + 2 * p[0, 1] * X * Y + 2 * p[0, 2] * X * Z + 2 * p[1, 2] * Y * Z)
        valid = (L["vy"][:, :, None, None] & L["vx"][:, None, :, None]
                 & L["vz"][:, None, None, :])
        phi = np.where(valid, np.exp(-0.5 * quad), 0.0)
        del quad

        lin = ((L["iyc"][:, :, None, None] * W + L["jxc"][:, None, :, None]) * K
               + L["kzc"][:, None, None, :])                              # (N,By,Bx,Bz)
        Vhat = np.bincount(lin.ravel(),
                           weights=(a[:, None, None, None] * phi).ravel(),
                           minlength=H * W * K)

        target = np.where(np.isfinite(vol.V), vol.V, 0.0).reshape(-1)
        mask = (vol.mask3d & np.isfinite(vol.V)).reshape(-1)
        resid = np.where(mask, Vhat - target, 0.0)
        data_loss = 0.5 * float(resid @ resid)
        reg_loss = self.lam_a * float(a.sum()) + self.lam_s * float(np.sum(self.log_s ** 2))

        w = resid[lin] * phi                                             # (N,By,Bx,Bz)
        del phi, lin

        dx, dy, dz = L["dxg"], L["dyg"], L["dzg"]
        # first and second moments of w against each box axis
        S = np.stack([np.einsum("nyxz,nx->n", w, dx),
                      np.einsum("nyxz,ny->n", w, dy),
                      np.einsum("nyxz,nz->n", w, dz)], axis=1)            # (N,3)
        T = np.empty((N, 3, 3))
        T[:, 0, 0] = np.einsum("nyxz,nx->n", w, dx * dx)
        T[:, 1, 1] = np.einsum("nyxz,ny->n", w, dy * dy)
        T[:, 2, 2] = np.einsum("nyxz,nz->n", w, dz * dz)
        T[:, 0, 1] = T[:, 1, 0] = np.einsum("nyxz,nx,ny->n", w, dx, dy)
        T[:, 0, 2] = T[:, 2, 0] = np.einsum("nyxz,nx,nz->n", w, dx, dz)
        T[:, 1, 2] = T[:, 2, 1] = np.einsum("nyxz,ny,nz->n", w, dy, dz)
        sum_w = w.reshape(N, -1).sum(1)
        del w

        g_la = a * sum_w + self.lam_a * a
        g_mu = a[:, None] * np.einsum("nij,nj->ni", P, S)
        M = -0.5 * a[:, None, None] * T
        pos_grad = np.linalg.norm(g_mu, axis=1)

        # dL/dP -> dL/d log_s   and   dL/dP -> dL/dR -> dL/dq
        RtMR = np.matmul(np.matmul(np.transpose(R, (0, 2, 1)), M), R)     # (N,3,3)
        g_ls = np.einsum("nii->ni", RtMR) * (-2.0 * inv_s2) + 2 * self.lam_s * self.log_s

        GR = 2.0 * np.matmul(M, R) * inv_s2[:, None, :]                   # dL/dR = 2 M R Lambda
        dRdq = _dR_dq_batch(qn)                                          # (N,4,3,3)
        g_qn = np.einsum("nkij,nij->nk", dRdq, GR)                       # (N,4)
        qnorm = np.linalg.norm(self.quat, axis=1).clip(1e-12)
        proj = np.eye(4)[None] - qn[:, :, None] * qn[:, None, :]
        g_q = np.einsum("nij,nj->ni", proj, g_qn) / qnorm[:, None]

        grads = Grads(g_mu, g_ls, g_q, g_la)
        aux = {"data_loss": data_loss, "reg_loss": reg_loss, "pos_grad": pos_grad}
        return data_loss + reg_loss, grads, aux

    # -- rendering conveniences --------------------------------
    def render_band(self, wavenumber: float, vol: Volume, width: float = 0.0) -> np.ndarray:
        xa, ya, _ = self.world_axes
        H, W = ya.size, xa.size
        zs = (np.array([vol.z_of(wavenumber)]) if width <= 0
              else np.linspace(vol.z_of(wavenumber - width), vol.z_of(wavenumber + width), 9))
        cx = np.broadcast_to(xa[None, :, None], (H, W, zs.size))
        cy = np.broadcast_to(ya[:, None, None], (H, W, zs.size))
        cz = np.broadcast_to(zs[None, None, :], (H, W, zs.size))
        coords = np.stack(np.broadcast_arrays(cx, cy, cz), axis=-1)
        img = self.render_at(coords)
        return img[..., 0] if width <= 0 else img.mean(-1)

    # -- persistence -----------------------------------------
    def save(self, path: str) -> None:
        np.savez_compressed(
            path, mu=self.mu, log_s=self.log_s, quat=self.quat, log_a=self.log_a,
            lam_a=self.lam_a, lam_s=self.lam_s, trunc=self.trunc,
            x_axis=self.world_axes[0], y_axis=self.world_axes[1], z_axis=self.world_axes[2],
        )

    @classmethod
    def load(cls, path: str) -> "GaussianField":
        d = np.load(path)
        return cls(d["mu"], d["log_s"], d["quat"], d["log_a"],
                   world_axes=(d["x_axis"], d["y_axis"], d["z_axis"]),
                   lam_a=float(d["lam_a"]), lam_s=float(d["lam_s"]), trunc=float(d["trunc"]))

    def n_params(self) -> int:
        return self.n * (3 + 3 + 4 + 1)

    # -- densification -------------------------------------
    def prune(self, a_min: float) -> int:
        keep = self.amplitude >= a_min
        removed = int((~keep).sum())
        if removed and keep.sum() >= 8:
            self.mu, self.log_s = self.mu[keep], self.log_s[keep]
            self.quat, self.log_a = self.quat[keep], self.log_a[keep]
        return removed

    def densify(self, pos_grad: np.ndarray, max_gaussians: int,
                grad_frac: float = 0.12, seed: int = 0) -> int:
        if self.n >= max_gaussians or pos_grad.size != self.n:
            return 0
        rng = np.random.default_rng(seed)
        k = min(max_gaussians - self.n, max(1, int(grad_frac * self.n)))
        sel = np.argsort(pos_grad)[-k:]
        s = self.sigma[sel]
        big = s.max(1) > 2.0 * np.median(self.sigma.max(1))
        new_mu = np.clip(self.mu[sel] + rng.normal(0, 1, size=(k, 3)) * (0.6 * s), 0.0, 1.0)
        new_ls = self.log_s[sel].copy(); new_ls[big] -= np.log(1.6)
        self.log_a[sel] -= np.log(2.0)
        self.mu = np.vstack([self.mu, new_mu])
        self.log_s = np.vstack([self.log_s, new_ls])
        self.quat = np.vstack([self.quat, self.quat[sel].copy()])
        self.log_a = np.concatenate([self.log_a, self.log_a[sel].copy()])
        return int(k)
