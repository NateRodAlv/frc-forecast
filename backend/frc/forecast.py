"""Year-over-year trajectory forecaster ("technical analysis" for FRC teams).

Each team-season is reduced to z = end-of-season strength standardised within the year,
making seasons comparable even though the game (and score scale) changes yearly.
A team's history is a price series; we predict next season's z from its shape:
level, 1y / 2y momentum, drawdown from peak (slump depth), volatility, tenure.

Three predictors, all walk-forward backtested (train only on targets < test year):
  persist : z_next = z (naive baseline)
  ridge   : linear mean-reversion + momentum
  analog  : k-nearest-neighbour over *other teams'* historical trajectories with the same
            shape (e.g. "slump then rebound"), distance-weighted. Gives full outcome
            distribution and the named analog teams as the explanation.
Final = blend(ridge, analog) with the weight picked on the backtest.
"""
import numpy as np

FEATS = ["z0", "d1", "d2", "drawdown", "vol", "tenure"]
SKIP_YEARS = {2020, 2021}   # no FRC season played


def zscores(end_strength, min_teams=20):
    out = {}
    for y, d in end_strength.items():
        if len(d) < min_teams:
            continue
        v = np.array(list(d.values()))
        mu, sd = v.mean(), v.std() or 1.0
        out[y] = {t: (s - mu) / sd for t, s in d.items()}
    return out


def histories(z):
    """team -> list of (year, z) over seasons actually played (skip-years removed)."""
    h = {}
    for y in sorted(z):
        if y in SKIP_YEARS:
            continue
        for t, v in z[y].items():
            h.setdefault(t, []).append((y, v))
    return h


def features(series):
    """series: list of z oldest->newest (>=1). Returns feature vector or None."""
    z0 = series[-1]
    z1 = series[-2] if len(series) > 1 else z0
    z2 = series[-3] if len(series) > 2 else z1
    window = series[-4:]
    return np.array([z0, z0 - z1, z1 - z2, max(window) - z0, float(np.std(window)) if len(window) > 1 else 0.5,
                     min(len(series), 8) / 8.0])


def build_pairs(h, z, max_target=None):
    """Training pairs (features at season i -> z at next played season)."""
    X, y, meta = [], [], []
    for t, ser in h.items():
        for i in range(len(ser) - 1):
            ty = ser[i + 1][0]
            if max_target is not None and ty > max_target:
                continue
            X.append(features([v for _, v in ser[: i + 1]]))
            y.append(ser[i + 1][1])
            meta.append((t, ser[i][0], ty, [round(v, 2) for _, v in ser[max(0, i - 3): i + 1]], round(ser[i + 1][1], 2)))
    return np.array(X), np.array(y), meta


class Ridge:
    def __init__(self, lam=3.0):
        self.lam = lam

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        A = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        R = self.lam * np.eye(A.shape[1])
        R[0, 0] = 0
        self.w = np.linalg.solve(A.T @ A + R, A.T @ y)
        self.resid_sd = float(np.std(y - A @ self.w))
        return self

    def predict(self, X):
        return np.c_[np.ones(len(X)), (X - self.mu) / self.sd] @ self.w


class Analog:
    """kNN on standardised shape features; weights emphasise trajectory shape over level."""

    def __init__(self, k=60, wts=(1.0, 0.9, 0.6, 0.8, 0.5, 0.3)):
        self.k, self.wts = k, np.array(wts)

    def fit(self, X, y, meta):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.Z = (X - self.mu) / self.sd * self.wts
        self.y, self.meta = y, meta
        return self

    def neighbours(self, x):
        q = (x - self.mu) / self.sd * self.wts
        d = np.sqrt(((self.Z - q) ** 2).sum(1))
        idx = np.argsort(d)[: self.k]
        w = 1.0 / (d[idx] + 0.25)
        return idx, w / w.sum(), d[idx]

    def predict(self, X, quantiles=(0.1, 0.5, 0.9)):
        mean, qs = [], []
        for x in X:
            idx, w, _ = self.neighbours(x)
            yy = self.y[idx]
            m = float((w * yy).sum())
            o = np.argsort(yy)
            cw = np.cumsum(w[o])
            qs.append([float(yy[o][np.searchsorted(cw, q)]) for q in quantiles])
            mean.append(m)
        return np.array(mean), np.array(qs)


def walk_forward(z, test_years, k=60):
    """Return metrics per method and the best blend weight (analog share)."""
    h = histories(z)
    rows = []
    for ty in test_years:
        Xtr, ytr, mtr = build_pairs(h, z, max_target=ty - 1)
        if len(Xtr) < 200:
            continue
        # test pairs: target year == ty
        Xte, yte = [], []
        for t, ser in h.items():
            yrs = [a for a, _ in ser]
            if ty in yrs:
                i = yrs.index(ty)
                if i >= 1:
                    Xte.append(features([v for _, v in ser[:i]]))
                    yte.append(ser[i][1])
        if len(Xte) < 50:
            continue
        Xte, yte = np.array(Xte), np.array(yte)
        rg = Ridge().fit(Xtr, ytr)
        an = Analog(k=k).fit(Xtr, ytr, mtr)
        pr = rg.predict(Xte)
        pa, qa = an.predict(Xte)
        rows.append({"year": ty, "yte": yte, "persist": Xte[:, 0], "ridge": pr, "analog": pa, "qa": qa})
    if not rows:
        return None
    y = np.concatenate([r["yte"] for r in rows])
    out = {}
    for m in ("persist", "ridge", "analog"):
        p = np.concatenate([r[m] for r in rows])
        out[m] = {"mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}
    pr, pa = np.concatenate([r["ridge"] for r in rows]), np.concatenate([r["analog"] for r in rows])
    best_w, best = 0.5, 9
    for w in np.linspace(0, 1, 11):
        e = float(np.sqrt(np.mean((y - ((1 - w) * pr + w * pa)) ** 2)))
        if e < best:
            best, best_w = e, float(w)
    out["blend"] = {"analog_weight": best_w, "rmse": best}
    qa = np.concatenate([r["qa"] for r in rows])
    out["analog_interval_coverage_10_90"] = float(np.mean((y >= qa[:, 0]) & (y <= qa[:, 2])))
    out["n_test"] = int(len(y))
    out["years"] = [r["year"] for r in rows]
    return out


def predict_next(z, next_year, blend_w=0.5, k=60, rookie_prior=None):
    """Forecast next season z for every team with history. Returns dict team -> forecast."""
    h = histories(z)
    Xtr, ytr, mtr = build_pairs(h, z)
    rg = Ridge().fit(Xtr, ytr)
    an = Analog(k=k).fit(Xtr, ytr, mtr)
    teams = [t for t, s in h.items() if s[-1][0] >= max(z) - 1 or s[-1][0] >= next_year - 3]
    X = np.array([features([v for _, v in h[t]]) for t in teams])
    pr = rg.predict(X)
    pa, qa = an.predict(X)
    mean = (1 - blend_w) * pr + blend_w * pa
    out = {}
    for i, t in enumerate(teams):
        idx, w, d = an.neighbours(X[i])
        yy = an.y[idx]
        # residual spread of analogs around their own mean = outcome uncertainty
        sd = float(np.sqrt(np.sum(w * (yy - pa[i]) ** 2)))
        sd = float(np.sqrt(sd ** 2 + 0.0))  # kept explicit: no extra inflation
        top = np.argsort(d)[:5]
        analogs = [{"team": an.meta[idx[j]][0], "year": an.meta[idx[j]][1],
                    "path": an.meta[idx[j]][3], "next": an.meta[idx[j]][4]} for j in top]
        edge = float(pa[i] - pr[i])
        flags = []
        if X[i, 3] > 0.8 and X[i, 1] > 0.3:
            flags.append("rebounding after slump")
        if X[i, 3] > 1.0 and X[i, 1] < 0:
            flags.append("in drawdown")
        if X[i, 1] < -0.6:
            flags.append("sharp decline")
        if X[i, 1] > 0.6:
            flags.append("breakout momentum")
        if edge > 0.25:
            flags.append("analogs see upside")
        if edge < -0.25:
            flags.append("analogs see downside")
        out[t] = {"z_last": float(X[i, 0]), "mean": float(mean[i]), "sd": max(sd, 0.25),
                  "q10": float(qa[i, 0]), "q50": float(qa[i, 1]), "q90": float(qa[i, 2]),
                  "ridge": float(pr[i]), "analog": float(pa[i]), "analog_edge": edge,
                  "history": [[a, round(v, 2)] for a, v in h[t][-6:]], "analogs": analogs, "flags": flags}
    return out


def rookie_prior(z, rookie_year):
    """Mean/sd of z in a team's first season (and second) from history."""
    first, second = [], []
    for t, ser in histories(z).items():
        if len(ser) >= 1 and rookie_year.get(t) == ser[0][0]:
            first.append(ser[0][1])
            if len(ser) > 1:
                second.append(ser[1][1])
    f = (float(np.mean(first)), float(np.std(first))) if first else (-0.6, 0.8)
    return {"mean": f[0], "sd": max(f[1], 0.5)}
