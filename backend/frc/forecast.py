"""Year-over-year trajectory forecaster ("technical analysis" for FRC teams), age-aware.

Each team-season is reduced to z = end-of-season strength standardised within the year (see zscores), making
seasons comparable even though the game changes yearly. A team's history is a price series; we predict the next
season's z from its shape (level, 1y / 2y momentum, drawdown from peak, volatility) AND its true age
(season - rookie year + 1, from TBA).

Predictors, all walk-forward backtested (trained only on targets < test year):
  persist : z_next = z (naive baseline)
  ridge   : the trend model (reported under this key): linear mean-reversion + momentum + age terms (rookie / 2nd /
            3rd year, log age, short-history flags), plus top-end terms, averaged with a shallow boosted-tree model
            (see Trend). Plain ridge under-forecast elite teams.
  analog  : k-nearest-neighbour over OTHER teams' past team-seasons, restricted to the same age band AND the same
            amount of observed history, comparing only the history that exists (no padding). A 2-season team is
            only compared with other teams when THEY had 2 seasons. If a pool is too small it widens (see `level`).
Final = blend(ridge, analog), weight chosen on the backtest.
"""
import math

import numpy as np

SKIP_YEARS = {2020, 2021}   # no FRC season played
FEATS = ["z0", "d1", "d2", "drawdown", "vol", "age1", "age2", "age3", "log_age", "L1", "L2"]
BANDS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 7, "5-7"), (8, 12, "8-12"), (13, 999, "13+")]
W5 = np.array([1.0, 0.9, 0.6, 0.8, 0.5])           # analog distance weights for z0, d1, d2, drawdown, vol
SUBSETS = {1: [0], 2: [0, 1], 3: [0, 1, 2, 3, 4]}   # which of those exist with 1, 2, 3+ seasons of history


# ---------------------------------------------------------------- season normalisation
def zscores(end_strength, counts=None, min_teams=20, min_matches=8):
    """Standardise strengths within each season (mean 0, sd 1). When per-team match counts are given the
    mean/sd come only from teams with at least `min_matches` matches, so thinly-observed teams (whose ratings sit
    near the prior) don't distort the scale; every team is then expressed on that season scale."""
    out = {}
    for y, d in end_strength.items():
        if len(d) < min_teams:
            continue
        ref = list(d.values())
        if counts and y in counts:
            r2 = [s for t, s in d.items() if counts[y].get(t, 0) >= min_matches]
            if len(r2) >= min_teams:
                ref = r2
        v = np.array(ref)
        mu, sd = v.mean(), v.std() or 1.0
        out[y] = {t: (s - mu) / sd for t, s in d.items()}
    return out


def percentiles(z, counts=None, min_matches=8):
    """Percentile (0-100) of each team's season rating among teams with >= min_matches matches that season."""
    out = {}
    for y, d in z.items():
        ref = [v for t, v in d.items() if not counts or counts.get(y, {}).get(t, 0) >= min_matches] or list(d.values())
        ref = np.sort(np.array(ref))
        out[y] = {t: float(100.0 * np.searchsorted(ref, v, side="right") / len(ref)) for t, v in d.items()}
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


# ---------------------------------------------------------------- age
def age_band(age):
    if age is None:
        return "?"
    for lo, hi, name in BANDS:
        if lo <= age <= hi:
            return name
    return "13+"


def rookie_of(team, ser, rookie, first_year):
    """Rookie year from TBA; if unknown, the first season seen -- but only when that is after the data window
    starts (otherwise the team may be much older and we leave its age unknown rather than guess it is a rookie)."""
    r = (rookie or {}).get(team)
    if r:
        return r
    return ser[0][0] if ser[0][0] > first_year else None


def age_at(season, rookie_year):
    return None if rookie_year is None else max(1, season - rookie_year + 1)


def features(series, age):
    """series: z values oldest->newest (>=1). age: true age at the last season (None if unknown)."""
    L = len(series)
    z0 = series[-1]
    z1 = series[-2] if L > 1 else z0
    z2 = series[-3] if L > 2 else z1
    window = series[-4:]
    a = age if age is not None else max(L, 6)
    return np.array([z0, z0 - z1, z1 - z2, max(window) - z0, float(np.std(window)) if L > 1 else 0.5,
                     float(a == 1), float(a == 2), float(a == 3), math.log(min(a, 25)), float(L == 1), float(L == 2)])


def build_pairs(h, rookie=None, first_year=None, max_target=None):
    """Training pairs: (shape+age features at season i) -> z at the team's next played season."""
    X, y, meta = [], [], []
    for t, ser in h.items():
        rk = rookie_of(t, ser, rookie, first_year if first_year is not None else ser[0][0] - 1)
        for i in range(len(ser) - 1):
            ty = ser[i + 1][0]
            if max_target is not None and ty > max_target:
                continue
            age = age_at(ser[i][0], rk)
            X.append(features([v for _, v in ser[: i + 1]], age))
            y.append(ser[i + 1][1])
            meta.append((t, ser[i][0], ty, [round(v, 2) for _, v in ser[max(0, i - 3): i + 1]], round(ser[i + 1][1], 2),
                         age_band(age), min(i + 1, 3)))
    return np.array(X), np.array(y), meta


# ---------------------------------------------------------------- v2 features (NEW; the originals above are unchanged)
NB = len(FEATS)
EXT_FEATS = ["log_matches", "mean_last3", "mean_all", "max_all", "min_last4", "d3", "log_seasons"]


def features_v2(team, ser, age, counts):
    """FEATS plus 7 columns that tell the model how far to trust last season and where the team sits long-run.
    ser: [(year, z)] up to the last observed season. counts: {year: {team: matches played}} (unknown -> 8, the
    minimum used when standardising). Everything here is known before the target season starts."""
    v = [a for _, a in ser]
    base = features(v, age)
    n0 = (counts or {}).get(ser[-1][0], {}).get(team, 8)
    z3 = v[-4] if len(v) > 3 else v[-3] if len(v) > 2 else v[-2] if len(v) > 1 else v[-1]
    ext = [math.log(max(n0, 1)), float(np.mean(v[-3:])), float(np.mean(v)), float(max(v)), float(min(v[-4:])),
           (v[-3] - z3) if len(v) > 2 else 0.0, math.log(len(v))]
    return np.r_[base, ext]


def build_pairs_v2(h, counts, rookie=None, first_year=None, max_target=None):
    """Same pairs, in the same order, as build_pairs, but with the v2 feature columns.
    Returns (X, y, target_year per row)."""
    X, y, ty = [], [], []
    for t, ser in h.items():
        rk = rookie_of(t, ser, rookie, first_year if first_year is not None else ser[0][0] - 1)
        for i in range(len(ser) - 1):
            if max_target is not None and ser[i + 1][0] > max_target:
                continue
            X.append(features_v2(t, ser[: i + 1], age_at(ser[i][0], rk), counts))
            y.append(ser[i + 1][1])
            ty.append(ser[i + 1][0])
    return np.array(X), np.array(y), np.array(ty)


# ---------------------------------------------------------------- models
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


def _top_terms(X):
    """Base features plus terms that let a linear model bend at the top end. Plain ridge systematically under-
    forecasts elite teams (teams forecast above z=2 later averaged ~0.4 higher): strength is convex at the top,
    and young elite teams keep climbing. Columns of X follow FEATS."""
    z0, d1 = X[:, 0], X[:, 1]
    young = X[:, 5:8].sum(1)
    hinge = np.maximum(z0 - 1.0, 0.0)
    return np.c_[X, z0 ** 2, hinge, hinge * young, z0 * d1]


class Trend:
    """Trend model = average of (a) ridge on the base features + top-end terms and (b) a shallow gradient-boosted
    tree model on the base features. Walk-forward 2023-2026 RMSE 0.6386 vs 0.6429 for plain ridge, and the top-end
    bias roughly halves. Same fit/predict interface as Ridge, so it drops in wherever Ridge was used."""

    def __init__(self, lam=3.0):
        self.lam = lam

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.ridge = Ridge(self.lam).fit(_top_terms(X), y)
        self.gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=60,
                                                 l2_regularization=2.0, random_state=0).fit(X, y)
        self.resid_sd = self.ridge.resid_sd
        return self

    def predict(self, X):
        return 0.5 * self.ridge.predict(_top_terms(X)) + 0.5 * self.gbm.predict(X)


def _gbm(**kw):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(**kw)


class Trend2:
    """NEW. Trend v2: the same ridge (base features + top-end terms) averaged with a boosted-tree model that also sees
    EXT_FEATS (how many matches the last rating rests on, the team's long-run level, best/worst recent seasons).
    Takes the 18-column matrix from features_v2 / build_pairs_v2. Walk-forward on real 2013-2026 history: RMSE
    0.6393 vs 0.6420 (2023-26) and 0.7471 vs 0.7511 (2013-19) for Trend, i.e. small but it held in both windows."""

    def __init__(self, lam=3.0):
        self.lam = lam

    def fit(self, X, y):
        self.ridge = Ridge(self.lam).fit(_top_terms(X[:, :NB]), y)
        self.gbm = _gbm(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=60,
                        l2_regularization=2.0, random_state=0).fit(X, y)
        self.resid_sd = self.ridge.resid_sd
        return self

    def predict(self, X):
        return 0.5 * self.ridge.predict(_top_terms(X[:, :NB])) + 0.5 * self.gbm.predict(X)


def oof_residuals(X, y, ty, folds=5):
    """Out-of-fold Trend2 residuals (folds split by target season), so the variance model sees honest errors."""
    r = np.zeros(len(y))
    fid = {v: i % folds for i, v in enumerate(sorted(set(ty.tolist())))}
    f = np.array([fid[v] for v in ty.tolist()])
    for k in range(folds):
        tr = f != k
        if tr.sum() < 100 or (~tr).sum() == 0:
            continue
        r[~tr] = y[~tr] - Trend2().fit(X[tr], y[tr]).predict(X[~tr])
    return r


class SdModel:
    """NEW. Forecast-uncertainty model: predicts each team's squared forecast error from its features AND the season
    (so it follows the era: ratings have got steadier as events got longer and rosters better known). Season is the
    target season; for a forecast beyond the data it falls into the most recent bin. Replaces the spread of analog
    teams (which is ~15% too wide in 2023-26: a nominal 80% interval covered 87%)."""
    FLOOR = 0.09      # variance floor -> sd >= 0.3

    def fit(self, X, y, ty):
        o = oof_residuals(X, y, ty)
        self.m = _gbm(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=150,
                      l2_regularization=5.0, random_state=0).fit(np.c_[X, ty], o ** 2)
        return self

    def predict(self, X, target_year):
        return np.sqrt(np.clip(self.m.predict(np.c_[X, np.full(len(X), target_year)]), self.FLOOR, None))


def analog_sd(an, X, bands, lcs, pa):
    """The original uncertainty: weighted spread of the analog neighbours' outcomes (floor 0.25)."""
    out = []
    for i in range(len(X)):
        idx, w, _, _, _ = an.neighbours(X[i], bands[i], lcs[i])
        out.append(max(float(np.sqrt(np.sum(w * (an.y[idx] - pa[i]) ** 2))), 0.25))
    return np.array(out)


def sd_report(res, sd):
    """Quality of a Normal(mean, sd) forecast: log score (lower is better) and how often the 80% / 50% interval hit."""
    r = res / sd
    return {"nll": float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * r ** 2)),
            "cov80": float(np.mean(np.abs(r) < 1.2816)), "cov50": float(np.mean(np.abs(r) < 0.6745)),
            "mean_sd": float(np.mean(sd)), "resid_over_sd": float(np.std(r))}


def preseason_seeds(z, counts, rookie=None, min_year_offset=5):
    """NEW. A preseason strength forecast (in z units) for every team and every season, each made ONLY from seasons
    before it: for season y, Trend2 is fitted on target seasons < y and applied to the team histories up to y-1.
    The seeded rating models (ratings.SEEDED) start each season from these instead of from `carry x last rating`.
    Returns {"seeds": {year: {team: z_hat}}, "rookie_z": {year: mean first-season z of earlier rookies}}."""
    h = histories(z)
    first = min(z)
    seeds, rookie_z = {}, {}
    for y in sorted(z):
        if y < first + min_year_offset:
            continue
        Xtr, ytr, _ = build_pairs_v2(h, counts, rookie, first, max_target=y - 1)
        if len(Xtr) < 500:
            continue
        m = Trend2().fit(Xtr, ytr)
        teams, X = [], []
        for t, ser in h.items():
            sub = [(yy, v) for yy, v in ser if yy < y]
            if not sub or sub[-1][0] < y - 3:
                continue
            age = age_at(sub[-1][0], rookie_of(t, ser, rookie, first))
            teams.append(t)
            X.append(features_v2(t, sub, age, counts))
        if not teams:
            continue
        seeds[y] = {t: float(v) for t, v in zip(teams, m.predict(np.array(X)))}
        firsts = [ser[0][1] for ser in h.values() if first < ser[0][0] < y]
        rookie_z[y] = float(np.mean(firsts)) if firsts else -0.3
    return {"seeds": seeds, "rookie_z": rookie_z}


class Analog:
    """kNN over past team-seasons. With age_match=True the candidate pool is restricted to the same age band and
    the same history length, and only the features that exist for that history length are compared."""

    def __init__(self, k=60, min_pool=80, age_match=True):
        self.k, self.min_pool, self.age_match = k, min_pool, age_match

    def fit(self, X, y, meta):
        self.mu, self.sd = X[:, :5].mean(0), X[:, :5].std(0) + 1e-9
        self.Z = (X[:, :5] - self.mu) / self.sd * W5
        self.y, self.meta = y, meta
        self.band = np.array([m[5] for m in meta])
        self.lc = np.array([m[6] for m in meta])
        return self

    def neighbours(self, x, band, lc):
        """Returns (indices, weights, distances, level, pool_size). level 0 = same age band + same history length,
        1 = same age band only, 2 = everyone (pool too small)."""
        if self.age_match:
            cols = SUBSETS[lc]
            pool = np.where((self.band == band) & (self.lc == lc))[0]
            level = 0
            if len(pool) < self.min_pool:
                pool, level = np.where(self.band == band)[0], 1
            if len(pool) < self.min_pool:
                pool, level = np.arange(len(self.y)), 2
        else:
            cols, pool, level = SUBSETS[3], np.arange(len(self.y)), 2
        q = (x[:5] - self.mu) / self.sd * W5
        d = np.sqrt(((self.Z[pool][:, cols] - q[cols]) ** 2).sum(1))
        o = np.argsort(d)[: self.k]
        w = 1.0 / (d[o] + 0.25)
        return pool[o], w / w.sum(), d[o], level, len(pool)

    def predict(self, X, bands, lcs, quantiles=(0.1, 0.5, 0.9)):
        mean, qs = [], []
        for x, b, lc in zip(X, bands, lcs):
            idx, w, _, _, _ = self.neighbours(x, b, lc)
            yy = self.y[idx]
            o = np.argsort(yy)
            cw = np.cumsum(w[o])
            qs.append([float(yy[o][min(np.searchsorted(cw, q), len(yy) - 1)]) for q in quantiles])
            mean.append(float((w * yy).sum()))
        return np.array(mean), np.array(qs)


# ---------------------------------------------------------------- backtest
def walk_forward(z, test_years, rookie=None, k=60, counts=None):
    """Return metrics per method and the best blend weight (analog share). Trains only on targets < test year.
    With `counts` ({year: {team: matches}}) the NEW models (Trend v2, variance model v2) are scored on the same
    seasons next to the originals; the original keys are computed exactly as before."""
    h = histories(z)
    first_year = min(z)
    rows = []
    for ty in test_years:
        Xtr, ytr, mtr = build_pairs(h, rookie, first_year, max_target=ty - 1)
        if len(Xtr) < 200:
            continue
        Xte, yte, bands, lcs, Xte2 = [], [], [], [], []
        for t, ser in h.items():
            yrs = [a for a, _ in ser]
            if ty in yrs:
                i = yrs.index(ty)
                if i >= 1:
                    sub = ser[:i]
                    age = age_at(sub[-1][0], rookie_of(t, ser, rookie, first_year))
                    Xte.append(features([v for _, v in sub], age)); yte.append(ser[i][1])
                    bands.append(age_band(age)); lcs.append(min(len(sub), 3))
                    if counts is not None:
                        Xte2.append(features_v2(t, sub, age, counts))
        if len(Xte) < 50:
            continue
        Xte, yte = np.array(Xte), np.array(yte)
        rg = Trend().fit(Xtr, ytr)
        an = Analog(k=k).fit(Xtr, ytr, mtr)
        an0 = Analog(k=k, age_match=False).fit(Xtr, ytr, mtr)
        pa, qa = an.predict(Xte, bands, lcs)
        pn, _ = an0.predict(Xte, bands, lcs)
        row = {"year": ty, "yte": yte, "persist": Xte[:, 0], "ridge": rg.predict(Xte), "analog": pa,
               "analog_nomatch": pn, "qa": qa, "bands": np.array(bands)}
        if counts is not None:
            X2tr, y2tr, ty2tr = build_pairs_v2(h, counts, rookie, first_year, max_target=ty - 1)
            Xte2 = np.array(Xte2)
            row["trend2"] = Trend2().fit(X2tr, y2tr).predict(Xte2)
            row["sd2"] = SdModel().fit(X2tr, y2tr, ty2tr).predict(Xte2, ty)
            row["sd_an"] = analog_sd(an, Xte, bands, lcs, pa)
        rows.append(row)
    if not rows:
        return None
    cat = lambda key: np.concatenate([r[key] for r in rows])  # noqa
    y = cat("yte")
    out = {}
    for m in ("persist", "ridge", "analog", "analog_nomatch"):
        p = cat(m)
        out[m] = {"mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}
    pr, pa = cat("ridge"), cat("analog")
    best_w, best = 0.5, 9
    for w in np.linspace(0, 1, 11):
        e = float(np.sqrt(np.mean((y - ((1 - w) * pr + w * pa)) ** 2)))
        if e < best:
            best, best_w = e, float(w)
    out["blend"] = {"analog_weight": best_w, "rmse": best}
    qa = cat("qa")
    out["analog_interval_coverage_10_90"] = float(np.mean((y >= qa[:, 0]) & (y <= qa[:, 2])))
    bl = (1 - best_w) * pr + best_w * pa
    bands = cat("bands")
    out["by_age"] = {}
    for name in [b[2] for b in BANDS] + ["?"]:
        m = bands == name
        if m.sum() >= 30:
            out["by_age"][name] = {"n": int(m.sum()), **{k2: float(np.sqrt(np.mean((y[m] - v[m]) ** 2))) for k2, v in
                                                          (("persist", cat("persist")), ("ridge", pr), ("analog", pa), ("blend", bl))}}
    out["n_test"] = int(len(y))
    out["years"] = [r["year"] for r in rows]
    if counts is not None:
        # ---- NEW models, scored on the same team-seasons
        p2 = cat("trend2")
        out["trend2"] = {"mae": float(np.mean(np.abs(y - p2))), "rmse": float(np.sqrt(np.mean((y - p2) ** 2)))}
        b2_w, b2 = 0.5, 9
        for w in np.linspace(0, 1, 11):
            e = float(np.sqrt(np.mean((y - ((1 - w) * p2 + w * pa)) ** 2)))
            if e < b2:
                b2, b2_w = e, float(w)
        out["blend2"] = {"analog_weight": b2_w, "rmse": b2}
        mean2 = (1 - b2_w) * p2 + b2_w * pa               # same mean for both spreads, so only the sd differs
        out["sd"] = {"analog": sd_report(y - mean2, cat("sd_an")), "sd2": sd_report(y - mean2, cat("sd2"))}
        for name, d in out["by_age"].items():
            m = bands == name
            d["trend2"] = float(np.sqrt(np.mean((y[m] - p2[m]) ** 2)))
    return out


# ---------------------------------------------------------------- forecast
def predict_next(z, next_year, blend_w=0.5, k=60, rookie=None, counts=None, trend="trend", sd_model="analog"):
    """Forecast next season z for every recently active team. Returns dict team -> forecast.
    Tags are split in two groups: `shape` (what last season looked like) and `outlook` (what the forecast expects).
    trend: "trend" (original) or "trend2" (NEW); sd_model: "analog" (original) or "sd2" (NEW). The NEW ones need
    `counts`. Both generations are always computed when counts are given and stored side by side on each team
    (`ridge`/`trend2`, `sd_analog`/`sd2`); `ridge` keeps meaning the original trend model, `mean` and `sd` follow the choice."""
    h = histories(z)
    first_year = min(z)
    Xtr, ytr, mtr = build_pairs(h, rookie, first_year)
    rg = Trend().fit(Xtr, ytr)
    an = Analog(k=k).fit(Xtr, ytr, mtr)
    teams = [t for t, s in h.items() if s[-1][0] >= next_year - 3]
    X, X2, bands, lcs, ages = [], [], [], [], []
    for t in teams:
        ser = h[t]
        age = age_at(ser[-1][0], rookie_of(t, ser, rookie, first_year))
        X.append(features([v for _, v in ser], age)); ages.append(age)
        if counts is not None:
            X2.append(features_v2(t, ser, age, counts))
        bands.append(age_band(age)); lcs.append(min(len(ser), 3))
    X = np.array(X)
    pr = rg.predict(X)
    pa, qa = an.predict(X, bands, lcs)
    p2 = sd2 = None
    if counts is not None:
        X2 = np.array(X2)
        X2tr, y2tr, ty2tr = build_pairs_v2(h, counts, rookie, first_year)
        p2 = Trend2().fit(X2tr, y2tr).predict(X2)
        sd2 = SdModel().fit(X2tr, y2tr, ty2tr).predict(X2, next_year)
    elif trend == "trend2" or sd_model == "sd2":
        raise ValueError("trend2 / sd2 need per-season match counts (counts=...)")
    base = p2 if trend == "trend2" else pr
    mean = (1 - blend_w) * base + blend_w * pa
    out = {}
    for i, t in enumerate(teams):
        idx, w, d, level, pool_n = an.neighbours(X[i], bands[i], lcs[i])
        yy = an.y[idx]
        sd_an = max(float(np.sqrt(np.sum(w * (yy - pa[i]) ** 2))), 0.25)
        sd = float(sd2[i]) if sd_model == "sd2" else sd_an
        analogs, seen = [], set()
        for j in np.argsort(d):                         # nearest first, one entry per team
            m = an.meta[idx[j]]
            if m[0] in seen:
                continue
            seen.add(m[0])
            analogs.append({"team": m[0], "year": m[1], "path": m[3], "next": m[4], "age_band": m[5]})
            if len(analogs) == 5:
                break
        d1, dd, z0 = X[i, 1], X[i, 3], X[i, 0]
        edge = float(pa[i] - base[i])
        shape, outlook = [], []
        if ages[i] is not None and ages[i] <= 3:
            shape.append(f"young team (season {ages[i]} last year)")
        if dd > 0.8 and d1 > 0.3:
            shape.append("rebounded last season after a slump")
        if dd > 1.0 and d1 < 0:
            shape.append("still well below its recent peak")
        if d1 < -0.6:
            shape.append("big drop last season")
        if d1 > 0.6:
            shape.append("big jump last season")
        delta = float(mean[i] - z0)
        if delta > 0.25:
            outlook.append("expected to improve")
        elif delta < -0.25:
            outlook.append("expected to regress")
        if blend_w >= 0.1 and abs(edge) > 0.25:       # analog tags only when analogs actually feed the forecast
            outlook.append("similar teams did better than the trend model expects" if edge > 0 else
                           "similar teams did worse than the trend model expects")
        out[t] = {"z_last": float(z0), "mean": float(mean[i]), "sd": sd,
                  "q10": float(qa[i, 0]), "q50": float(qa[i, 1]), "q90": float(qa[i, 2]),
                  "ridge": float(pr[i]), "analog": float(pa[i]), "analog_edge": edge,
                  **({"trend2": float(p2[i]), "sd_analog": sd_an, "sd2": float(sd2[i])} if p2 is not None else {}),
                  "age": ages[i], "analog_pool": {"age_band": bands[i], "history": ("3+" if lcs[i] == 3 else str(lcs[i])),
                                                  "n": int(pool_n), "level": int(level)},
                  "history": [[a, round(v, 2)] for a, v in h[t][-6:]], "analogs": analogs,
                  "shape_flags": shape, "outlook_flags": outlook, "flags": shape + outlook}
    return out


def rookie_prior(z, rookie_year):
    """Mean/sd of z in a team's first season from history."""
    first = []
    for t, ser in histories(z).items():
        if len(ser) >= 1 and rookie_year.get(t) == ser[0][0]:
            first.append(ser[0][1])
    f = (float(np.mean(first)), float(np.std(first))) if first else (-0.6, 0.8)
    return {"mean": f[0], "sd": max(f[1], 0.5)}
