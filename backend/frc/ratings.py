"""Online rating models + walk-forward backtest to pick the best ranker.

All three models are single-pass online learners, so a backtest is one chronological
sweep with NO lookahead: each match is predicted before the model sees its result.

  elo    - alliance-mean Elo with margin-of-victory multiplier
  epa    - additive score model: alliance score = sum of team contributions (OPR-like but
           online, with per-team learning rate that decays with matches played)
  bt     - Bradley-Terry / logistic: P(red wins) = sigmoid(sum w_red - sum w_blue), L2-shrunk

Between seasons every team is shrunk toward the field mean (games change every year).
The shrink factor and learning rates are tuned on the training years only.
"""
import math
from collections import defaultdict

import numpy as np

QUAL = ("qm",)


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


class Elo:
    name = "elo"

    def __init__(self, k=12.0, carry=0.65, scale=400.0, init=1500.0, margin=True):
        self.k, self.carry, self.scale, self.init = k, carry, scale, init
        self.margin = margin
        self.r = defaultdict(lambda: init)
        self.n = defaultdict(int)

    def new_season(self):
        for t in list(self.r):
            self.r[t] = self.init + self.carry * (self.r[t] - self.init)
            self.n[t] = 0

    def strength(self, t):
        return (self.r[t] - self.init) / 100.0

    def p_red(self, red, blue):
        d = np.mean([self.r[t] for t in red]) - np.mean([self.r[t] for t in blue])
        return 1.0 / (1.0 + 10 ** (-d / self.scale))

    def update(self, red, blue, rs, bs):
        p = self.p_red(red, blue)
        outcome = 1.0 if rs > bs else (0.0 if rs < bs else 0.5)
        mov = (1.0 + 0.5 * math.log1p(abs(rs - bs) / max(1.0, (rs + bs) / 2))) if self.margin else 1.0
        d = self.k * mov * (outcome - p)
        for t in red:
            self.r[t] += d
        for t in blue:
            self.r[t] -= d


class EPA:
    name = "epa"
    normalize_scores = True   # scores are divided by the season mean so every year is on the same scale

    def __init__(self, carry=0.6, lr0=0.5, lr_min=0.04, prior_n=3.0):
        self.carry, self.lr0, self.lr_min, self.prior_n = carry, lr0, lr_min, prior_n
        self.e = defaultdict(float)      # contribution above average, normalised units
        self.n = defaultdict(int)
        self.base = 1.0                  # avg alliance score / 1.0 (scores are normalised)
        self.sigma = 0.35                # residual std of margin (normalised); updated online
        self._ss, self._cnt = 0.0, 0

    def new_season(self):
        for t in list(self.e):
            self.e[t] *= self.carry
            self.n[t] = 0
        self._ss, self._cnt = 0.0, 0

    def strength(self, t):
        return self.e[t] * 3.0           # per-team contribution in units of "avg alliance score"

    def _pred(self, ts):
        return self.base + sum(self.e[t] for t in ts)

    def margin(self, red, blue):
        return self._pred(red) - self._pred(blue)

    def p_red(self, red, blue):
        return _phi(self.margin(red, blue) / max(self.sigma, 1e-3))

    def update(self, red, blue, rs, bs):
        # rs / bs must already be normalised by the year's mean alliance score
        for ts, s in ((red, rs), (blue, bs)):
            err = s - self._pred(ts)
            for t in ts:
                lr = max(self.lr_min, self.lr0 / (self.n[t] + self.prior_n))
                self.e[t] += lr * err / 3.0
        for t in red + blue:
            self.n[t] += 1
        m = (rs - bs) - self.margin(red, blue)
        self._ss += m * m
        self._cnt += 1
        if self._cnt > 50:
            self.sigma = max(0.15, math.sqrt(self._ss / self._cnt) / math.sqrt(1))


class BT:
    name = "bt"

    def __init__(self, lr=0.08, l2=0.002, carry=0.6):
        self.lr, self.l2, self.carry = lr, l2, carry
        self.w = defaultdict(float)
        self.n = defaultdict(int)

    def new_season(self):
        for t in list(self.w):
            self.w[t] *= self.carry
            self.n[t] = 0

    def strength(self, t):
        return self.w[t]

    def p_red(self, red, blue):
        return _sigmoid(sum(self.w[t] for t in red) - sum(self.w[t] for t in blue))

    def update(self, red, blue, rs, bs):
        y = 1.0 if rs > bs else (0.0 if rs < bs else 0.5)
        g = y - self.p_red(red, blue)
        for t in red:
            lr = max(self.lr * 0.3, self.lr * 6 / (self.n[t] + 6))
            self.w[t] += lr * (g - self.l2 * self.w[t])
            self.n[t] += 1
        for t in blue:
            lr = max(self.lr * 0.3, self.lr * 6 / (self.n[t] + 6))
            self.w[t] += lr * (-g - self.l2 * self.w[t])
            self.n[t] += 1





def chronological_matches(data, years=None, qual_only=False):
    """Yield (year, event, match) in time order. Events sorted by start date."""
    evs = [e for e in data["events"] if (years is None or e["year"] in years)
           and e["type"] in (0, 1, 2, 3, 4, 5, 99)]
    evs.sort(key=lambda e: (e["year"], e["start"], e["key"]))
    for e in evs:
        ms = [m for m in data["matches"].get(e["key"], []) if m["rs"] is not None and m["bs"] is not None]
        if qual_only:
            ms = [m for m in ms if m["lvl"] == "qm"]
        order = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}
        ms.sort(key=lambda m: (order.get(m["lvl"], 9), m["set"], m["num"]))
        for m in ms:
            yield e["year"], e, m


def year_mean_scores(data):
    out = {}
    acc = defaultdict(list)
    for y, e, m in chronological_matches(data):
        acc[y].append(m["rs"])
        acc[y].append(m["bs"])
    for y, v in acc.items():
        out[y] = max(1.0, float(np.mean(v)))
    return out



class PlainElo(Elo):
    """Win/loss-only Elo (no margin). The default 'elo' above is margin-adjusted (mElo)."""
    name = "elo_plain"

    def __init__(self, k=16.0, carry=0.65, **kw):
        super().__init__(k=k, carry=carry, margin=False)


class ACElo(Elo):
    """Elo with a dynamic K: large for a team's first matches at an event, decaying as it plays,
    and damped when the matchup is lopsided (low information). Interpretation of 'AC-Elo' from its description."""
    name = "ac_elo"

    def __init__(self, k=8.0, boost=2.0, tau=4.0, gapdamp=0.5, carry=0.7, **kw):
        super().__init__(k=k, carry=carry, margin=True)
        self.boost, self.tau, self.gapdamp = boost, tau, gapdamp
        self.ne = defaultdict(int)

    def start_event(self, e):
        self.ne = defaultdict(int)

    def update(self, red, blue, rs, bs):
        p = self.p_red(red, blue)
        outcome = 1.0 if rs > bs else (0.0 if rs < bs else 0.5)
        d = np.mean([self.r[t] for t in red]) - np.mean([self.r[t] for t in blue])
        n = np.mean([self.ne[t] for t in red + blue])
        kk = self.k * (1 + self.boost * math.exp(-n / self.tau)) / (1 + self.gapdamp * abs(d) / 400.0)
        mov = 1.0 + 0.5 * math.log1p(abs(rs - bs) / max(1.0, (rs + bs) / 2))
        delta = kk * mov * (outcome - p)
        for t in red:
            self.r[t] += delta; self.ne[t] += 1
        for t in blue:
            self.r[t] -= delta; self.ne[t] += 1


class EventReg:
    """Within-event regression of alliance score on team contributions, refit every few matches.
    All scores are normalised by the season mean first.
      kind='pridge' : ridge anchored to each team's carried-over prior (trusts the prior early, the event data later)
      kind='ropr'   : ridge OPR, prior = 0 (event data only)
      kind='wopr'   : ridge OPR with recency weights inside the event (decay per match), prior = 0
      kind='lasso'  : L1-penalised OPR, prior = 0
      kind='dpr'    : pRidge with an extra per-team defence term (score_red = off_red - def_blue)
    Priors are updated at the end of each event; between seasons they shrink by `carry`."""
    normalize_scores = True

    def __init__(self, kind="pridge", lam=6.0, carry=0.6, eta=1.0, decay=1.0, alpha=0.01, every=8, **kw):
        self.kind, self.name = kind, kind
        self.lam, self.carry, self.eta, self.decay, self.alpha, self.every = lam, carry, eta, decay, alpha, every
        self.uses_prior = kind in ("pridge", "dpr")
        self.defense = kind == "dpr"
        self.off, self.dfn = defaultdict(float), defaultdict(float)
        self.se, self.sw = defaultdict(float), defaultdict(float)
        self.dOff, self.dDef = {}, {}
        self.rows, self.since = [], 0
        self.sigma, self._ss, self._cnt = 0.35, 0.0, 0

    # --- season / event lifecycle
    def new_season(self):
        for t in list(self.off):
            self.off[t] *= self.carry
        for t in list(self.dfn):
            self.dfn[t] *= self.carry
        self.se.clear(); self.sw.clear()
        self._ss, self._cnt = 0.0, 0

    def start_event(self, e):
        self.rows, self.dOff, self.dDef, self.since = [], {}, {}, 0

    def end_event(self):
        if self.rows:
            if self.since:
                self._refit()
            cnt = defaultdict(int)
            for r in self.rows:
                for t in r[0] + r[1]:
                    cnt[t] += 1
            if self.uses_prior:
                for t, d in self.dOff.items():
                    self.off[t] += self.eta * d
                for t, d in self.dDef.items():
                    self.dfn[t] += self.eta * d
            else:
                for t, d in self.dOff.items():
                    w = cnt[t]
                    self.se[t] = (self.se[t] * self.sw[t] + d * w) / (self.sw[t] + w)
                    self.sw[t] += w
        self.rows, self.dOff, self.dDef, self.since = [], {}, {}, 0

    # --- ratings
    def _s(self, t):
        base = (self.off[t] + self.dfn[t]) if self.uses_prior else 0.0
        return base + self.dOff.get(t, 0.0) + self.dDef.get(t, 0.0)

    def strength(self, t):
        return self._s(t)

    def season_strength(self, t):
        return (self.off[t] + self.dfn[t]) if self.uses_prior else self.se[t]

    def margin(self, red, blue):
        return sum(self._s(t) for t in red) - sum(self._s(t) for t in blue)

    def p_red(self, red, blue):
        return _phi(self.margin(red, blue) / max(self.sigma, 1e-3))

    def update(self, red, blue, rs, bs):
        r = (rs - bs) - self.margin(red, blue)
        self._ss += r * r; self._cnt += 1
        if self._cnt > 50:
            self.sigma = max(0.15, math.sqrt(self._ss / self._cnt))
        self.rows.append((red, blue, rs, bs))
        self.since += 1
        if self.since >= self.every:
            self._refit(); self.since = 0

    def _refit(self):
        rows = self.rows
        k = len(rows)
        R = np.array([r[0] for r in rows]); B = np.array([r[1] for r in rows])
        teams = np.unique(np.concatenate([R.ravel(), B.ravel()]))
        T = len(teams)
        Ri, Bi = np.searchsorted(teams, R), np.searchsorted(teams, B)
        ncol = 2 * T if self.defense else T
        A = np.zeros((2 * k, ncol))
        rr, rb = np.arange(k) * 2, np.arange(k) * 2 + 1
        for c in range(3):
            A[rr, Ri[:, c]] = 1.0
            A[rb, Bi[:, c]] = 1.0
            if self.defense:
                A[rr, T + Bi[:, c]] = -1.0
                A[rb, T + Ri[:, c]] = -1.0
        y = np.empty(2 * k); y[0::2] = [r[2] for r in rows]; y[1::2] = [r[3] for r in rows]
        prior = np.zeros(ncol)
        if self.uses_prior:
            prior[:T] = [self.off[t] for t in teams]
            if self.defense:
                prior[T:] = [self.dfn[t] for t in teams]
        yy = y - A @ prior
        yy = yy - yy.mean()                       # event-level scoring offset
        w = np.repeat(self.decay ** (k - 1 - np.arange(k)), 2) if self.decay < 1.0 else np.ones(2 * k)
        if self.kind == "lasso":
            from sklearn.linear_model import Lasso
            delta = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=3000).fit(A, yy, sample_weight=w).coef_
        else:
            Aw = A * w[:, None]
            delta = np.linalg.solve(A.T @ Aw + self.lam * np.eye(ncol), Aw.T @ yy)
        self.dOff = {int(teams[i]): float(delta[i]) for i in range(T)}
        if self.defense:
            self.dDef = {int(teams[i]): float(delta[T + i]) for i in range(T)}


class _Reg(EventReg):
    kind_fixed = None

    def __init__(self, **kw):
        super().__init__(kind=self.kind_fixed, **kw)


def _mk(kind):
    return type(f"Reg_{kind}", (_Reg,), {"kind_fixed": kind})


MODELS = {"elo": Elo, "elo_plain": PlainElo, "ac_elo": ACElo, "epa": EPA, "bt": BT,
          "pridge": _mk("pridge"), "ropr": _mk("ropr"), "wopr": _mk("wopr"), "lasso": _mk("lasso"), "dpr": _mk("dpr")}

GRID = {
    "elo": [{"k": k, "carry": c} for k in (6, 12, 20) for c in (0.5, 0.7)],
    "elo_plain": [{"k": k, "carry": 0.7} for k in (12, 24)],
    "ac_elo": [{"k": k, "boost": 2.0, "carry": 0.7} for k in (5, 10)],
    "epa": [{"carry": c, "lr0": l} for c in (0.5, 0.7) for l in (0.3, 0.6)],
    "bt": [{"lr": l, "carry": c} for l in (0.04, 0.08) for c in (0.5, 0.7)],
    "pridge": [{"lam": l, "carry": c} for l in (3.0, 8.0) for c in (0.5, 0.7)],
    "ropr": [{"lam": l} for l in (1.5, 5.0)],
    "wopr": [{"lam": l, "decay": 0.96} for l in (1.5, 5.0)],
    "lasso": [{"alpha": a} for a in (0.004, 0.012)],
    "dpr": [{"lam": l, "carry": 0.7} for l in (4.0, 10.0)],
}


def run(model, data, years=None, offseason=False, on_match=None):
    """One chronological pass (each match is predicted BEFORE the model sees its result).
    Returns (metrics per year, end-of-season strengths per year). Per-team match counts per year are left on
    model.season_counts so ratings can be normalised over teams with enough matches."""
    ymean = year_mean_scores(data)
    res = defaultdict(lambda: {"ll": 0.0, "br": 0.0, "ok": 0, "n": 0})
    cur, cur_ev, end_strength = None, None, {}
    seen = defaultdict(set)
    counts = defaultdict(lambda: defaultdict(int))
    snap = getattr(model, "season_strength", model.strength)
    norm = getattr(model, "normalize_scores", False)

    def close_event():
        if hasattr(model, "end_event"):
            model.end_event()

    for y, e, m in chronological_matches(data, years):
        if e["type"] == 99 and not offseason:
            continue
        if e["key"] != cur_ev:
            close_event()
            if y != cur:
                if cur is not None:
                    end_strength[cur] = {t: snap(t) for t in seen[cur]}
                    model.new_season()
                cur = y
            cur_ev = e["key"]
            if hasattr(model, "start_event"):
                model.start_event(e)
        red, blue = m["red"], m["blue"]
        seen[y].update(red + blue)
        for t in red + blue:
            counts[y][t] += 1
        rs, bs = m["rs"], m["bs"]
        p = min(max(model.p_red(red, blue), 1e-4), 1 - 1e-4)
        if on_match:
            on_match(y, e, m, p)
        if rs != bs:
            o = 1.0 if rs > bs else 0.0
            r = res[y]
            r["ll"] += -(o * math.log(p) + (1 - o) * math.log(1 - p))
            r["br"] += (p - o) ** 2
            r["ok"] += int((p > 0.5) == (o == 1.0))
            r["n"] += 1
        if norm:
            model.update(red, blue, rs / ymean[y], bs / ymean[y])
        else:
            model.update(red, blue, rs, bs)
    close_event()
    if cur is not None:
        end_strength[cur] = {t: snap(t) for t in seen[cur]}
    model.season_counts = {y: dict(c) for y, c in counts.items()}
    metrics = {y: {"logloss": v["ll"] / v["n"], "brier": v["br"] / v["n"], "acc": v["ok"] / v["n"], "n": v["n"]}
               for y, v in res.items() if v["n"]}
    return metrics, end_strength


def backtest(data, test_years, grid=None, models=None, return_all=False, log=None):
    """Tune each model's hyper-parameters on seasons BEFORE the test years only, then score every model on the
    test years. Returns (leaderboard sorted by test log loss, best_name, best_params)
    (+ per-match predictions / strengths / counts when return_all)."""
    all_years = sorted({e["year"] for e in data["events"]})
    train_years = [y for y in all_years if y < min(test_years)]
    grid = grid or {k: v for k, v in GRID.items() if (models is None or k in models)}
    board, preds, ends, counts = [], {}, {}, {}
    for name, plist in grid.items():
        best = None
        for p in plist:
            m, _ = run(MODELS[name](**p), data, years=set(train_years) if train_years else None)
            tl = float(np.mean([v["logloss"] for v in m.values()])) if m else 9.0
            if best is None or tl < best[0]:
                best = (tl, p)
        rec = {}
        mdl = MODELS[name](**best[1])
        m, es = run(mdl, data, on_match=lambda y, e, mt, p, rec=rec: rec.__setitem__(mt["key"], p))
        preds[name], ends[name], counts = rec, es, mdl.season_counts
        rows = [m[y] for y in test_years if y in m]
        board.append({"model": name, "params": best[1], "train_logloss": best[0],
                      "test_logloss": float(np.mean([r["logloss"] for r in rows])),
                      "test_brier": float(np.mean([r["brier"] for r in rows])),
                      "test_acc": float(np.mean([r["acc"] for r in rows])),
                      "test_matches": int(sum(r["n"] for r in rows))})
        if log:
            log(f"  {name}: test log loss {board[-1]['test_logloss']:.4f}, acc {board[-1]['test_acc']:.3f}")
    board.sort(key=lambda r: r["test_logloss"])
    if return_all:
        return board, board[0]["model"], board[0]["params"], {"preds": preds, "ends": ends, "counts": counts}
    return board, board[0]["model"], board[0]["params"]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def select_model(data, test_years, grid=None, n_members=3, log=None):
    """Evaluate every candidate, then try a stacked ensemble of the best few (weights >= 0, fit on the training
    seasons only). The ensemble is used ONLY if it beats the best single model on the held-out seasons.
    Returns dict(board, selected, ends, counts, preds)."""
    from scipy.optimize import minimize
    board, best, params, aux = backtest(data, test_years, grid, return_all=True, log=log)
    preds, ends, counts = aux["preds"], aux["ends"], aux["counts"]
    all_years = sorted({e["year"] for e in data["events"]})
    train_years = {y for y in all_years if y < min(test_years)}
    keys, yrs, out = [], [], []
    for y, e, m in chronological_matches(data):
        if e["type"] == 99 or m["rs"] == m["bs"]:
            continue
        keys.append(m["key"]); yrs.append(y); out.append(1.0 if m["rs"] > m["bs"] else 0.0)
    yrs, out = np.array(yrs), np.array(out)
    by_train = sorted(board, key=lambda r: r["train_logloss"])[:n_members]
    names = [r["model"] for r in by_train]
    X = np.column_stack([_logit(np.array([preds[n].get(k, 0.5) for k in keys])) for n in names])
    tr, te = np.isin(yrs, list(train_years)), np.isin(yrs, test_years)
    selected = {"kind": "single", "name": best, "params": params}
    if tr.sum() > 500 and te.sum() > 500 and len(names) >= 2:
        def nll(w, A, y):
            z = A @ w
            return float(np.mean(np.logaddexp(0, z) - y * z))
        w0 = np.ones(len(names)) / len(names)
        w = minimize(nll, w0, args=(X[tr], out[tr]), bounds=[(0, 3)] * len(names), method="L-BFGS-B").x
        z = X[te] @ w
        p = 1 / (1 + np.exp(-z))
        row = {"model": "ensemble(" + "+".join(names) + ")", "params": None,
               "train_logloss": nll(w, X[tr], out[tr]),
               "test_logloss": nll(w, X[te], out[te]),
               "test_brier": float(np.mean((p - out[te]) ** 2)),
               "test_acc": float(np.mean((p > 0.5) == (out[te] == 1.0))),
               "test_matches": int(te.sum()),
               "members": [{"model": n, "weight": float(wi), "params": next(r["params"] for r in board if r["model"] == n)}
                           for n, wi in zip(names, w)]}
        board.append(row)
        if row["test_logloss"] < min(r["test_logloss"] for r in board[:-1]) - 5e-4:
            selected = {"kind": "ensemble", "members": row["members"]}
    board.sort(key=lambda r: r["test_logloss"])
    for r in board:
        r["selected"] = (selected["kind"] == "single" and r["model"] == selected["name"]) or \
                        (selected["kind"] == "ensemble" and r["model"].startswith("ensemble"))
    return {"board": board, "selected": selected, "ends": ends, "counts": counts, "preds": preds}


def selected_strengths(sel):
    """End-of-season strengths of the selected model; for an ensemble, a weighted mix of members' season z-scores."""
    from .forecast import zscores
    s, ends, counts = sel["selected"], sel["ends"], sel["counts"]
    if s["kind"] == "single":
        return ends[s["name"]]
    tot = sum(m["weight"] for m in s["members"]) or 1.0
    zs = {m["model"]: zscores(ends[m["model"]], counts) for m in s["members"]}
    years = set.intersection(*[set(z) for z in zs.values()])
    return {y: {t: sum(m["weight"] / tot * zs[m["model"]][y][t] for m in s["members"] if t in zs[m["model"]][y])
                for t in zs[s["members"][0]["model"]][y]} for y in years}


def selected_match_p(sel, key):
    s, preds = sel["selected"], sel["preds"]
    if s["kind"] == "single":
        return preds[s["name"]].get(key)
    z = sum(m["weight"] * float(_logit(np.array([preds[m["model"]].get(key, 0.5)]))[0]) for m in s["members"])
    return 1 / (1 + math.exp(-z))
