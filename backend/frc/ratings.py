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

    def __init__(self, k=12.0, carry=0.65, scale=400.0, init=1500.0):
        self.k, self.carry, self.scale, self.init = k, carry, scale, init
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
        mov = 1.0 + 0.5 * math.log1p(abs(rs - bs) / max(1.0, (rs + bs) / 2))
        d = self.k * mov * (outcome - p)
        for t in red:
            self.r[t] += d
        for t in blue:
            self.r[t] -= d


class EPA:
    name = "epa"

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


MODELS = {"elo": Elo, "epa": EPA, "bt": BT}


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


def run(model, data, years=None, offseason=False, on_match=None):
    """One chronological pass. Returns metrics dict (log loss, brier, acc) per year,
    plus per-year end-of-season strengths."""
    ymean = year_mean_scores(data)
    res = defaultdict(lambda: {"ll": 0.0, "br": 0.0, "ok": 0, "n": 0})
    cur, end_strength = None, {}
    seen_teams = defaultdict(set)
    for y, e, m in chronological_matches(data, years):
        if y != cur:
            if cur is not None:
                end_strength[cur] = {t: model.strength(t) for t in seen_teams[cur]}
                model.new_season()
            cur = y
        if e["type"] == 99 and not offseason:
            continue
        red, blue = m["red"], m["blue"]
        seen_teams[y].update(red + blue)
        rs, bs = m["rs"], m["bs"]
        p = min(max(model.p_red(red, blue), 1e-4), 1 - 1e-4)
        if on_match:
            on_match(y, e, m, p)
        if rs == bs:
            continue
        o = 1.0 if rs > bs else 0.0
        r = res[y]
        r["ll"] += -(o * math.log(p) + (1 - o) * math.log(1 - p))
        r["br"] += (p - o) ** 2
        r["ok"] += int((p > 0.5) == (o == 1.0))
        r["n"] += 1
        if model.name == "epa":
            model.update(red, blue, rs / ymean[y], bs / ymean[y])
        else:
            model.update(red, blue, rs, bs)
    if cur is not None:
        end_strength[cur] = {t: model.strength(t) for t in seen_teams[cur]}
    metrics = {y: {"logloss": v["ll"] / v["n"], "brier": v["br"] / v["n"], "acc": v["ok"] / v["n"], "n": v["n"]}
               for y, v in res.items() if v["n"]}
    return metrics, end_strength


def backtest(data, test_years, grid=None):
    """Compare models across hyper-parameters. Tune on years < min(test_years) only,
    report on test years. Returns (leaderboard, best_name, best_params)."""
    all_years = sorted({e["year"] for e in data["events"]})
    train_years = [y for y in all_years if y < min(test_years)]
    grid = grid or {
        "elo": [{"k": k, "carry": c} for k in (6, 12, 20) for c in (0.5, 0.7)],
        "epa": [{"carry": c, "lr0": l} for c in (0.5, 0.7) for l in (0.3, 0.6)],
        "bt": [{"lr": l, "carry": c} for l in (0.04, 0.08) for c in (0.5, 0.7)],
    }
    board = []
    for name, plist in grid.items():
        best = None
        for p in plist:
            m, _ = run(MODELS[name](**p), data, years=set(train_years) if train_years else None)
            tl = np.mean([v["logloss"] for v in m.values()]) if m else 9
            if best is None or tl < best[0]:
                best = (tl, p)
        m, _ = run(MODELS[name](**best[1]), data)
        rows = [m[y] for y in test_years if y in m]
        board.append({"model": name, "params": best[1],
                      "train_logloss": best[0],
                      "test_logloss": float(np.mean([r["logloss"] for r in rows])),
                      "test_brier": float(np.mean([r["brier"] for r in rows])),
                      "test_acc": float(np.mean([r["acc"] for r in rows])),
                      "test_matches": int(sum(r["n"] for r in rows))})
    board.sort(key=lambda r: r["test_logloss"])
    return board, board[0]["model"], board[0]["params"]
