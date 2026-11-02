"""Synthetic FRC-like dataset, used ONLY for tests / demo when no TBA key is available.

Teams have a latent skill that follows a mean-reverting random walk with occasional
"breakout" jumps and "collapse" drops, so the trajectory forecaster has real structure
to find. Output mimics the slim dict shapes returned by tba.py.
"""
import numpy as np


def make_dataset(n_teams=600, years=range(2015, 2027), seed=7, matches_per_team=10):
    rng = np.random.default_rng(seed)
    teams = list(range(1, n_teams + 1))
    rookie = {t: int(rng.integers(years[0] - 6, years[-1] + 1)) if t > 150 else years[0] - 10 for t in teams}
    skill = {t: rng.normal(0, 1) for t in teams}
    events, matches, ev_teams, state = [], {}, {}, {}
    for yi, y in enumerate(years):
        active = [t for t in teams if rookie[t] <= y]
        # latent dynamics: momentum + mean reversion + regime jumps
        for t in teams:
            if rookie[t] > y:
                continue
            prev = state.get(t, {"s": skill[t], "d": 0.0})
            jump = 0.0
            r = rng.random()
            if prev["d"] < -0.35 and r < 0.30:       # slump -> rebound
                jump = rng.normal(1.3, 0.3)
            elif r < 0.03:
                jump = rng.normal(-1.0, 0.3)
            new = 0.65 * prev["s"] + 0.15 * prev["d"] + rng.normal(0, 0.35) + jump
            state[t] = {"s": new, "d": new - prev["s"]}
        scale = 12.0
        order = rng.permutation(active)
        n_ev = max(1, len(active) // 40)
        for e in range(n_ev):
            chunk = [int(t) for t in order[e::n_ev]]
            key = f"{y}ev{e:02d}"
            events.append({"key": key, "year": y, "name": f"Synthetic Event {e}", "type": 1,
                           "type_name": "district", "district": "syn", "start": f"{y}-03-{1 + e % 25:02d}",
                           "end": f"{y}-03-{3 + e % 25:02d}", "week": 1 + e % 6, "country": "USA"})
            ev_teams[key] = chunk
            ms = []
            nm = len(chunk) * matches_per_team // 6
            for k in range(nm):
                ts = [int(x) for x in rng.choice(chunk, 6, replace=False)]
                r_, b_ = ts[:3], ts[3:]
                rs = 30 + scale * sum(state[t]["s"] for t in r_) + rng.normal(0, 14)
                bs = 30 + scale * sum(state[t]["s"] for t in b_) + rng.normal(0, 14)
                ms.append({"key": f"{key}_qm{k+1}", "lvl": "qm", "set": 1, "num": k + 1, "time": k,
                           "red": r_, "blue": b_, "rs": max(0, round(rs)), "bs": max(0, round(bs))})
            matches[key] = ms
    return {"events": events, "matches": matches, "event_teams": ev_teams, "rookie": rookie}
