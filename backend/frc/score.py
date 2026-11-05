"""Scoring: compare predictions made BEFORE a season with what actually happened.
Baseline for every probability is the observed base rate (a hard-to-beat 'climatology' guess);
skill = 1 - Brier / Brier_baseline (>0 means better than that baseline)."""
import math

import numpy as np

BINS = [0, 0.01, 0.03, 0.07, 0.15, 0.30, 0.50, 0.75, 1.0001]


def prob_metrics(ps, os_):
    ps, os_ = np.clip(np.array(ps, float), 1e-4, 1 - 1e-4), np.array(os_, float)
    n = len(ps)
    if n < 20:
        return None
    base = float(os_.mean())
    brier = float(np.mean((ps - os_) ** 2))
    b0 = float(np.mean((base - os_) ** 2))
    ll = float(-np.mean(os_ * np.log(ps) + (1 - os_) * np.log(1 - ps)))
    bc = min(max(base, 1e-4), 1 - 1e-4)
    ll0 = float(-np.mean(os_ * math.log(bc) + (1 - os_) * math.log(1 - bc)))
    bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (ps >= lo) & (ps < hi)
        if m.sum():
            bins.append({"range": f"{lo*100:.0f}-{min(hi,1)*100:.0f}%", "n": int(m.sum()),
                         "pred": float(ps[m].mean()), "actual": float(os_[m].mean())})
    return {"n": n, "base_rate": base, "brier": brier, "brier_base": b0, "skill": (1 - brier / b0) if b0 > 0 else None,
            "logloss": ll, "logloss_base": ll0, "expected": float(ps.sum()), "actual": float(os_.sum()), "bins": bins}


def finals_outcome(matches):
    """(winning alliance teams, losing alliance teams) from the finals matches, or None if undecided."""
    f = [m for m in matches if m["lvl"] == "f" and m["rs"] is not None and m["rs"] != m["bs"]]
    wins = {"red": 0, "blue": 0}
    teams = {"red": set(), "blue": set()}
    for m in f:
        wins["red" if m["rs"] > m["bs"] else "blue"] += 1
        teams["red"].update(m["red"]); teams["blue"].update(m["blue"])
    if max(wins.values()) < 2:
        return None
    w = "red" if wins["red"] > wins["blue"] else "blue"
    return teams[w], teams["blue" if w == "red" else "red"]


def score_events(year, data, season_events):
    """Event win / finalist probabilities vs actual finals results."""
    pw, ow, pf, of = [], [], [], []
    n_ev = 0
    for e in data["events"]:
        sim = season_events.get(e["key"])
        if e["year"] != year or not sim:
            continue
        res = finals_outcome(data["matches"].get(e["key"], []))
        if not res:
            continue
        n_ev += 1
        win, lose = res
        for t, v in sim["teams"].items():
            pw.append(v["p_win"]); ow.append(1.0 if int(t) in win else 0.0)
            pf.append(v["p_finalist"]); of.append(1.0 if int(t) in win or int(t) in lose else 0.0)
    return {"events": n_ev, "win": prob_metrics(pw, ow), "finalist": prob_metrics(pf, of)}


def score_matches(year, data, strengths, beta, inseason_p=None):
    """Preseason-only match predictions (forecast strengths, no in-season information) vs actual results."""
    from .simulate import predict_matches
    ms = []
    for e in data["events"]:
        if e["year"] == year and e["type"] in (0, 1):
            ms += [m for m in data["matches"].get(e["key"], []) if m["lvl"] == "qm" and m["rs"] is not None and m["rs"] != m["bs"]]
    if len(ms) < 50:
        return None
    pred = predict_matches(ms, strengths, beta)
    p = np.array([x["p_red"] for x in pred]); o = np.array([1.0 if m["rs"] > m["bs"] else 0.0 for m in ms])
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    out = {"n": len(ms), "accuracy": float(np.mean((p > 0.5) == (o == 1.0))),
           "logloss": float(-np.mean(o * np.log(pc) + (1 - o) * np.log(1 - pc))), "logloss_coinflip": math.log(2),
           "brier": float(np.mean((p - o) ** 2))}
    if inseason_p:
        q = np.array([inseason_p(m["key"]) for m in ms], dtype=float)
        ok = ~np.isnan(q)
        if ok.sum() > 50:
            qc = np.clip(q[ok], 1e-4, 1 - 1e-4)
            out["inseason_logloss"] = float(-np.mean(o[ok] * np.log(qc) + (1 - o[ok]) * np.log(1 - qc)))
            out["inseason_accuracy"] = float(np.mean((q[ok] > 0.5) == (o[ok] == 1.0)))
    return out


def score_strength(fc, z_actual, counts_actual, rookie, year, min_matches=8):
    """Forecast next-season rating vs the rating the team actually earned (same-season standardisation)."""
    from scipy.stats import spearmanr
    from .forecast import age_band, age_at
    rows = []
    for t, f in fc.items():
        if t in z_actual and counts_actual.get(t, 0) >= min_matches:
            rk = rookie.get(t)
            band = age_band(age_at(year - 1, rk)) if rk else "?"
            rows.append((f["mean"], f["z_last"], f["q10"], f["q90"], z_actual[t], band))
    if len(rows) < 50:
        return None
    a = np.array([r[:5] for r in rows], float)
    y = a[:, 4]

    def rm(p, m=None):
        m = np.ones(len(y), bool) if m is None else m
        return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
    out = {"n": len(rows), "rmse_forecast": rm(a[:, 0]), "rmse_persistence": rm(a[:, 1]), "rmse_always_average": rm(np.zeros(len(y))),
           "coverage_10_90": float(np.mean((y >= a[:, 2]) & (y <= a[:, 3]))),
           "spearman": float(spearmanr(a[:, 0], y)[0]), "by_age": {}}
    bands = np.array([r[5] for r in rows])
    for b in sorted(set(bands)):
        m = bands == b
        if m.sum() >= 30:
            out["by_age"][b] = {"n": int(m.sum()), "forecast": rm(a[:, 0], m), "persistence": rm(a[:, 1], m)}
    return out


def score_advancement(year, data, season, rules, alias):
    """District Championship / Worlds odds vs who actually showed up. Attendance is a proxy for qualification
    (it ignores declines and waitlist moves)."""
    dcmp = {}
    for e in data["events"]:
        if e["year"] == year and e["type"] in (2, 5) and e.get("district"):
            k = alias.get(e["district"].lower())
            if k:
                dcmp.setdefault(k, set()).update(data["event_teams"].get(e["key"], []))
    cmp_ = {t for e in data["events"] if e["year"] == year and e["type"] in (3, 4) for t in data["event_teams"].get(e["key"], [])}
    have_dcmp, have_cmp = any(dcmp.values()), bool(cmp_)
    d_p, d_o, c_p, c_o, r_p, r_o = [], [], [], [], [], []
    by_dist = {}
    for t, v in season["teams"].items():
        if v["model"] == "district":
            d_p.append(v["p_dcmp"]); d_o.append(1.0 if t in dcmp.get(v["district"], set()) else 0.0)
            c_p.append(v["p_cmp"]); c_o.append(1.0 if t in cmp_ else 0.0)
            b = by_dist.setdefault(v["district"], {"dcmp_exp": 0.0, "dcmp_act": 0, "cmp_exp": 0.0, "cmp_act": 0, "teams": 0})
            b["dcmp_exp"] += v["p_dcmp"]; b["dcmp_act"] += d_o[-1]; b["cmp_exp"] += v["p_cmp"]; b["cmp_act"] += c_o[-1]; b["teams"] += 1
        else:
            r_p.append(v["p_cmp"]); r_o.append(1.0 if t in cmp_ else 0.0)
    slots = [{"district": rules["districts"][k]["name"], "teams": b["teams"], "dcmp_expected": round(b["dcmp_exp"], 1),
              "dcmp_actual": int(b["dcmp_act"]) if have_dcmp else None, "cmp_expected": round(b["cmp_exp"], 1),
              "cmp_actual": int(b["cmp_act"]) if have_cmp else None,
              "cmp_slots_rule": rules["districts"][k]["cmp_slots"]} for k, b in sorted(by_dist.items())]
    return {"dcmp": prob_metrics(d_p, d_o) if have_dcmp else None,
            "cmp_district": prob_metrics(c_p, c_o) if have_cmp else None,
            "cmp_regional": prob_metrics(r_p, r_o) if have_cmp else None,
            "slots": slots, "worlds_attendees_found": len(cmp_), "dcmp_rosters_found": sum(len(v) for v in dcmp.values())}
