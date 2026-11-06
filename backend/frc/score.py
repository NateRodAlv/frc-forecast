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
            "logloss": ll, "logloss_base": ll0, "bins": bins}


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


def playoff_format(matches):
    """'single8' (best-of-3 single elimination, 8 alliances: quarterfinal sets 1-4, semifinal sets 1-2, final),
    'double' (double elimination: semifinal sets beyond 2), or 'other' / None when there are no playoff matches.
    Decided from the match records themselves so a season is only scored when the simulated bracket matches what was played."""
    sets = {(m["lvl"], m["set"]) for m in matches if m["lvl"] in ("qf", "sf", "f")}
    if not sets:
        return None
    if any(l == "sf" and s > 2 for l, s in sets):
        return "double"
    if {s for l, s in sets if l == "qf"} == {1, 2, 3, 4} and {s for l, s in sets if l == "sf"} == {1, 2}:
        return "single8"
    return "other"


def score_events(year, data, season_events, fmt):
    """Event win / finalist probabilities vs actual finals results, over events whose played bracket is `fmt`
    (the format the simulator used for that season). Events with another format are counted in `skipped_format`."""
    pw, ow, pf, of = [], [], [], []
    n_ev = skipped = 0
    for e in data["events"]:
        sim = season_events.get(e["key"])
        if e["year"] != year or not sim:
            continue
        ms = data["matches"].get(e["key"], [])
        if playoff_format(ms) != fmt:
            skipped += playoff_format(ms) is not None
            continue
        res = finals_outcome(ms)
        if not res:
            continue
        n_ev += 1
        win, lose = res
        for t, v in sim["teams"].items():
            pw.append(v["p_win"]); ow.append(1.0 if int(t) in win else 0.0)
            pf.append(v["p_finalist"]); of.append(1.0 if int(t) in win or int(t) in lose else 0.0)
    return {"events": n_ev, "skipped_format": skipped, "format": fmt, "win": prob_metrics(pw, ow), "finalist": prob_metrics(pf, of)}


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
            rows.append((f["mean"], f["z_last"], f["q10"], f["q90"], z_actual[t], band, f.get("sd")))
    if len(rows) < 50:
        return None
    a = np.array([r[:5] for r in rows], float)
    y = a[:, 4]
    sd = np.array([r[6] if r[6] is not None else np.nan for r in rows], float)

    def rm(p, m=None):
        m = np.ones(len(y), bool) if m is None else m
        return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
    out = {"n": len(rows), "rmse_forecast": rm(a[:, 0]), "rmse_persistence": rm(a[:, 1]), "rmse_always_average": rm(np.zeros(len(y))),
           "coverage_10_90": float(np.mean((y >= a[:, 2]) & (y <= a[:, 3]))),
           # the interval the simulator actually uses: forecast mean +/- 1.2816 sd (ideal 80%)
           "coverage_sd_80": float(np.mean(np.abs(y - a[:, 0]) < 1.2816 * sd)) if not np.isnan(sd).any() else None,
           "spearman": float(spearmanr(a[:, 0], y)[0]), "by_age": {}}
    bands = np.array([r[5] for r in rows])
    for b in sorted(set(bands)):
        m = bands == b
        if m.sum() >= 30:
            out["by_age"][b] = {"n": int(m.sum()), "forecast": rm(a[:, 0], m), "persistence": rm(a[:, 1], m)}
    return out


def score_advancement(year, data, season, alias):
    """District Championship / Worlds odds vs who actually showed up. Attendance is a proxy for qualification
    (it ignores declines and waitlist moves). Per-district expected-vs-actual slot counts are deliberately not
    reported: the simulator fills each district's allocation exactly, so they match by construction."""
    dcmp = {}
    for e in data["events"]:
        if e["year"] == year and e["type"] in (2, 5) and e.get("district"):
            k = alias.get(e["district"].lower())
            if k:
                dcmp.setdefault(k, set()).update(data["event_teams"].get(e["key"], []))
    cmp_ = {t for e in data["events"] if e["year"] == year and e["type"] in (3, 4) for t in data["event_teams"].get(e["key"], [])}
    have_dcmp, have_cmp = any(dcmp.values()), bool(cmp_)
    d_p, d_o, c_p, c_o, r_p, r_o = [], [], [], [], [], []
    for t, v in season["teams"].items():
        if v["model"] == "district":
            d_p.append(v["p_dcmp"]); d_o.append(1.0 if t in dcmp.get(v["district"], set()) else 0.0)
            c_p.append(v["p_cmp"]); c_o.append(1.0 if t in cmp_ else 0.0)
        else:
            r_p.append(v["p_cmp"]); r_o.append(1.0 if t in cmp_ else 0.0)
    return {"dcmp": prob_metrics(d_p, d_o) if have_dcmp else None,
            "cmp_district": prob_metrics(c_p, c_o) if have_cmp else None,
            "cmp_regional": prob_metrics(r_p, r_o) if have_cmp else None,
            "worlds_attendees_found": len(cmp_), "dcmp_rosters_found": sum(len(v) for v in dcmp.values())}


# ---- pooling: combine several seasons' scores into one summary --------------------------------------------------
def _wmean(items, key):
    n = sum(i["n"] for i in items)
    return sum(i["n"] * i[key] for i in items) / n


def pool_prob(pms):
    """Pool prob_metrics() results from several seasons. Calibration bins merge exactly; Brier / log loss are
    case-weighted means, and the skill baseline is the case-weighted mean of each season's own base-rate Brier."""
    pms = [p for p in pms if p]
    if not pms:
        return None
    n = sum(p["n"] for p in pms)
    brier, b0 = _wmean(pms, "brier"), _wmean(pms, "brier_base")
    merged = {}
    for p in pms:
        for b in p["bins"]:
            m = merged.setdefault(b["range"], {"range": b["range"], "n": 0, "pred": 0.0, "actual": 0.0})
            m["n"] += b["n"]; m["pred"] += b["n"] * b["pred"]; m["actual"] += b["n"] * b["actual"]
    order = {f"{lo*100:.0f}-{min(hi,1)*100:.0f}%": i for i, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:]))}
    bins = [{"range": r, "n": m["n"], "pred": m["pred"] / m["n"], "actual": m["actual"] / m["n"]}
            for r, m in sorted(merged.items(), key=lambda kv: order.get(kv[0], 99))]
    return {"n": n, "seasons": len(pms), "base_rate": _wmean(pms, "base_rate"), "brier": brier, "brier_base": b0,
            "skill": (1 - brier / b0) if b0 > 0 else None, "logloss": _wmean(pms, "logloss"),
            "logloss_base": _wmean(pms, "logloss_base"), "bins": bins}


def pool_replays(replays):
    """One summary across all replayed seasons. Event and advancement scores pool only the seasons that were scored."""
    if not replays:
        return None
    ev = [r["events"] for r in replays if r.get("events")]
    adv = [r["advancement"] for r in replays if r.get("advancement")]
    return {"seasons": [r["year"] for r in replays],
            "strength": pool_strength([r.get("strength") for r in replays]),
            "matches": pool_matches([r.get("matches") for r in replays]),
            "events": {"seasons": [r["year"] for r in replays if r.get("events")], "events": sum(e["events"] for e in ev),
                       "win": pool_prob([e["win"] for e in ev]), "finalist": pool_prob([e["finalist"] for e in ev])} if ev else None,
            "advancement": {"seasons": [r["year"] for r in replays if r.get("advancement")],
                            **{k: pool_prob([a[k] for a in adv if a.get(k)]) for k in ("dcmp", "cmp_district", "cmp_regional")}} if adv else None}


def pool_matches(ms):
    ms = [m for m in ms if m]
    if not ms:
        return None
    out = {"n": sum(m["n"] for m in ms), "seasons": len(ms), "logloss_coinflip": math.log(2)}
    for k in ("accuracy", "logloss", "brier"):
        out[k] = _wmean(ms, k)
    ins = [m for m in ms if m.get("inseason_logloss") is not None]
    if ins:
        out["inseason_logloss"] = _wmean(ins, "inseason_logloss"); out["inseason_accuracy"] = _wmean(ins, "inseason_accuracy")
    return out


def pool_strength(ss):
    ss = [s for s in ss if s]
    if not ss:
        return None
    n = sum(s["n"] for s in ss)
    out = {"n": n, "seasons": len(ss), "coverage_10_90": _wmean(ss, "coverage_10_90"), "spearman": _wmean(ss, "spearman"), "by_age": {}}
    sdc = [s for s in ss if s.get("coverage_sd_80") is not None]
    if sdc:
        out["coverage_sd_80"] = _wmean(sdc, "coverage_sd_80")
    for k in ("rmse_forecast", "rmse_persistence", "rmse_always_average"):
        out[k] = math.sqrt(sum(s["n"] * s[k] ** 2 for s in ss) / n)
    for band in {b for s in ss for b in s["by_age"]}:
        parts = [(s["by_age"][band]) for s in ss if band in s["by_age"]]
        m = sum(p["n"] for p in parts)
        out["by_age"][band] = {"n": m, "forecast": math.sqrt(sum(p["n"] * p["forecast"] ** 2 for p in parts) / m),
                               "persistence": math.sqrt(sum(p["n"] * p["persistence"] ** 2 for p in parts) / m)}
    return out
