"""Everything that turns data from seasons < target into a forecast of season `target`.
The same function serves the real forecast (target = next year) and the replay used for the scorecard
(target = last played year), so the replay exercises exactly the production code path."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from . import forecast, ratings, simulate


def cut(data, before):
    """Data restricted to events of seasons < `before` (matches AND rosters). Used so nothing from the target
    season (or later) can influence ratings, model choice, forecasts or calibration."""
    evs = [e for e in data["events"] if e["year"] < before]
    keys = {e["key"] for e in evs}
    return {**data, "events": evs,
            "matches": {k: v for k, v in data["matches"].items() if k in keys},
            "event_teams": {k: v for k, v in data["event_teams"].items() if k in keys}}


def calibrate_beta(data, z_by_year, years):
    """P(red wins) = sigmoid(beta * (sum z_red - sum z_blue)), z = end-of-season standardised strength."""
    X, y = [], []
    for yr in years:
        z = z_by_year.get(yr)
        if not z:
            continue
        for e in data["events"]:
            if e["year"] != yr or e["type"] not in (0, 1):
                continue
            for m in data["matches"].get(e["key"], []):
                if m["lvl"] != "qm" or m["rs"] is None or m["rs"] == m["bs"]:
                    continue
                X.append(sum(z.get(t, 0) for t in m["red"]) - sum(z.get(t, 0) for t in m["blue"]))
                y.append(int(m["rs"] > m["bs"]))
    if len(X) < 200:
        return 1.0
    return float(LogisticRegression(fit_intercept=False, C=1e6).fit(np.array(X)[:, None], y).coef_[0][0])


def build_forecast(data, target, rules, nsims=400, log=print, use_schedules=True, seed=1):
    """Train on seasons < target only, forecast `target`, simulate its events. Rosters/schedules of the target
    season come from `data` (known before the season); its match RESULTS are never used."""
    train = cut(data, target)
    played = sorted({e["year"] for e in train["events"] if e["year"] not in forecast.SKIP_YEARS
                     and any(m["rs"] is not None for m in train["matches"].get(e["key"], []))})
    test_years = played[-3:]
    log(f"[{target}] trained through {played[-1] if played else None}; model test years {test_years}")
    sel = ratings.select_model(train, test_years, log=log)
    end_strength, counts = ratings.selected_strengths(sel), sel["counts"]
    z = forecast.zscores(end_strength, counts)
    pct = forecast.percentiles(z, counts)
    rookie = data.get("rookie", {})
    fb = forecast.walk_forward(z, [y for y in played if y >= played[0] + 4][-4:], rookie) if len(played) > 6 else None
    w = fb["blend"]["analog_weight"] if fb else 0.4
    fc = forecast.predict_next(z, target, blend_w=w, rookie=rookie)
    rk_prior = forecast.rookie_prior(z, rookie)
    beta = calibrate_beta(train, z, test_years)

    tevents = [e for e in data["events"] if e["year"] == target]
    tteams = {t for e in tevents for t in data["event_teams"].get(e["key"], [])}
    strengths = {}
    for t in tteams:
        if t in fc:
            strengths[t] = (fc[t]["mean"], fc[t]["sd"])
        elif rookie.get(t) == target:
            strengths[t] = (rk_prior["mean"], rk_prior["sd"])
        else:
            strengths[t] = (-0.3, 0.9)           # returning after a hiatus / unknown: wide prior
    n_prior = sum(1 for t in tteams if t not in fc)
    sched = {}
    if use_schedules:
        for e in tevents:
            qm = [m for m in data["matches"].get(e["key"], []) if m["lvl"] == "qm"]
            if qm:
                sched[e["key"]] = [(m["red"], m["blue"]) for m in qm]
    sim = simulate.Sim(rules, beta=beta, nsims=nsims, seed=seed)
    season = simulate.run_season(sim, target, tevents, data["event_teams"], strengths, rookie, sched, log)
    return {"target": target, "train": train, "played": played, "test_years": test_years, "sel": sel,
            "end_strength": end_strength, "counts": counts, "z": z, "pct": pct, "fb": fb, "blend_w": w, "fc": fc,
            "rookie_prior": rk_prior, "beta": beta, "tevents": tevents, "strengths": strengths, "n_prior": n_prior,
            "sched": sched, "season": season}


def model_label(sel):
    s = sel["selected"]
    if s["kind"] == "single":
        return s["name"]
    return "ensemble: " + " + ".join(f"{m['model']} ({m['weight']:.2f})" for m in s["members"])
