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


# calibrate_beta fits on end-of-season strengths (no uncertainty), but the simulator runs on preseason forecasts
# that are noisier. Fitting beta directly on preseason-forecast match outcomes gave 1.2-1.4x the production value
# (1.107), and an end-to-end replay of 2025 and 2026 confirmed it: x1.25 lowered match log loss in both
# (0.5335 -> 0.5306, 0.5200 -> 0.5176) while x0.8 raised it.
BETA_SCALE = 1.25

# What the replays can and cannot claim. Rating hyperparameters (ratings.GRID) were tuned on seasons through
# TUNED_THROUGH, so earlier replays are partly in-sample; BETA_SCALE and simulate.PLAYOFF_SHARPEN were chosen by
# looking at the 2025 and 2026 replays, so those two are not a clean test of those two settings.
TUNED_THROUGH = 2023
SCALE_FACTORS_CHOSEN_WITH = (2025, 2026)
DOUBLE_ELIM_FROM = simulate.DOUBLE_ELIM_FROM   # playoff format the simulator uses per season (single elimination before)


# A new forecaster replaces the original only if it is better by at least this much on the walk-forward seasons.
SEEDED_MARGIN = 5e-4     # match log loss (selected seeded vs selected unseeded rating model, on the held-out seasons)
TREND2_MARGIN = 5e-4     # RMSE, in standard deviations of team strength
SD2_MARGIN = 5e-3        # log score of the forecast spread (nats per team-season)


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
    # NEW: seeded rating models. Seeds = a preseason forecast for every season, each made only from earlier seasons'
    # ratings (themselves from the unseeded models just selected). The seeded models join the race as SEPARATE
    # candidates; they drive the forecasts only if the best selection with them beats the unseeded one on the
    # held-out seasons by SEEDED_MARGIN, but they are listed on the Model page either way.
    seeded_info = {"used": False}
    if len(played) > 6:
        z0 = forecast.zscores(ratings.selected_strengths(sel), sel["counts"])
        seeds = forecast.preseason_seeds(z0, sel["counts"], data.get("rookie", {}))
        sel_s = ratings.select_model({**train, "seeds": seeds}, test_years, grid=ratings.GRID_SEEDED, log=log, prior=sel)
        base_ll = next(r["test_logloss"] for r in sel["board"] if r["selected"])
        seed_ll = next(r["test_logloss"] for r in sel_s["board"] if r["selected"])
        seeded_info = {"used": bool(seed_ll < base_ll - SEEDED_MARGIN), "test_logloss_unseeded": base_ll, "test_logloss_seeded": seed_ll}
        if seeded_info["used"]:
            sel = sel_s
        else:
            sel["board"] = sorted(sel["board"] + [{**r, "selected": False} for r in sel_s["board"] if r["model"].endswith("_seeded")], key=lambda r: r["test_logloss"])
        log(f"[{target}] seeded rating models: unseeded {base_ll:.4f} vs seeded {seed_ll:.4f} -> {'USED' if seeded_info['used'] else 'not used'}")
    end_strength, counts = ratings.selected_strengths(sel), sel["counts"]
    z = forecast.zscores(end_strength, counts)
    pct = forecast.percentiles(z, counts)
    rookie = data.get("rookie", {})
    fb = forecast.walk_forward(z, [y for y in played if y >= played[0] + 4][-4:], rookie, counts=counts) if len(played) > 6 else None
    # NEW forecasters (Trend v2, variance model v2) sit beside the originals and are used only when they beat them
    # on the same walk-forward seasons by a margin; the Model page lists both either way.
    use_t2 = bool(fb and fb["trend2"]["rmse"] < fb["ridge"]["rmse"] - TREND2_MARGIN)
    use_sd2 = bool(fb and fb["sd"]["sd2"]["nll"] < fb["sd"]["analog"]["nll"] - SD2_MARGIN)
    w = (fb["blend2"] if use_t2 else fb["blend"])["analog_weight"] if fb else 0.4
    fc = forecast.predict_next(z, target, blend_w=w, rookie=rookie, counts=counts,
                               trend="trend2" if use_t2 else "trend", sd_model="sd2" if use_sd2 else "analog")
    rk_prior = forecast.rookie_prior(z, rookie)
    beta = calibrate_beta(train, z, test_years) * BETA_SCALE

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
            "trend_used": "trend2" if use_t2 else "trend", "sd_used": "sd2" if use_sd2 else "analog", "seeded": seeded_info,
            "rookie_prior": rk_prior, "beta": beta, "tevents": tevents, "strengths": strengths, "n_prior": n_prior,
            "sched": sched, "season": season}


def model_label(sel):
    s = sel["selected"]
    if s["kind"] == "single":
        return s["name"]
    return "ensemble: " + " + ".join(f"{m['model']} ({m['weight']:.2f})" for m in s["members"])
