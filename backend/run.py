"""Pipeline: fetch -> rate -> backtest -> forecast -> simulate -> write static JSON for the site.

  python run.py --target-year 2027            # real data, needs TBA_KEY
  python run.py --synthetic                   # demo data, no network
"""
import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from frc import forecast, ratings, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data"


def jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, default=jdefault, separators=(",", ":")))


def load_real(start, target, log):
    from frc.tba import TBA
    api = TBA()
    today = dt.date.today().isoformat()
    data = {"events": [], "matches": {}, "event_teams": {}}
    team_info = {}
    for y in range(start, target + 1):
        if y in forecast.SKIP_YEARS:
            continue
        evs = api.events(y)
        log(f"{y}: {len(evs)} events")
        for e in evs:
            data["events"].append(e)
            future = e["start"] > today
            if e["type"] not in (0, 1, 2, 5, 99):
                continue
            try:
                if not future:
                    ms = api.event_matches(e["key"])
                    data["matches"][e["key"]] = ms
                    data["event_teams"][e["key"]] = sorted({t for m in ms for t in m["red"] + m["blue"]}) or api.event_teams(e["key"])
                else:
                    # future event: roster + published schedule, if any
                    data["event_teams"][e["key"]] = api.event_teams(e["key"])
                    data["matches"][e["key"]] = api.event_matches(e["key"])
            except RuntimeError:
                raise                                   # bad API key etc: stop loudly
            except Exception as ex:                     # one flaky event shouldn't kill the whole refresh
                log(f"  skipped {e['key']}: {type(ex).__name__}")
                data["matches"].pop(e["key"], None)
                data["event_teams"].pop(e["key"], None)
    for y in (target, target - 1):
        try:
            team_info.update(api.teams_year(y))
        except Exception as ex:  # noqa
            log(f"teams_year {y} failed: {ex}")
    data["team_info"] = team_info
    data["rookie"] = {t: v.get("rookie_year") for t, v in team_info.items()}
    return data


def calibrate_beta(data, model_cls, params, z_by_year, years):
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
    lr = LogisticRegression(fit_intercept=False, C=1e6).fit(np.array(X)[:, None], y)
    return float(lr.coef_[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-year", type=int, default=2027)
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--nsims", type=int, default=400)
    a = ap.parse_args()
    log = lambda *x: print(*x, flush=True)  # noqa
    warnings = []
    rules = json.loads((ROOT / "config" / "rules.json").read_text())

    if a.synthetic:
        from frc import synth
        data = synth.make_dataset(n_teams=900, years=range(2016, a.target_year))
        # fabricate a next-year district event roster from the synthetic population
        rng = np.random.default_rng(5)
        pop = [t for t in range(1, 901)]
        for e in range(18):
            k = f"{a.target_year}syn{e:02d}"
            data["events"].append({"key": k, "year": a.target_year, "name": f"Synthetic District Event {e+1}", "type": 1,
                                   "type_name": "district", "district": "syn", "start": f"{a.target_year}-03-{1+e:02d}",
                                   "end": f"{a.target_year}-03-{3+e:02d}", "week": 1 + e // 4, "country": "USA"})
            data["event_teams"][k] = [int(t) for t in rng.choice(pop, 40, replace=False)]
            data["matches"][k] = []
        rules["districts"]["syn"] = {"name": "Synthetic district", "cmp_slots": 60, "dcmp_capacity": 120,
                                     "impact": 2, "ei": 1, "ras": 1, "aliases": ["syn"]}
        data["team_info"] = {t: {"name": f"Synthetic {t}", "rookie_year": data["rookie"][t]} for t in pop}
        warnings.append("SYNTHETIC DEMO DATA - not real FRC results.")
    else:
        data = load_real(a.start_year, a.target_year, log)

    played_years = sorted({e["year"] for e in data["events"]
                           if e["year"] < a.target_year and e["year"] not in forecast.SKIP_YEARS
                           and any(m["rs"] is not None for m in data["matches"].get(e["key"], []))})
    test_years = played_years[-3:]
    log("played years", played_years, "test", test_years)

    # 1) pick the best ranking model
    board, best_name, best_params = ratings.backtest(data, test_years)
    log("leaderboard:", json.dumps(board, default=jdefault)[:400])
    model = ratings.MODELS[best_name](**best_params)
    live = {}

    def hook(y, e, m, p):
        if y == played_years[-1]:
            live[m["key"]] = p
    _, end_strength = ratings.run(model, data, on_match=hook)

    # 2) standardise, forecast backtest, forecast
    z = forecast.zscores(end_strength)
    fb = forecast.walk_forward(z, [y for y in played_years if y >= played_years[0] + 4][-4:]) if len(played_years) > 6 else None
    w = fb["blend"]["analog_weight"] if fb else 0.4
    fc = forecast.predict_next(z, a.target_year, blend_w=w)
    rk_prior = forecast.rookie_prior(z, data["rookie"])
    log("forecast backtest:", json.dumps(fb, default=jdefault)[:300] if fb else "n/a")

    # 3) match model calibration
    beta = calibrate_beta(data, model, best_params, z, test_years)
    log("beta", beta)

    # 4) target-year strengths
    strengths = {}
    tteams = {t for e in data["events"] if e["year"] == a.target_year for t in data["event_teams"].get(e["key"], [])}
    for t in tteams:
        if t in fc:
            strengths[t] = (fc[t]["mean"], fc[t]["sd"])
        elif data["rookie"].get(t) == a.target_year:
            strengths[t] = (rk_prior["mean"], rk_prior["sd"])
        else:
            strengths[t] = (-0.3, 0.9)   # returning after hiatus / not found: wide prior
    n_prior = sum(1 for t in tteams if t not in fc)

    # 5) simulate target season
    tevents = [e for e in data["events"] if e["year"] == a.target_year]
    sched = {}
    for e in tevents:
        qm = [m for m in data["matches"].get(e["key"], []) if m["lvl"] == "qm"]
        if qm:
            sched[e["key"]] = [(m["red"], m["blue"]) for m in qm]
    sim = simulate.Sim(rules, beta=beta, nsims=a.nsims)
    season = simulate.run_season(sim, a.target_year, tevents, data["event_teams"], strengths, data["rookie"], sched, log)
    if season["unmatched_districts"]:
        warnings.append("District abbreviations not in config/rules.json (excluded from advancement odds): "
                        + ", ".join(season["unmatched_districts"]))
    if not tevents:
        warnings.append(f"No {a.target_year} events on TBA yet.")
    if n_prior:
        warnings.append(f"{n_prior} {a.target_year} teams had no rating history; using rookie/wide priors.")

    # 6) match predictions: target-year scheduled matches + recent past matches
    mp = {}
    for e in tevents:
        ms = [m for m in data["matches"].get(e["key"], [])]
        if ms:
            mp[e["key"]] = simulate.predict_matches(ms, strengths, beta)
    recent_year = played_years[-1] if played_years else None
    for e in data["events"]:
        if e["year"] == recent_year and e["type"] in (0, 1, 99):
            if e["start"] >= (dt.date.today() - dt.timedelta(days=60)).isoformat() or a.synthetic:
                mp.setdefault(e["key"], [
                    {"key": m["key"], "lvl": m["lvl"], "num": m["num"], "set": m["set"], "red": m["red"], "blue": m["blue"],
                     "rs": m["rs"], "bs": m["bs"], "p_red": round(live.get(m["key"], 0.5), 4), "pre_match": True}
                    for m in data["matches"].get(e["key"], [])])

    # 7) team table
    info = data.get("team_info", {})
    teams_out = []
    cur_year = played_years[-1] if played_years else None
    for t in sorted(set(fc) | tteams):
        f = fc.get(t)
        s = season["teams"].get(t, {})
        teams_out.append({
            "team": t, "name": (info.get(t) or {}).get("name"), "rookie_year": data["rookie"].get(t),
            "z_now": round(z[cur_year].get(t), 3) if cur_year and t in z.get(cur_year, {}) else None,
            "forecast": ({k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()} if f else
                         {"mean": strengths[t][0], "sd": strengths[t][1], "flags": ["no history"], "history": [], "analogs": []}),
            "adv": s})
    teams_out.sort(key=lambda r: -r["forecast"]["mean"])
    for i, r in enumerate(teams_out):
        r["rank"] = i + 1

    ev_list = [{"key": e["key"], "name": e["name"], "type": e["type_name"], "district": e.get("district"),
                "start": e["start"], "end": e["end"], "week": e.get("week"), "year": e["year"],
                "country": e.get("country"), "n_teams": len(data["event_teams"].get(e["key"], [])),
                "has_schedule": e["key"] in sched, "sim": season["events"].get(e["key"])}
               for e in data["events"] if e["year"] >= (cur_year or a.target_year)]
    meta = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(), "target_year": a.target_year,
            "synthetic": a.synthetic, "best_rating_model": best_name, "best_params": best_params,
            "rating_leaderboard": board, "test_years": test_years, "forecast_backtest": fb,
            "blend_analog_weight": w, "beta": beta, "nsims": a.nsims, "warnings": warnings,
            "districts": season["districts"], "rules_note": rules["_comment"],
            "game_note": "Qualification bonus-RP model is a placeholder until the 2027 game is released (Jan 9 2027)."}
    dump("meta.json", meta)
    dump("teams.json", teams_out)
    dump("events.json", ev_list)
    dump("matches.json", mp)
    log(f"wrote {len(teams_out)} teams, {len(ev_list)} events, {len(mp)} match sets -> {OUT}")


if __name__ == "__main__":
    main()
