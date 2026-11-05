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

sys.path.insert(0, str(Path(__file__).parent))
from frc import forecast, pipeline, ratings, score  # noqa: E402

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
            if e["type"] in (3, 4):                    # Championship divisions / Einstein: rosters only (for scoring)
                try:
                    if not future:
                        data["event_teams"][e["key"]] = api.event_teams(e["key"])
                except RuntimeError:
                    raise
                except Exception as ex:  # noqa
                    log(f"  skipped {e['key']}: {type(ex).__name__}")
                continue
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
    for y in sorted({e["year"] for e in data["events"]}):      # rookie year for every team ever seen (true team age)
        try:
            team_info.update(api.teams_year(y))
        except Exception as ex:  # noqa
            log(f"teams_year {y} failed: {ex}")
    data["team_info"] = team_info
    data["rookie"] = {t: v.get("rookie_year") for t, v in team_info.items()}
    return data



def alias_map(rules):
    out = {}
    for k, d in rules["districts"].items():
        for a in d.get("aliases", [k]):
            out[a.lower()] = k
        out[k] = k
    return out


def make_scorecard(rep, data, main, rules):
    """Score a replay (pipeline run as if it were the preseason of rep['target']) against what actually happened."""
    year = rep["target"]
    nan = float("nan")

    def inseason(key):
        p = ratings.selected_match_p(main["sel"], key)
        return nan if p is None else p
    return {"year": year, "trained_through": rep["played"][-1], "rating_model": pipeline.model_label(rep["sel"]),
            "analog_weight": rep["blend_w"], "forecast_backtest": rep["fb"],
            "strength": score.score_strength(rep["fc"], main["z"].get(year, {}), main["counts"].get(year, {}), data["rookie"], year),
            "matches": score.score_matches(year, data, rep["strengths"], rep["beta"], inseason),
            "events": score.score_events(year, data, rep["season"]["events"]),
            "advancement": score.score_advancement(year, data, rep["season"], rules, alias_map(rules))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-year", type=int, default=2027)
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--nsims", type=int, default=400)
    ap.add_argument("--no-replay", action="store_true", help="skip the previous-season replay / scorecard")
    a = ap.parse_args()
    log = lambda *x: print(*x, flush=True)  # noqa
    warnings = []
    rules = json.loads((ROOT / "config" / "rules.json").read_text())

    if a.synthetic:
        from frc import synth
        data = synth.make_dataset(n_teams=900, years=range(2016, a.target_year))
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

    # ---- the forecast for the target season (trained ONLY on earlier seasons, even once target-year matches exist)
    M = pipeline.build_forecast(data, a.target_year, rules, a.nsims, log, use_schedules=True)
    sel, z, pct, counts, fc, season = M["sel"], M["z"], M["pct"], M["counts"], M["fc"], M["season"]
    end_strength, strengths, beta, fb, w = M["end_strength"], M["strengths"], M["beta"], M["fb"], M["blend_w"]
    selected, played_years, test_years = sel["selected"], M["played"], M["test_years"]
    model_label = pipeline.model_label(sel)
    tevents, sched, tteams = M["tevents"], M["sched"], set(strengths)
    log("selected rating model:", model_label, "| beta", beta)

    if season["unmatched_districts"]:
        warnings.append("District abbreviations not in config/rules.json (excluded from advancement odds): "
                        + ", ".join(season["unmatched_districts"]))
    if season.get("skipped_small_events"):
        warnings.append(f"{len(season['skipped_small_events'])} events with fewer than 24 teams were not simulated.")
    if not tevents:
        warnings.append(f"No {a.target_year} events on TBA yet.")
    if M["n_prior"]:
        warnings.append(f"{M['n_prior']} {a.target_year} teams had no rating history; using rookie/wide priors.")

    # ---- scorecard: replay the previous season with a strict cutoff, plus live scoring of the target season
    scorecard = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(), "replays": [], "live": None}
    last = played_years[-1] if played_years else None
    if not a.no_replay and last and len(played_years) >= 6:
        log(f"=== replay: forecasting {last} using only seasons before it ===")
        rep = pipeline.build_forecast(data, last, rules, a.nsims, log, use_schedules=False, seed=2)
        scorecard["replays"].append(make_scorecard(rep, data, M, rules))
    live_ev = score.score_events(a.target_year, data, season["events"])
    live_m = score.score_matches(a.target_year, data, strengths, beta)
    if live_ev["events"] or live_m:
        scorecard["live"] = {"year": a.target_year, "events": live_ev, "matches": live_m}

    # ---- match predictions: target-year scheduled matches + recent past matches (pre-match model probabilities)
    mp = {}
    for e in tevents:
        ms = data["matches"].get(e["key"], [])
        if ms:
            mp[e["key"]] = pipeline.simulate.predict_matches(ms, strengths, beta)
    recent_year = played_years[-1] if played_years else None
    for e in data["events"]:
        if e["year"] == recent_year and e["type"] in (0, 1, 99):
            if e["start"] >= (dt.date.today() - dt.timedelta(days=60)).isoformat() or a.synthetic:
                mp.setdefault(e["key"], [
                    {"key": m["key"], "lvl": m["lvl"], "num": m["num"], "set": m["set"], "red": m["red"], "blue": m["blue"],
                     "rs": m["rs"], "bs": m["bs"], "p_red": round(ratings.selected_match_p(sel, m["key"]) or 0.5, 4), "pre_match": True}
                    for m in data["matches"].get(e["key"], [])])

    # ---- team table
    info = data.get("team_info", {})
    teams_out = []
    cur_year = played_years[-1] if played_years else None
    for t in sorted(set(fc) | tteams):
        f = fc.get(t)
        s = season["teams"].get(t, {})
        hist = [[y, round(z[y][t], 3), round(pct[y][t], 1), counts.get(y, {}).get(t, 0),
                 (round(float(end_strength[y][t]), 4) if selected["kind"] == "single" else None)]
                for y in sorted(z) if t in z[y]]
        rnd = lambda v: round(v, 3) if isinstance(v, float) else v  # noqa
        teams_out.append({
            "team": t, "name": (info.get(t) or {}).get("name"), "rookie_year": data["rookie"].get(t), "hist": hist,
            "z_now": round(z[cur_year].get(t), 3) if cur_year and t in z.get(cur_year, {}) else None,
            "forecast": ({k: rnd(v) for k, v in f.items()} if f else
                         {"mean": strengths[t][0], "sd": strengths[t][1], "shape_flags": ["new / no history"], "outlook_flags": [],
                          "flags": ["new / no history"], "history": [], "analogs": []}),
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
            "synthetic": a.synthetic, "best_rating_model": model_label, "selected_model": selected,
            "rating_leaderboard": sel["board"], "test_years": test_years, "forecast_backtest": fb,
            "blend_analog_weight": w, "beta": beta, "nsims": a.nsims, "warnings": warnings,
            "districts": season["districts"], "rules_note": rules["_comment"],
            "normalization": "Match scores are divided by the season's mean alliance score before any model sees them, and every season's team ratings are standardised (mean 0, sd 1) over teams with at least 8 matches that season, so years with different games are comparable.",
            "not_included": [
                {"model": "COPR (component OPR)", "why": "Needs per-year score_breakdown fields (game-specific). When components sum to the total score, summed component OPRs equal ordinary OPR, so it adds no predictive power for win/score prediction."},
                {"model": "lOPR (logistic bonus-RP model)", "why": "Needs ranking-point bonus flags from score_breakdown, which are game-specific; the logistic win model (bt) is the nearest equivalent."},
                {"model": "ACE", "why": "No published specification found; its description (blend of Elo-type and OPR-type models) is covered by the stacked ensemble, whose weights are learned from data."}],
            "game_note": "Qualification bonus-RP model is a placeholder until the 2027 game is released (Jan 9 2027).",
            "in_season_note": "Forecasts are trained on seasons before the target year only; they do not yet update as target-year matches are played."}
    dump("meta.json", meta)
    dump("teams.json", teams_out)
    dump("events.json", ev_list)
    dump("matches.json", mp)
    dump("scorecard.json", scorecard)
    log(f"wrote {len(teams_out)} teams, {len(ev_list)} events, {len(mp)} match sets, {len(scorecard['replays'])} replay(s) -> {OUT}")


if __name__ == "__main__":
    main()
