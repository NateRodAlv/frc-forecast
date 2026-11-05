"""Rules + model sanity tests. Run: cd frc-forecast && python -m pytest -q tests"""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from frc import forecast, ratings, simulate, synth

ROOT = Path(__file__).resolve().parents[1]


def test_qual_points_match_manual_sample():
    # Manual 11.1.1 sample, 40-team event: rank1=22, 2=21, 3=20, 4=19, 19=13, 20=13, 21=12, 37=6, 38=6, 39=5, 40=4
    exp = {1: 22, 2: 21, 3: 20, 4: 19, 19: 13, 20: 13, 21: 12, 37: 6, 38: 6, 39: 5, 40: 4}
    for r, v in exp.items():
        assert simulate.qual_points(np.array([r]), 40)[0] == v, r


def test_age_points():
    assert simulate.age_points(2027, 2027) == 10
    assert simulate.age_points(2027, 2026) == 5
    assert simulate.age_points(2027, 2020) == 0


def test_bracket_is_valid_double_elim_and_favours_strong():
    rng = np.random.default_rng(0)
    SA = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5]
    wins = np.zeros(8)
    for _ in range(3000):
        w, f, th, fo, ww, wf = simulate.play_bracket(SA, 1.0, rng.random(30))
        assert len({w, f, th, fo}) == 4 and ww == 2 and wf in (0, 1)
        wins[w] += 1
    assert wins[0] == wins.max() and wins.sum() == 3000


def test_each_event_places_exactly_three_winners_and_eight_captains():
    rules = json.loads((ROOT / "config/rules.json").read_text())
    sim = simulate.Sim(rules, beta=1.0, nsims=60)
    n = 40
    S = np.random.default_rng(1).normal(size=(n, 60))
    r = sim.event(list(range(n)), S.mean(1), S, {}, 2027, rules["district_points"]["awards"])
    assert np.all(r["won"].sum(0) == 3)
    assert np.all((r["rank"] <= 8).sum(0) == 8)
    assert r["qual"].max() == 22 and r["qual"].min() >= 4


def test_district_slots_conserved():
    rules = json.loads((ROOT / "config/rules.json").read_text())
    rules["districts"]["syn"] = {"name": "s", "cmp_slots": 20, "dcmp_capacity": 50, "impact": 2, "ei": 1, "ras": 1, "aliases": ["syn"]}
    d = synth.make_dataset(n_teams=400, years=range(2024, 2027), seed=2)
    ev = [e for e in d["events"] if e["year"] == 2026]
    st = {t: (0.0, 0.8) for k, v in d["event_teams"].items() if k.startswith("2026") for t in v}
    out = simulate.run_season(simulate.Sim(rules, 1.0, nsims=60), 2026, ev, {k: v for k, v in d["event_teams"].items() if k.startswith("2026")}, st, d["rookie"], log=lambda *_: None)
    assert abs(out["districts"]["syn"]["expected_cmp_teams"] - 20) < 1.0
    assert out["districts"]["syn"]["expected_dcmp_teams"] <= 50.0001


def test_ratings_beat_coinflip_and_no_lookahead():
    d = synth.make_dataset(n_teams=400, years=range(2018, 2027), seed=4)
    board, name, _ = ratings.backtest(d, [2025, 2026])
    assert board[0]["test_logloss"] < math.log(2) - 0.02
    # shuffle match results -> accuracy must collapse to ~50% (a leak would keep it high)
    rng = np.random.default_rng(0)
    for ms in d["matches"].values():
        sc = [(m["rs"], m["bs"]) for m in ms]; rng.shuffle(sc)
        for m, (a, b) in zip(ms, sc): m["rs"], m["bs"] = a, b
    board2, _, _ = ratings.backtest(d, [2025, 2026])
    assert board2[0]["test_acc"] < 0.55


def test_forecaster_beats_persistence_and_finds_planted_rebound():
    d = synth.make_dataset(n_teams=900, years=range(2012, 2027), seed=3)
    _, es = ratings.run(ratings.EPA(carry=0.7, lr0=0.6), d)
    z = forecast.zscores(es)
    fb = forecast.walk_forward(z, [2023, 2024, 2025, 2026])
    assert fb["ridge"]["rmse"] < fb["persist"]["rmse"]
    assert 0.6 < fb["analog_interval_coverage_10_90"] < 0.95


def test_tba_parsing_with_schema_shaped_fixtures(monkeypatch, tmp_path):
    from frc import tba as tbamod
    monkeypatch.setattr(tbamod, "CACHE", tmp_path)
    api = tbamod.TBA(key="x")
    fake = {
        "/events/2027": [{"key": "2027chcmp", "year": 2027, "name": "Dist Event", "event_type": 1, "district": {"abbreviation": "chs"},
                          "start_date": "2027-03-05", "end_date": "2027-03-07", "week": 1, "country": "USA", "state_prov": "VA"}],
        "/event/2027chcmp/teams/keys": ["frc1", "frc2"],
        "/event/2027chcmp/matches/simple": [{"key": "2027chcmp_qm1", "comp_level": "qm", "set_number": 1, "match_number": 1, "time": 1, "predicted_time": 2,
            "alliances": {"red": {"team_keys": ["frc1", "frc2", "frc3"], "score": -1}, "blue": {"team_keys": ["frc4", "frc5", "frc6"], "score": 70}}}],
        "/teams/2027/0": [{"team_number": 1, "nickname": "A", "rookie_year": 2000, "state_prov": "VA", "country": "USA"}],
    }
    monkeypatch.setattr(api, "get", lambda p, retries=3: fake.get(p))
    ev = api.events(2027)[0]
    assert ev["district"] == "chs" and ev["type_name"] == "district" and ev["start"] == "2027-03-05"
    assert api.event_teams("2027chcmp") == [1, 2]
    m = api.event_matches("2027chcmp")[0]
    assert m["rs"] is None and m["bs"] == 70 and m["red"] == [1, 2, 3]
    assert api.teams_year(2027)[1]["rookie_year"] == 2000


def test_tba_suffixed_offseason_team_keys(monkeypatch, tmp_path):
    from frc import tba as tbamod
    monkeypatch.setattr(tbamod, "CACHE", tmp_path)
    api = tbamod.TBA(key="x")
    ok = {"team_keys": ["frc1", "frc2", "frc3"], "score": 10}
    bad = {"team_keys": ["frc4B", "frc5", "frc6"], "score": 20}
    fake = {"/event/e/matches/simple": [
        {"key": "e_qm1", "comp_level": "qm", "set_number": 1, "match_number": 1, "alliances": {"red": ok, "blue": bad}},
        {"key": "e_qm2", "comp_level": "qm", "set_number": 1, "match_number": 2, "alliances": {"red": ok, "blue": {"team_keys": ["frc7", "frc8", "frc9"], "score": 5}}}],
        "/event/e/teams/keys": ["frc1", "frc254B", "frc2"]}
    monkeypatch.setattr(api, "get", lambda p, retries=3: fake.get(p))
    assert [m["key"] for m in api.event_matches("e")] == ["e_qm2"]
    assert api.event_teams("e") == [1, 2]


def test_tba_retries_dropped_connections(monkeypatch, tmp_path):
    import requests
    from frc import tba as tbamod
    monkeypatch.setattr(tbamod, "CACHE", tmp_path)
    monkeypatch.setattr(tbamod.time, "sleep", lambda s: None)
    api = tbamod.TBA(key="x")
    calls = {"n": 0}

    class R:
        status_code, headers, text = 200, {}, "[1]"
        def json(self): return [1]

    def flaky(url, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("Remote end closed connection")
        return R()
    monkeypatch.setattr(api.s, "get", flaky)
    assert api.get("/x") == [1] and calls["n"] == 3


def test_schedule_all_event_sizes():
    rules = json.loads((ROOT / "config/rules.json").read_text())
    sim = simulate.Sim(rules, beta=1.0, nsims=10, seed=3)
    for n in list(range(6, 110)) + [150]:
        sch = sim.schedule(n)
        assert all(len(set(r + b)) == 6 for r, b in sch)
        counts = np.bincount([t for r, b in sch for t in r + b], minlength=n)
        assert counts.min() >= 1 and counts.max() <= sim.mpt + 2


def test_small_events_are_skipped_not_crashed():
    rules = json.loads((ROOT / "config/rules.json").read_text())
    rules["districts"]["syn"] = {"name": "s", "cmp_slots": 5, "dcmp_capacity": 30, "impact": 1, "ei": 1, "ras": 1, "aliases": ["syn"]}
    evs = [{"key": f"e{i}", "year": 2027, "name": "x", "type": 1, "type_name": "district", "district": "syn",
            "start": f"2027-03-{i+1:02d}", "country": "USA"} for i in range(3)]
    teams = {"e0": list(range(1, 21)), "e1": list(range(10, 60)), "e2": list(range(100, 117))}   # 20, 50, 17 teams
    st = {t: (0.0, 0.8) for v in teams.values() for t in v}
    out = simulate.run_season(simulate.Sim(rules, 1.0, nsims=30), 2027, evs, teams, st, {}, log=lambda *_: None)
    assert set(out["events"]) == {"e1"} and sorted(out["skipped_small_events"]) == ["e0", "e2"]


def test_season_normalisation_uses_teams_with_enough_matches():
    rng = np.random.default_rng(0)
    d = {2024: {t: float(rng.normal(5, 2)) for t in range(300)}}
    counts = {2024: {t: (20 if t < 200 else 2) for t in range(300)}}
    for t in range(200, 300):
        d[2024][t] = 5.0                      # thinly observed teams sit on the prior
    z = forecast.zscores(d, counts)[2024]
    ref = np.array([z[t] for t in range(200)])
    assert abs(ref.mean()) < 1e-9 and abs(ref.std() - 1) < 1e-9
    pc = forecast.percentiles(forecast.zscores(d, counts), counts)[2024]
    assert all(0 <= v <= 100 for v in pc.values())
    assert pc[int(np.argmax([z[t] for t in range(200)]))] > 99


def test_every_candidate_model_runs_and_ensemble_only_when_better():
    d = synth.make_dataset(n_teams=500, years=range(2018, 2027), seed=9)
    sel = ratings.select_model(d, [2025, 2026])
    names = {r["model"] for r in sel["board"] if not r["model"].startswith("ensemble")}
    assert names == set(ratings.GRID)
    singles = [r for r in sel["board"] if not r["model"].startswith("ensemble")]
    ens = [r for r in sel["board"] if r["model"].startswith("ensemble")]
    if sel["selected"]["kind"] == "ensemble":
        assert ens and ens[0]["test_logloss"] < min(r["test_logloss"] for r in singles)
    else:
        assert not ens or ens[0]["test_logloss"] >= min(r["test_logloss"] for r in singles) - 5e-4
    es = ratings.selected_strengths(sel)
    assert 2026 in es and len(es[2026]) > 100
    assert sum(r["selected"] for r in sel["board"]) == 1


# ------------------------------------------------------------------ replay / age / scoring
def _rules_syn():
    rules = json.loads((ROOT / "config/rules.json").read_text())
    rules["districts"]["syn"] = {"name": "s", "cmp_slots": 20, "dcmp_capacity": 60, "impact": 1, "ei": 1, "ras": 1, "aliases": ["syn"]}
    return rules


def test_cutoff_blocks_target_season_results():
    """Forecast for season T must not change if T's match RESULTS are deleted or scrambled."""
    from frc import pipeline
    d = synth.make_dataset(n_teams=450, years=range(2017, 2027), seed=21)
    rules = _rules_syn()
    for e in d["events"]:
        if e["year"] == 2026:
            e["district"] = "syn"
    a = pipeline.build_forecast(d, 2026, rules, nsims=30, log=lambda *_: None, use_schedules=False)
    rng = np.random.default_rng(0)
    for e in d["events"]:
        if e["year"] == 2026:
            for m in d["matches"][e["key"]]:
                m["rs"], m["bs"] = int(rng.integers(0, 200)), int(rng.integers(0, 200))
    b = pipeline.build_forecast(d, 2026, rules, nsims=30, log=lambda *_: None, use_schedules=False)
    assert a["sel"]["selected"] == b["sel"]["selected"] and a["beta"] == b["beta"]
    assert all(abs(a["fc"][t]["mean"] - b["fc"][t]["mean"]) < 1e-12 for t in a["fc"])
    assert a["season"]["teams"] == b["season"]["teams"]
    assert max(a["z"]) == 2025                     # 2026 never reaches the rating history


def test_age_matched_analogs_use_same_age_and_history():
    d = synth.make_dataset(n_teams=700, years=range(2012, 2027), seed=5)
    _, es = ratings.run(ratings.EPA(carry=0.7, lr0=0.6), d)
    z = forecast.zscores(es)
    h = forecast.histories(z)
    X, y, meta = forecast.build_pairs(h, d["rookie"], min(z))
    an = forecast.Analog(k=40, min_pool=50).fit(X, y, meta)
    q = X[0]
    for band, lc in (("2", 2), ("1", 1), ("13+", 3)):
        idx, w, dist, level, n = an.neighbours(q, band, lc)
        if level == 0:
            assert all(meta[i][5] == band and meta[i][6] == lc for i in idx)
    # with 1 season of history only the rating level is compared: changing momentum features must not change neighbours
    q2 = q.copy(); q2[1] += 3.0; q2[3] += 2.0
    i1 = an.neighbours(q, "13+", 1)[0]; i2 = an.neighbours(q2, "13+", 1)[0]
    assert list(i1) == list(i2)


def test_tags_are_consistent():
    d = synth.make_dataset(n_teams=700, years=range(2012, 2027), seed=6)
    _, es = ratings.run(ratings.EPA(carry=0.7, lr0=0.6), d)
    z = forecast.zscores(es)
    for w in (0.0, 0.5):
        fc = forecast.predict_next(z, 2027, blend_w=w, rookie=d["rookie"])
        for v in fc.values():
            s, o = " ".join(v["shape_flags"]), " ".join(v["outlook_flags"])
            assert not ("big jump" in s and "big drop" in s)
            assert not ("did better" in o and "did worse" in o)
            assert not ("improve" in o and "regress" in o)
            if w == 0.0:
                assert "similar teams" not in o       # analog tags hidden while analogs have no weight


def test_scoring_helpers():
    from frc import score
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.6, 3000); o = (rng.random(3000) < p).astype(float)
    m = score.prob_metrics(p, o)
    assert m["skill"] > 0.05 and abs(m["expected"] - m["actual"]) < 0.06 * m["expected"]
    assert all(abs(b["pred"] - b["actual"]) < 0.12 for b in m["bins"] if b["n"] > 100)     # calibrated by construction
    assert score.prob_metrics([0.5] * 5, [1] * 5) is None                                 # too few cases
    mk = lambda lvl, n, r, b: {"lvl": lvl, "set": 1, "num": n, "red": [1, 2, 3], "blue": [4, 5, 6], "rs": r, "bs": b}
    assert score.finals_outcome([mk("f", 1, 10, 5), mk("f", 2, 4, 9), mk("f", 3, 8, 2)]) == ({1, 2, 3}, {4, 5, 6})
    assert score.finals_outcome([mk("f", 1, 10, 5)]) is None                              # undecided
    ev = {"year": 2026, "key": "e", "type": 1}
    data = {"events": [ev], "matches": {"e": [mk("f", 1, 10, 5), mk("f", 2, 9, 3)]}}
    sim = {"e": {"teams": {str(t): {"p_win": 0.5 if t <= 3 else 0.0, "p_finalist": 1.0 if t <= 6 else 0.0} for t in range(1, 25)}}}
    r = score.score_events(2026, data, sim)
    assert r["events"] == 1 and r["win"]["actual"] == 3


def test_advancement_not_scored_without_rosters():
    from frc import score
    rules = _rules_syn()
    season = {"teams": {t: {"model": "district", "district": "syn", "p_dcmp": 0.5, "p_cmp": 0.1} for t in range(60)}}
    data = {"events": [], "event_teams": {}, "matches": {}}
    r = score.score_advancement(2026, data, season, rules, {"syn": "syn"})
    assert r["dcmp"] is None and r["cmp_district"] is None and r["slots"][0]["cmp_actual"] is None
