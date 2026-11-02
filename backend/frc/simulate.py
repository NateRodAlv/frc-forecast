"""Monte Carlo season simulator.

Per simulation we draw ONE true strength per team (from its forecast distribution) and reuse
it at every event, so district-point totals are properly correlated across a team's events.

Event: qualification schedule (real if published, else random balanced) -> RP ranking ->
       alliance selection (snake) -> 8-alliance double-elimination bracket (matches 1-13 + best-of-3 finals,
       bracket confirmed against FIRST's 2026 playoff bracket PDF) -> district/regional points.
District: first two home events + age -> DCMP qualification (capacity, Impact winners) ->
          DCMP sim (points x3) -> FIRST Championship slots (merit first, then points order).
Regional: points, top 3 (US) / 4 (intl) per event advance directly, remaining slots from the
          Regional Pool ranked by first-two-event points.

Game-specific unknowns (bonus RPs, scoring variance) are placeholders until the 2027 game is revealed;
they only influence how spread-out qualification rankings are, not the strength ordering.
"""
import math
from collections import defaultdict

import numpy as np
from scipy.special import erfinv

QUAL_ALPHA = 1.07
BONUS_A, BONUS_B, N_BONUS = -0.3, 0.5, 2       # placeholder bonus-RP model (game specific)
SIG = lambda x: 1.0 / (1.0 + np.exp(-x))


def qual_points(rank, n, alpha=QUAL_ALPHA):
    """Section 11.1.1: ceil(erfinv((N-2R+2)/(aN)) * 10/erfinv(1/a) + 12); 22 for rank 1, ~4 for last."""
    x = (n - 2.0 * rank + 2.0) / (alpha * n)
    return np.ceil(erfinv(np.clip(x, -0.999999, 0.999999)) * (10.0 / erfinv(1.0 / alpha)) + 12.0)


def age_points(year, rookie_year):
    if rookie_year is None:
        return 0
    return 10 if rookie_year == year else (5 if rookie_year == year - 1 else 0)


def play_bracket(SA, beta, u):
    """SA: strengths of 8 alliances (seed order). u: >=20 uniforms.
    Returns (winner, finalist, third, fourth, finals_wins_winner, finals_wins_finalist)."""
    k = [0]

    def play(a, b):
        p = 1.0 / (1.0 + math.exp(-beta * (SA[a] - SA[b])))
        r = u[k[0]]
        k[0] += 1
        return (a, b) if r < p else (b, a)

    w1, l1 = play(0, 7); w2, l2 = play(3, 4); w3, l3 = play(1, 6); w4, l4 = play(2, 5)
    w5, _ = play(l1, l2); w6, _ = play(l3, l4)
    w7, l7 = play(w1, w2); w8, l8 = play(w3, w4)
    w9, _ = play(l7, w6); w10, _ = play(l8, w5)
    w11, l11 = play(w7, w8); w12, l12 = play(w9, w10)
    w13, l13 = play(l11, w12)
    a, b, wa, wb = w11, w13, 0, 0
    while wa < 2 and wb < 2:
        w, _ = play(a, b)
        if w == a:
            wa += 1
        else:
            wb += 1
    if wa == 2:
        return a, b, l13, l12, wa, wb
    return b, a, l13, l12, wb, wa


def alliances_and_playoffs(order, S, beta, pick_noise, u, award_pts):
    """order: team indices by qualification rank. S: strengths (indexable by team idx).
    Returns per-team dict: as_pts, po_pts, and winner/finalist alliance members (team idx lists)."""
    caps = list(order[:8])
    capset = set(caps)
    avail = [int(i) for i in np.argsort(-(S[: len(S)] + pick_noise)) if int(i) not in capset]
    A = [[int(c)] for c in caps]
    pi, pickno = 0, {}
    for i in range(8):
        A[i].append(avail[pi]); pickno[avail[pi]] = i + 1; pi += 1
    for i in range(7, -1, -1):
        A[i].append(avail[pi]); pickno[avail[pi]] = 16 - i; pi += 1
    as_pts = {}
    for i, c in enumerate(caps):
        as_pts[int(c)] = 16 - i
    for t, k in pickno.items():
        as_pts[t] = 17 - k
    SA = [sum(S[t] for t in a) for a in A]
    w, f, th, fo, fw_w, fw_f = play_bracket(SA, beta, u)
    po = {}
    for t in A[w]:
        po[t] = 20 + min(10, 5 * fw_w)
    for t in A[f]:
        po[t] = 20 + min(10, 5 * fw_f)
    for t in A[th]:
        po[t] = 13
    for t in A[fo]:
        po[t] = 7
    return as_pts, po, A[w], A[f]


class Sim:
    def __init__(self, rules, beta, nsims=400, seed=1, matches_per_team=10):
        self.rules, self.beta, self.m = rules, beta, nsims
        self.rng = np.random.default_rng(seed)
        self.mpt = matches_per_team
        dp, rp = rules["district_points"], rules["regional_points"]
        self.dp_awards, self.rg_awards = dp["awards"], rp["awards"]
        self.alias = {}
        for k, d in rules["districts"].items():
            for a in d.get("aliases", [k]):
                self.alias[a.lower()] = k
            self.alias[k] = k

    # ------------------------------------------------------------------ one event
    def schedule(self, n):
        rng = self.rng
        nm = math.ceil(n * self.mpt / 6)
        flat = []
        while len(flat) < nm * 6:
            flat += list(rng.permutation(n))
        out, i = [], 0
        for _ in range(nm):
            chunk = []
            while len(chunk) < 6:
                t = int(flat[i]); i += 1
                if t not in chunk:
                    chunk.append(t)
            out.append((chunk[:3], chunk[3:]))
        return out

    def awards(self, mu, rookie_mask, pts, n, m, tilt_judged=0.5):
        """Vectorised judged awards via Gumbel-max. Returns dict of winner idx arrays and adds points."""
        rng = self.rng
        res = {}

        def draw(tilt, mask=None):
            g = -np.log(-np.log(rng.random((n, m)) + 1e-12) + 1e-12)
            z = tilt * mu[:, None] + g
            if mask is not None:
                z = np.where(mask[:, None], z, -1e9)
            return z.argmax(0)

        aw, ev_aw = self.cur_awards, {}
        res["impact"] = draw(0.3)
        np.add.at(pts, (res["impact"], np.arange(m)), aw["impact"])
        res["ei"] = draw(0.3)
        np.add.at(pts, (res["ei"], np.arange(m)), aw["ei"])
        if rookie_mask.any():
            res["ras"] = draw(0.0, rookie_mask)
            np.add.at(pts, (res["ras"], np.arange(m)), aw["rookie_all_star"])
        for _ in range(7):
            w = draw(tilt_judged)
            np.add.at(pts, (w, np.arange(m)), aw["other_judged"])
        return res

    def event(self, teams, mu, S, rookie_years, year, award_table, sched=None):
        """teams: list of ids; mu: (n,) mean strengths; S: (n, m) sampled strengths."""
        n, m = S.shape
        rng, beta = self.rng, self.beta
        self.cur_awards = award_table
        sched = sched or self.schedule(n)
        rp = np.zeros((n, m))
        for red, blue in sched:
            sr, sb = S[red].sum(0), S[blue].sum(0)
            rw = rng.random(m) < SIG(beta * (sr - sb))
            rp[red] += 2 * rw
            rp[blue] += 2 * (~rw)
            for alli, s in ((red, sr), (blue, sb)):
                pb = SIG(BONUS_A + BONUS_B * s)
                for _ in range(N_BONUS):
                    rp[alli] += rng.random(m) < pb
        key = rp + rng.random((n, m)) * 0.5
        order = np.argsort(-key, axis=0)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(1, n + 1)[:, None] * np.ones((1, m), int), axis=0)
        qpts = qual_points(rank, n)
        as_pts, po_pts = np.zeros((n, m)), np.zeros((n, m))
        won, fin = np.zeros((n, m), bool), np.zeros((n, m), bool)
        pnoise = rng.normal(0, 0.3, (n, m))
        U = rng.random((m, 24))
        for s in range(m):
            a, p, w, f = alliances_and_playoffs(order[:, s], S[:, s], beta, pnoise[:, s], U[s], None)
            for t, v in a.items():
                as_pts[t, s] = v
            for t, v in p.items():
                po_pts[t, s] = v
            won[w, s] = True
            for t in f:
                fin[t, s] = True
        rookie_mask = np.array([rookie_years.get(t) == year for t in teams])
        aw_pts = np.zeros((n, m))
        aw = self.awards(mu, rookie_mask, aw_pts, n, m)
        return {"rank": rank, "qual": qpts, "as": as_pts, "po": po_pts, "aw": aw_pts, "aw_win": aw,
                "won": won, "finalist": fin | won,
                "pts": qpts + as_pts + po_pts + aw_pts}

    def mini_event(self, idx, S_col):
        """Fast scalar event for DCMP (qual ranking approximated by noisy strength)."""
        rng = self.rng
        n = len(idx)
        s_loc = S_col[idx]
        order = np.argsort(-(s_loc + rng.normal(0, 0.9, n)))
        rank = np.empty(n, int); rank[order] = np.arange(1, n + 1)
        q = qual_points(rank, n)
        a, p, w, f = alliances_and_playoffs(order, s_loc, self.beta, rng.normal(0, 0.3, n), rng.random(24), None)
        pts = q.copy()
        for t, v in a.items():
            pts[t] += v
        for t, v in p.items():
            pts[t] += v
        return pts, set(w), order


# ---------------------------------------------------------------------- season driver
def run_season(sim, year, events, event_teams, strengths, rookie_years, schedules=None, log=print):
    """strengths: team -> (mean, sd) in z units. Returns output dict (JSON-able)."""
    rules, m, rng = sim.rules, sim.m, sim.rng
    schedules = schedules or {}
    all_teams = sorted({t for e in events for t in event_teams.get(e["key"], [])})
    tid = {t: i for i, t in enumerate(all_teams)}
    mu = np.array([strengths[t][0] for t in all_teams])
    sd = np.array([strengths[t][1] for t in all_teams])
    S = mu[:, None] + sd[:, None] * rng.normal(size=(len(all_teams), m))
    awards_dist = rules["district_points"]["awards"]
    awards_reg = rules["regional_points"]["awards"]

    out_events, team_pts = {}, defaultdict(list)   # team -> [(start, key, pts(m,), type, district, country)]
    impact_win = {}
    for e in sorted(events, key=lambda e: (e["start"], e["key"])):
        teams = event_teams.get(e["key"], [])
        if e["type"] not in (0, 1, 99) or len(teams) < 16:
            continue
        idx = [tid[t] for t in teams]
        sch = None
        if e["key"] in schedules:
            loc = {t: i for i, t in enumerate(teams)}
            sch = [([loc[t] for t in r], [loc[t] for t in b]) for r, b in schedules[e["key"]]
                   if all(t in loc for t in r + b)] or None
        table = awards_reg if e["type"] == 0 else awards_dist
        r = sim.event(teams, mu[idx], S[idx], rookie_years, year, table, sch)
        out_events[e["key"]] = {
            "name": e["name"], "start": e["start"], "type": e["type_name"], "district": e.get("district"),
            "teams": {int(t): {
                "p_win": round(float(r["won"][i].mean()), 4),
                "p_finalist": round(float(r["finalist"][i].mean()), 4),
                "exp_rank": round(float(r["rank"][i].mean()), 2),
                "p_top8": round(float((r["rank"][i] <= 8).mean()), 4),
                "exp_pts": round(float(r["pts"][i].mean()), 2),
            } for i, t in enumerate(teams)},
            "assumes_8_alliance_bracket": True, "scheduled": sch is not None}
        if e["type"] in (0, 1):
            for i, t in enumerate(teams):
                team_pts[t].append({"start": e["start"], "key": e["key"], "type": e["type"],
                                    "district": e.get("district"), "country": e.get("country"),
                                    "pts": r["pts"][i], "impact": r["aw_win"]["impact"] == i, "idx": i,
                                    "won": r["won"][i], "rank": r["rank"][i]})
    log(f"simulated {len(out_events)} events x {m} sims")

    # ------------------ district models
    team_out = {}
    districts_out, unmatched = {}, set()
    by_dist = defaultdict(list)
    for e in events:
        if e["type"] == 1 and e.get("district"):
            by_dist[(e["district"] or "").lower()].append(e)
    reg_alloc_total = rules["season_assumptions"]["championship_total_slots"] - rules["season_assumptions"]["championship_prequalified"]
    for dkey, devs in by_dist.items():
        cfg_key = sim.alias.get(dkey)
        if not cfg_key:
            unmatched.add(dkey)
            continue
        cfg = rules["districts"][cfg_key]
        dteams = sorted({t for e in devs for t in event_teams.get(e["key"], [])})
        if not dteams:
            continue
        T = len(dteams)
        li = {t: i for i, t in enumerate(dteams)}
        tot = np.zeros((T, m)); auto = np.zeros((T, m), bool)
        for t in dteams:
            evs = sorted([x for x in team_pts.get(t, []) if x["type"] == 1 and (x["district"] or "").lower() == dkey],
                         key=lambda x: x["start"])[: rules["district_points"]["events_counted"]]
            tot[li[t]] += age_points(year, rookie_years.get(t))
            for x in evs:
                tot[li[t]] += x["pts"]
                auto[li[t]] |= x["impact"]
        cap = cfg["dcmp_capacity"]
        keyv = tot + 1e4 * auto + rng.random((T, m)) * 0.5
        rk = np.argsort(np.argsort(-keyv, axis=0), axis=0) + 1
        qdcmp = rk <= min(cap, T)
        # DCMP + CMP per sim
        qcmp = np.zeros((T, m), bool)
        exp_dcmp_pts = np.zeros((T, m))
        alloc = cfg["cmp_slots"]
        n_imp, n_ei, n_ras = cfg.get("impact", 2), cfg.get("ei", 1), cfg.get("ras", 1)
        gidx = np.array([tid[t] for t in dteams])
        for s in range(m):
            Q = np.where(qdcmp[:, s])[0]
            if len(Q) < 16:
                continue
            Sg = S[gidx[Q], s]
            pts, winners, order = sim.mini_event(np.arange(len(Q)), Sg)
            merit = set(Q[list(winners)].tolist())
            # cultural awards at DCMP (tilted lightly by strength)
            for cnt, tilt in ((n_imp, 0.2), (n_ei, 0.2), (n_ras, 0.0)):
                p = np.exp(tilt * mu[gidx[Q]]); p /= p.sum()
                merit |= set(rng.choice(Q, size=min(cnt, len(Q)), replace=False, p=p).tolist())
            total = tot[Q, s] + 3 * pts
            exp_dcmp_pts[Q, s] = 3 * pts
            remaining = max(0, alloc - len(merit))
            cand = [q for q in Q[np.argsort(-(total + rng.random(len(Q)) * 0.4))] if q not in merit]
            qcmp[list(merit), s] = True
            qcmp[cand[:remaining], s] = True
        for t in dteams:
            i = li[t]
            team_out[t] = {"model": "district", "district": cfg_key,
                           "p_dcmp": round(float(qdcmp[i].mean()), 4), "p_cmp": round(float(qcmp[i].mean()), 4),
                           "exp_district_pts": round(float(tot[i].mean()), 1)}
        districts_out[cfg_key] = {"name": cfg["name"], "teams": len(dteams), "dcmp_capacity": cap,
                                  "cmp_slots": alloc,
                                  "expected_dcmp_teams": round(float(qdcmp.sum(0).mean()), 1),
                                  "expected_cmp_teams": round(float(qcmp.sum(0).mean()), 1)}
    # ------------------ regional model
    regs = sorted([e for e in events if e["type"] == 0 and len(event_teams.get(e["key"], [])) >= 16],
                  key=lambda e: (e["start"], e["key"]))
    if regs:
        rt = sorted({t for e in regs for t in event_teams[e["key"]]})
        li = {t: i for i, t in enumerate(rt)}
        qual = np.zeros((len(rt), m), bool); direct = np.zeros((len(rt), m), bool)
        n_direct = 0
        for e in regs:
            k = rules["season_assumptions"]["regional_direct_per_event_us" if e.get("country") in ("USA", "United States") else "regional_direct_per_event_intl"]
            n_direct += k
            teams = event_teams[e["key"]]
            ep = np.array([x["pts"] for t in teams for x in team_pts[t] if x["key"] == e["key"]])
            ep = ep + np.array([age_points(year, rookie_years.get(t)) for t in teams])[:, None]
            rows = np.array([li[t] for t in teams])
            masked = np.where(qual[rows], -1e9, ep + rng.random(ep.shape) * 0.3)
            top = np.argsort(-masked, axis=0)[:k]
            for j in range(k):
                sel = top[j]
                ok = masked[sel, np.arange(m)] > -1e8
                direct[rows[sel[ok]], np.arange(m)[ok]] = True
                qual[rows[sel[ok]], np.arange(m)[ok]] = True
        pool_slots = max(0, reg_alloc_total - sum(d["cmp_slots"] for d in rules["districts"].values()) - n_direct)
        pts2 = np.zeros((len(rt), m))
        for t in rt:
            evs = sorted([x for x in team_pts[t] if x["type"] == 0], key=lambda x: x["start"])[:2]
            for x in evs:
                pts2[li[t]] += x["pts"] + age_points(year, rookie_years.get(t))
        kv = np.where(qual, -1e9, pts2 + rng.random(pts2.shape) * 0.3)
        rk = np.argsort(np.argsort(-kv, axis=0), axis=0) + 1
        pool_q = (rk <= pool_slots) & ~qual
        for t in rt:
            i = li[t]
            team_out[t] = {"model": "regional", "p_direct": round(float(direct[i].mean()), 4),
                           "p_cmp": round(float((qual[i] | pool_q[i]).mean()), 4),
                           "exp_regional_pts": round(float(pts2[i].mean()), 1)}
        districts_out["_regional"] = {"name": "Regional model", "teams": len(rt), "direct_slots": n_direct,
                                      "pool_slots": pool_slots}
    return {"events": out_events, "teams": {int(k): v for k, v in team_out.items()},
            "districts": districts_out, "unmatched_districts": sorted(unmatched)}


# ---------------------------------------------------------------------- match predictions
def predict_matches(matches, strengths, beta):
    """matches: list of dicts with red/blue. Integrates strength uncertainty (probit approximation)."""
    out = []
    for mt in matches:
        def agg(ts):
            mean = sum(strengths.get(t, (0.0, 1.0))[0] for t in ts)
            var = sum(strengths.get(t, (0.0, 1.0))[1] ** 2 for t in ts)
            return mean, var
        mr, vr = agg(mt["red"]); mb, vb = agg(mt["blue"])
        b = beta / math.sqrt(1 + (math.pi / 8) * beta ** 2 * (vr + vb))
        p = 1 / (1 + math.exp(-b * (mr - mb)))
        out.append({"key": mt["key"], "lvl": mt["lvl"], "num": mt["num"], "set": mt["set"],
                    "red": mt["red"], "blue": mt["blue"], "rs": mt.get("rs"), "bs": mt.get("bs"),
                    "p_red": round(p, 4), "edge": round(mr - mb, 3)})
    return out
