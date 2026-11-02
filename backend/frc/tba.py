"""Thin The Blue Alliance v3 client with ETag caching.

Needs env var TBA_KEY (free: https://www.thebluealliance.com/account -> Read API key).
Responses are cached on disk (data/cache) with ETags so scheduled refreshes only
re-download what changed (TBA returns 304, which is cheap and polite).
"""
import hashlib
import json
import os
import time
from pathlib import Path

import requests

BASE = "https://www.thebluealliance.com/api/v3"
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"

EVENT_TYPES = {
    0: "regional", 1: "district", 2: "district_cmp", 3: "cmp_division",
    4: "cmp_finals", 5: "district_cmp_division", 6: "foc",
    99: "offseason", 100: "preseason",
}


class TBA:
    def __init__(self, key=None, user_agent="frc-forecast (github pages model)"):
        self.key = key or os.environ.get("TBA_KEY")
        if not self.key:
            raise RuntimeError("Set TBA_KEY (free read key from thebluealliance.com/account)")
        self.s = requests.Session()
        self.s.headers.update({"X-TBA-Auth-Key": self.key, "User-Agent": user_agent})
        CACHE.mkdir(parents=True, exist_ok=True)

    def get(self, path, retries=3):
        h = hashlib.sha1(path.encode()).hexdigest()
        body_f, etag_f = CACHE / f"{h}.json", CACHE / f"{h}.etag"
        headers = {}
        if body_f.exists() and etag_f.exists():
            headers["If-None-Match"] = etag_f.read_text()
        for i in range(retries):
            r = self.s.get(BASE + path, headers=headers, timeout=30)
            if r.status_code == 304 and body_f.exists():
                return json.loads(body_f.read_text())
            if r.status_code == 200:
                body_f.write_text(r.text)
                if "ETag" in r.headers:
                    etag_f.write_text(r.headers["ETag"])
                return r.json()
            if r.status_code == 404:
                return None
            time.sleep(2 ** i)
        r.raise_for_status()

    # ---- convenience wrappers returning slim dicts ----
    def events(self, year):
        out = []
        for e in self.get(f"/events/{year}") or []:
            out.append({
                "key": e["key"], "year": e["year"], "name": e["name"],
                "type": e["event_type"], "type_name": EVENT_TYPES.get(e["event_type"], "other"),
                "district": (e.get("district") or {}).get("abbreviation"),
                "start": e["start_date"], "end": e["end_date"], "week": e.get("week"),
                "country": e.get("country"), "state": e.get("state_prov"),
                "parent": e.get("parent_event_key"),
                "division_keys": e.get("division_keys") or [],
            })
        return out

    def event_teams(self, ev_key):
        return [int(t[3:]) for t in (self.get(f"/event/{ev_key}/teams/keys") or [])]

    def event_matches(self, ev_key):
        out = []
        for m in self.get(f"/event/{ev_key}/matches/simple") or []:
            red, blue = m["alliances"]["red"], m["alliances"]["blue"]
            out.append({
                "key": m["key"], "lvl": m["comp_level"], "set": m["set_number"],
                "num": m["match_number"], "time": m.get("predicted_time") or m.get("time"),
                "red": [int(t[3:]) for t in red["team_keys"]],
                "blue": [int(t[3:]) for t in blue["team_keys"]],
                "rs": red["score"] if red["score"] is not None and red["score"] >= 0 else None,
                "bs": blue["score"] if blue["score"] is not None and blue["score"] >= 0 else None,
            })
        return out

    def event_alliances(self, ev_key):
        a = self.get(f"/event/{ev_key}/alliances") or []
        return [[int(t[3:]) for t in x["picks"]] for x in a]

    def event_awards(self, ev_key):
        return [{"type": a["award_type"], "name": a["name"],
                 "teams": [int(r["team_key"][3:]) for r in a["recipient_list"] if r.get("team_key")]}
                for a in (self.get(f"/event/{ev_key}/awards") or [])]

    def teams_year(self, year):
        """Return {team: {name, rookie_year, state, country}} for all teams registered that year."""
        out, p = {}, 0
        while True:
            page = self.get(f"/teams/{year}/{p}")
            if not page:
                break
            for t in page:
                out[t["team_number"]] = {"name": t.get("nickname"), "rookie_year": t.get("rookie_year"),
                                         "state": t.get("state_prov"), "country": t.get("country")}
            p += 1
        return out

    def team_rookie_years(self, teams):
        out = {}
        for t in teams:
            d = self.get(f"/team/frc{t}/simple")
            if d:
                out[t] = d.get("rookie_year")
        return out
