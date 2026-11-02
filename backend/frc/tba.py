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


def tnum(key):
    """'frc254' -> 254. Offseason-only duplicate teams have suffixed keys ('frc254B'): return None."""
    k = key[3:]
    return int(k) if k.isdigit() else None


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
        last = None
        for i in range(max(retries, 6)):
            try:
                r = self.s.get(BASE + path, headers=headers, timeout=30)
            except (requests.ConnectionError, requests.Timeout) as ex:   # dropped connection, DNS, timeout
                last = ex
                time.sleep(min(30, 2 ** i))
                continue
            if r.status_code == 304 and body_f.exists():
                return json.loads(body_f.read_text())
            if r.status_code == 200:
                body_f.write_text(r.text)
                if "ETag" in r.headers:
                    etag_f.write_text(r.headers["ETag"])
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code in (401, 403):
                raise RuntimeError(f"TBA rejected the request ({r.status_code}): check the TBA_KEY secret")
            last = RuntimeError(f"HTTP {r.status_code} for {path}")
            time.sleep(min(30, 2 ** i))
        raise last

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
        return [n for n in (tnum(t) for t in (self.get(f"/event/{ev_key}/teams/keys") or [])) if n is not None]

    def event_matches(self, ev_key):
        out = []
        for m in self.get(f"/event/{ev_key}/matches/simple") or []:
            red, blue = m["alliances"]["red"], m["alliances"]["blue"]
            rt = [tnum(t) for t in red["team_keys"]]
            bt = [tnum(t) for t in blue["team_keys"]]
            if None in rt or None in bt or len(rt) < 3 or len(bt) < 3:
                continue   # skip matches involving suffixed offseason teams / malformed rows
            out.append({
                "key": m["key"], "lvl": m["comp_level"], "set": m["set_number"],
                "num": m["match_number"], "time": m.get("predicted_time") or m.get("time"),
                "red": rt, "blue": bt,
                "rs": red["score"] if red["score"] is not None and red["score"] >= 0 else None,
                "bs": blue["score"] if blue["score"] is not None and blue["score"] >= 0 else None,
            })
        return out

    def event_alliances(self, ev_key):
        a = self.get(f"/event/{ev_key}/alliances") or []
        return [[n for n in (tnum(t) for t in x["picks"]) if n is not None] for x in a]

    def event_awards(self, ev_key):
        return [{"type": a["award_type"], "name": a["name"],
                 "teams": [n for n in (tnum(r["team_key"]) for r in a["recipient_list"] if r.get("team_key")) if n is not None]}
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
