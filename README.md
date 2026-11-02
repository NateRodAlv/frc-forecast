# FRC Forecast

Team rankings, next-season trend forecasts, match / event predictions and advancement odds
(district → district championship → Worlds; regional → Worlds). Python backend, static JS front end,
100% free hosting: **GitHub Pages + a scheduled GitHub Actions job** (no server, no database, no paid tier).

## Deploy (about 10 minutes)
1. Get a free TBA read key: thebluealliance.com → Account → *Read API Keys*.
2. Create a GitHub repo (public repos get unlimited Actions minutes), push this folder to `main`.
3. Repo → Settings → *Secrets and variables → Actions* → new secret **`TBA_KEY`**.
4. Settings → *Pages* → Source: **GitHub Actions**.
5. Actions tab → run **Refresh data and deploy site**. It then re-runs every 3 hours by itself.

Optional repo variable `TARGET_YEAR` (default 2027) changes which season is forecast.
TBA responses are cached with ETags, so refreshes after the first mostly receive HTTP 304.

## What it does
| Step | File | Notes |
|---|---|---|
| Fetch | `backend/frc/tba.py` | TBA v3: events, matches, rosters, awards, teams |
| Rank | `backend/frc/ratings.py` | Elo vs additive-score (EPA-like) vs Bradley-Terry, walk-forward backtest; best log-loss wins automatically |
| Trend forecast | `backend/frc/forecast.py` | Each team = a "price series" of yearly strength z-scores. Ridge (mean reversion + momentum) blended with **analog matching**: finds other teams whose past trajectory had the same shape (slump → rebound, breakout, decline) and uses what they did next. Blend weight set by backtest. |
| Simulate | `backend/frc/simulate.py` | Monte Carlo: quals → alliance selection → double-elim → district/regional points → DCMP → Worlds slots |
| Rules | `config/rules.json` | District slots, DCMP capacities, point values (2026 manual) |
| Site | `site/` | Rankings, team pages, events, match predictor, advancement, model report |

Local preview without a key: `python backend/run.py --synthetic && cd site && python -m http.server`
(shows a "SYNTHETIC DEMO DATA" banner). Tests: `python -m pytest -q tests`.

## Verified vs. NOT verified
Verified against official sources: district point formulas (qualification formula reproduces the manual's 40-team table
exactly), alliance-selection points, playoff points, bracket matches 1–13 + best-of-3 finals (FIRST 2026 bracket PDF),
2026 district Championship slot table and DCMP capacities, 2026 regional model (3 US / 4 intl direct, Regional Pool).

**Not verified / known gaps — read these**
- **No real-data run has been done.** The build sandbox could not reach TBA. Everything was tested on synthetic data
  (so the "best model" shown there means nothing). Your first Actions run produces the real backtest; check the *Model* tab.
- The **2027 (BIOCORE) manual releases Jan 9 2027**. Advancement numbers use 2026 rules until you update `config/rules.json`
  (2027 district slot allocations aren't published yet; Ontario is 21 in the manual table but 19 in the Dec-2025 allocation post).
- TBA district abbreviations are matched via `aliases` in rules.json (e.g. Chesapeake `chs`, California `fca`). Unmatched ones are
  listed in a yellow banner and excluded from advancement odds — fix the alias and re-run.
- Hall of Fame pre-qualified teams (`prequalified_teams`) are empty; fill in once FIRST lists them.
- Bonus ranking points are a placeholder until the 2027 game exists; this only shapes qualification seeding spread.
- Approximations: cultural-award winners are random with a mild strength tilt; declines/backups/waitlist and the Regional
  Pool's week-by-week release are ignored (final pool = top remaining by first-two-event points); teams with a single event
  aren't given projected second-event points; offseason events assume an 8-alliance bracket and give no advancement.
- 2020/2021 are skipped; strength carries across that gap with one season's worth of shrinkage.
