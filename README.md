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
| Rank | `backend/frc/ratings.py` | 10 candidates (margin Elo, plain Elo, AC-Elo, EPA-style, Bradley-Terry, pRidge, rOPR, wOPR, Lasso OPR, defence-augmented ridge). Each is tuned on early seasons only and scored by walk-forward backtest on the latest 3. A stacked ensemble of the best few is used **only if it beats every single model**. |
| Trend forecast | `backend/frc/forecast.py` | Each team = a "price series" of yearly strength z-scores. Ridge (mean reversion + momentum) blended with **analog matching**: finds other teams whose past trajectory had the same shape (slump → rebound, breakout, decline) and uses what they did next. Blend weight set by backtest. |
| Simulate | `backend/frc/simulate.py` | Monte Carlo: quals → alliance selection → double-elim → district/regional points → DCMP → Worlds slots |
| Rules | `config/rules.json` | District slots, DCMP capacities, point values (2026 manual) |
| Site | `site/` | Rankings, team pages, events, match predictor, advancement, model report |

Local preview without a key: `python backend/run.py --synthetic && cd site && python -m http.server`
(shows a "SYNTHETIC DEMO DATA" banner). Tests: `python -m pytest -q tests`.

## Forecast accuracy: the Scorecard tab
Every run replays the previous season as if it were still to come: the **same production code** (`pipeline.build_forecast`) is
trained on seasons before it only, forecasts it with random qualification schedules, and the results are scored against what
really happened — team strength error vs. "same as last year", preseason match accuracy, event-winner/finalist calibration,
District Championship and Worlds odds, and expected-vs-actual slot counts per district. Once the target season starts, its
completed events are scored live. A leak test (`tests/test_frc.py::test_cutoff_blocks_target_season_results`) checks that
scrambling the target season's results cannot change its forecast.

Caveats: replay uses the actual event rosters (known only approximately in a real preseason); "made Worlds / DCMP" is judged by
attendance, which ignores declines and waitlist moves; forecasts do **not** yet update as the target season is played.

## Team age
Age = season − rookie year + 1 (rookie years from TBA). The ridge forecaster has rookie / 2nd / 3rd-year and log-age terms, and
analog matching only compares a team with past team-seasons of the same age band (1, 2, 3, 4, 5-7, 8-12, 13+) AND the same amount of
history, using only the history that exists (no padding). Pools that are too small widen automatically. The Model tab shows
the backtest with and without age matching, and error by age.

## Normalisation
Scores are divided by the season's mean alliance score before any model sees them, and each season's team ratings are
standardised (mean 0, sd 1) over teams with at least 8 matches that season. Team pages show that rating, the percentile and the
match count for every season on hover.

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
- The extra models (pRidge, AC-Elo, etc.) are implemented from short descriptions, not from reference code, so details
  (e.g. AC-Elo's K schedule) are my interpretation. Their claimed improvements are unverified; the backtest decides what is used.
  COPR and lOPR are not included (they need game-specific score breakdowns; COPR adds nothing for total-score prediction), and
  "ACE" has no published spec I could find, so the learned ensemble stands in for it.
- 2020/2021 are skipped; strength carries across that gap with one season's worth of shrinkage.
