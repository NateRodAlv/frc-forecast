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

Replays are cached in `data/replay_cache` (keyed by a fingerprint of the code, rules and data), so later refreshes only
redo seasons whose inputs changed. The first run with the full history is the slow one (`--start-year` sets how far back to load).

Optional repo variable `TARGET_YEAR` (default 2027) changes which season is forecast.
TBA responses are cached with ETags, so refreshes after the first mostly receive HTTP 304.

## What it does
| Step | File | Notes |
|---|---|---|
| Fetch | `backend/frc/tba.py` | TBA v3: events, matches, rosters, awards, teams |
| Rank | `backend/frc/ratings.py` | 10 candidates (margin Elo, plain Elo, AC-Elo, EPA-style, Bradley-Terry, pRidge, rOPR, wOPR, Lasso OPR, defence-augmented ridge). Each is tuned on early seasons only and scored by walk-forward backtest on the latest 3. A stacked ensemble of the best few is used **only if it beats every single model**. |
| Trend forecast | `backend/frc/forecast.py` | Each team = a "price series" of yearly strength z-scores. Ridge (mean reversion + momentum) blended with **analog matching**: finds other teams whose past trajectory had the same shape (slump → rebound, breakout, decline) and uses what they did next. Blend weight set by backtest. |
| Simulate | `backend/frc/simulate.py` | Monte Carlo: quals → alliance selection → playoffs (best-of-3 single elim through 2022, double elim from 2023) → district/regional points → DCMP → Worlds slots |
| Rules | `config/rules.json` | District slots, DCMP capacities, point values (2026 manual) |
| Site | `site/` | Rankings, team pages, events, match predictor, advancement, model report |

Local preview without a key: `python backend/run.py --synthetic && cd site && python -m http.server`
(shows a "SYNTHETIC DEMO DATA" banner). Tests: `python -m pytest -q tests`.

## Forecast accuracy: the Scorecard tab
Every run replays **every past season that has enough history** (at least `--replay-min-seasons`, default 4, earlier played
seasons; history is loaded from 2002, the earliest season TBA has matches for, so that is nearly every season since 2006) as if it were still to come: the **same production code**
(`pipeline.build_forecast`) is trained on seasons before it only, forecasts it with random qualification schedules, and the
results are scored against what really happened: team strength error vs. "same as last year", preseason match accuracy,
event-winner/finalist calibration, and District Championship and Worlds odds. The tab shows one pooled summary, a
year-by-year table, and each season's full detail. Once the target season starts, its completed events are scored live.
`--replay-years 2024,2025` restricts the replays. A leak test (`tests/test_frc.py::test_cutoff_blocks_target_season_results`)
checks that scrambling the target season's results cannot change its forecast.

Deliberately **not** reported: expected-vs-actual counts (sum of probabilities vs. winners, or teams per district at DCMP /
Worlds). The simulator fills each event's three winning slots and each district's allocation exactly, so those match by
construction and say nothing about accuracy; calibration bins and Brier skill measure whether the right teams were picked.

What each season is scored on is limited to what the simulator models: event winners/finalists for every season, using that
season's real playoff format (best-of-3 single elimination through 2022, double elimination from 2023; an event is scored only
if its recorded bracket has that shape, detected from the match data), District Championship / Worlds odds only for seasons listed in
`scorecard_advancement_years` in `config/rules.json` (currently 2026, the manual transcribed there). Add a year only after
transcribing its allocations.

Caveats: rating hyperparameters (`ratings.GRID`) were tuned on seasons through 2023 and the match/playoff sharpness factors
(`pipeline.BETA_SCALE`, `simulate.PLAYOFF_SHARPEN`) by looking at the 2025 and 2026 replays, so the year-by-year table marks
which seasons are clean ("Tuning" column). Replay uses the actual event rosters (known only approximately in a real
preseason); "made Worlds / DCMP" is judged by attendance, which ignores declines and waitlist moves; forecasts do **not** yet
update as the target season is played.

## Experimental models (marked NEW on the Model page)
New variants are added **beside** the originals, never in place of them: both appear in the Model tab's tables and the
original stays selectable. A NEW model only drives the forecasts when it beats the original on the same walk-forward seasons by a margin
(`pipeline.TREND2_MARGIN`, `pipeline.SD2_MARGIN`); the **IN USE** badge shows which one does.

| Model | Replaces | What is different | Backtest (rebuilt history, see caveat) |
|---|---|---|---|
| **Trend v2** (`forecast.Trend2`) | Trend model | The boosted-tree half also sees how many matches last season's rating rests on, the team's long-run mean, its best/worst recent seasons, a 3-season change and the number of seasons seen. | RMSE 0.6393 vs 0.6420 (2023-26), 0.7471 vs 0.7511 (2013-19) |
| **Variance model v2** (`forecast.SdModel`) | Spread of analog teams | Predicts each team's squared forecast error (out-of-fold) from its features and the season, so it follows the era. The analog spread was ~15% too wide in 2023-26 (a nominal 80% range covered 87%), which flattens simulated win odds. | log score 0.923 vs 0.954 (2023-26), 1.108 vs 1.131 (2013-19); 80% range covers 85% vs 87% |
| **Seeded rating models** (`ratings.SEEDED`: `pridge_seeded`, `dpr_seeded`, `epa_seeded`) | pRidge, dRidge, EPA | Start every season from a preseason forecast (made only from earlier seasons) instead of only `carry x last rating`. | In-season match log loss 0.4555 -> 0.4505 on held-out 2024-26 (accuracy 78.4% -> 78.7%), biggest early in events. Preseason forecasts and event odds unchanged. |

Full tuning log (values, before/after accuracies, what was tried and left alone): [TUNING.md](TUNING.md).

Caveats: these numbers were measured offline on a history rebuilt from the site's `teams.json` export, which only contains teams
active since 2024 (so defunct teams are missing and absolute RMSEs run ~0.006 above the live backtest). Only the *relative* comparison is
meaningful, and it was consistent across both windows, but the gain is small (~0.4% RMSE). The next real build re-measures both on the full data and
the gate decides. The Scorecard also now reports `coverage_sd_80`: how often the simulator's own forecast ± 1.28 × spread contains the real rating.
Changing the spread changes the simulated win odds, so re-check `PLAYOFF_SHARPEN` against the next scorecard rather than stacking both fixes.

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
