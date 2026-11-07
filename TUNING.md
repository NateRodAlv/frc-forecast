# Tuning log

What was tuned, which values changed, and how the accuracies moved. Everything here was measured offline on the full
TBA cache of the 2026-11-06 build (the same data your Actions run uses; the offline loader reproduced the published
leaderboard to four decimals before anything was changed). "Held-out" means seasons the choice never saw.

## Bottom line

| What | Before | After | Verdict |
|---|---|---|---|
| In-season match prediction (rating models), held-out 2024-26, 49,381 matches | log loss **0.4555**, Brier 0.1489, accuracy 78.39% | log loss **0.4505**, Brier 0.1470, accuracy **78.69%** | **Better**, in all 3 held-out seasons |
| Preseason forecasts and event win / finalist odds (6 replayed seasons) | match LL 0.5487, event LL 0.3186 | match LL 0.5484, event LL 0.3186 | **Unchanged** (noise, mixed signs) |
| `BETA_SCALE`, `PLAYOFF_SHARPEN`, spread scale | 1.25, 1.5, 1.0 | 1.25, 1.5, 1.0 | **Already optimal**, not changed |

The one real gain is in-season match prediction. It does not move preseason forecasts or the simulated event odds.

## 1. New: seeded rating models (`pridge_seeded`, `dpr_seeded`, `epa_seeded`)

Separate models, tagged NEW on the Model page next to the originals (which are unchanged). Each starts every season from
a **preseason forecast** (Trend v2, fitted only on earlier seasons) instead of only `carry x last season's rating`.
`mix` is how far the carried-over state is moved toward the forecast (0 = identical to the original model, verified
bit for bit by a test). Teams the model has never seen start at the average first-season level.

Held-out 2024-26 match log loss (lower is better), each model alone:

| Model | Original | Seeded | Change |
|---|---|---|---|
| pRidge | 0.4566 | 0.4533 | -0.0033 |
| dRidge (`dpr`) | 0.4810 | 0.4733 | -0.0077 |
| EPA | 0.4753 | 0.4689 | -0.0064 |
| **Selected ensemble** | **0.4555** (pridge .53 + epa .22 + dpr .28) | **0.4505** (pridge_seeded .81 + dpr_seeded .26) | **-0.0050** |

Ensemble, held-out, by season (old -> new): 2024 0.4876 -> 0.4833, 2025 0.4485 -> 0.4427, 2026 0.4331 -> 0.4284.
Accuracy 78.39% -> 78.69%.

Where the gain is: by position within a qualification event (0% = first match) the improvement is
-0.0110 (first 10%), -0.0075 (10-25%), -0.0067 (25-50%), -0.0041 (50-75%), -0.0024 (last 25%). By match type:
qualification -0.0055, playoffs -0.0021. A better starting point matters most before the model has seen the event.

### Tuning values

Each model has two grid points in `ratings.GRID_SEEDED`; model selection picks by training-season log loss as for every
other model. The "searched" point came from random search (40 points) + local refinement (24) on seasons < 2024.

| Model | Point | Parameters | Train (< 2024) | Held-out 2024-26 |
|---|---|---|---|---|
| pRidge | original, unseeded | lam 2.8, carry 0.86, eta 0.92, decay 0.987, every 4 | 0.5701 | 0.4566 |
| | seeded, original params | same + mix 1.0 | 0.5652 | 0.4535 |
| | **seeded, searched** | lam 3.92, carry 0.721, eta 0.974, decay 0.991, every 2, mix 0.906 | 0.5642 | 0.4536 |
| dRidge | original, unseeded | lam 5.8, carry 0.65, eta 0.835, decay 1.0 | 0.5808 | 0.4810 |
| | seeded, original params | same + mix 1.0 | 0.5754 | 0.4720 |
| | **seeded, searched** | lam 6.32, carry 0.467, eta 0.908, decay 0.994, mix 0.87 | 0.5748 | 0.4737 |
| EPA | original, unseeded | carry 0.46, lr0 1.47, lr_min 0.18, prior_n 2.9 | 0.5791 | 0.4753 |
| | seeded, original params | same + mix 1.0 | 0.5774 | 0.4682 |
| | **seeded, searched** | carry 0.243, lr0 1.49, lr_min 0.155, prior_n 4.15, mix 0.786 | 0.5714 | 0.4691 |

Honest reading: the searched points are better on the seasons they were tuned on, but **not on 2024-26**. The gain is
the seeding itself, not the fine-tuning, so both points stay in the grid.

Other values: ensemble size `n_members` 3 -> **4** (chosen by training log loss; the 4-member ensemble scored 0.4506 vs
0.4517 with 3 on the held-out seasons, and lets `dpr_seeded` contribute). `pipeline.SEEDED_MARGIN` = 0.0005: the seeded
models drive the forecasts only if the best selection with them beats the best without them by this much on the
held-out seasons. `mix`, the seed model (Trend v2) and `min_matches` = 8 for the scale mapping are described in
`ratings.SeedMixin`.

## 2. Trend v2 and variance model v2 (from the previous patch), confirmed on the full data

The gate picked both in all 12 replayed seasons (2013-19, 2022-26) and in the final build.

| | Original | New |
|---|---|---|
| Next-season strength RMSE (final ratings) | 0.6367 | **0.6345** |
| Forecast-spread log score | 0.9375 | **0.9174** |
| 80% range hit rate (ideal 80%) | 86.7% | 84.7% |
| Event win + finalist log loss, 12 seasons | 0.3101 | **0.3087** (better in 11 of 12) |

## 3. Checked and left alone

* **`BETA_SCALE` 1.25, `PLAYOFF_SHARPEN` 1.5, spread scale 1.0.** 120 combinations (beta x0.8-1.2, sharpen 1.0-2.5,
  spread x0.8-1.1) over 12 seasons: the best beat the current setting by 0.0006 in event log loss, and
  leave-one-season-out gave **+0.0001** (no gain). They sit on a flat optimum.
* **Recency-weighted training of Trend v2** (half-life 3): RMSE 0.6784 -> 0.6763 (0.3%), but it closes only ~30% of the
  recent top-end lag. Not adopted.
* **Rolling recalibration from the last 3 seasons' errors:** strength RMSE -0.5%, event log loss -0.0002. Not adopted.
* **Seeded models on preseason forecasts / event odds:** no effect (their seeds *are* the preseason forecast).

## 4. Still open

Event win odds are under-confident at the top (a predicted 30-50% happens ~47% of the time) and the extreme low end is
slightly too thin. None of the knobs above moves it, so it is probably not a scale problem; the structure of how
alliances are formed and combined is the next thing to look at.

## 5. Caveats

* Rating hyperparameters in the grid were tuned on seasons < 2024, so replayed seasons before 2024 are partly in-sample
  for them (`pipeline.TUNED_THROUGH`). The held-out numbers above (2024-26) are the clean ones.
* The seeds come from a two-pass build: pass 1 = the unseeded models, pass 2 = seeded models seeded from pass 1's
  ratings. Every season's seed uses only earlier seasons (tested). The first build after this change takes longer
  (~1.5x on the rating step, and replay caches rebuild).
* The next real build re-measures everything on the live data; the gates decide what drives the forecasts.
