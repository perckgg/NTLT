# Observations written from real run results (data 2025-07-04 → 2026-11-14, Hanoi station Nhan Chinh,
# random_state=42, scikit-learn 1.9.1). Every number is printed by a cell of the notebook.
NOTES = {
"EDA": """
(Every number below is printed by a cell above: the statistics table in 1.3, the monthly table under Figure 3, the printout under Figure 4,
the Spearman printout under Figure 5, the autocorrelation table, and the missing-value tables of Section 2.1.)
- **Similar average pollution, different shapes:** mean hourly PM2.5 is 32.3 µg/m³ in Hanoi and 31.2 µg/m³ in HCMC.
  Hanoi is more volatile (std 20.4 vs 15.6 µg/m³).
- **WHO guideline (24-hour mean ≤ 15 µg/m³):** it is exceeded on **92.2% of the 476 valid days in Hanoi** and **97.6% of the 498 valid days in HCMC**
  (valid day = at least 18 measured hours). Both cities are almost always above the guideline.
- **Seasonality (monthly means):** Hanoi has a clear winter pollution season (December 54.1, January 48.7 µg/m³) and is cleanest in August (22.8),
  a factor of about 2.4. HCMC is also most polluted at the turn of the year (January 45.0, December 44.4) and cleanest in July (23.4).
  Figure 3 shows medians, which are a few µg/m³ lower (e.g. Hanoi December: median 51.7) because the distribution is right-skewed.
  The data covers **one** winter only, so each winter month is a single observation of that month.
- **Daily cycle (complete days only):** HCMC peaks at the 7h morning rush hour in both half-years (45.6 µg/m³ in Nov–Apr, 36.1 in May–Oct).
  Hanoi in Nov–Apr is highest at night (23h: 43.6) and lowest in the afternoon (15h: 37.9), consistent with a shallow night-time boundary layer
  trapping particles near the ground and daytime mixing diluting them; in May–Oct it peaks in the morning (9h: 35.3) and is lowest at 18h (18.5).
- **Correlation with weather (Spearman):** in Hanoi PM2.5 is lower with cloud cover (−0.29) and rain (−0.27) and higher with surface pressure (+0.26):
  dry, high-pressure winter air lets particles accumulate. In HCMC the strongest link is wind speed (−0.38): wind disperses particles.
  These are monotonic associations, not causal effects (the season drives several of them at once).
- **Autocorrelation:** 0.95 (Hanoi) / 0.88 (HCMC) at lag 1h, but only 0.59 / 0.47 at lag 24h. Forecasting 24h ahead is therefore much harder
  than forecasting the next hour. In HCMC the autocorrelation at 24h is higher than at 12h (0.47 vs 0.27): the same hour of the day tends to repeat.
- **Data quality:** Hanoi is missing 3.6% of PM2.5 hours (65 gaps, 48 of them ≤ 6h, longest 86h); the worst month is June 2026 (28% missing).
  HCMC is missing 0.6% (longest gap 8h). No value is ≤ 0 or > 1000 and no stuck-sensor run is found in either city (Section 2.2).
""",

"SPLIT": """
**Figure caption (split):** both cities use the same dates. Development has 9,598 (Hanoi) / 10,076 (HCMC) samples and Test 1,775 samples in each city
(01/09/2026 – 13/11/2026). The mean target is similar in Development and Test (Hanoi 32.3 vs 32.7, HCMC 31.0 vs 32.4 µg/m³), but the Test period
is the **start** of the polluted season, not its peak. The walk-forward training window grows from 2,075 (Hanoi) / 2,100 (HCMC) samples in fold 10/2025 to 8,870 / 9,336 in fold 08/2026;
each validation month has 653–743 samples, except Hanoi in June 2026 (487) because of the gaps in that month.
""",

"VAL": """
**Walk-forward findings** (Section 3.2; pooled over the 11 months 10/2025 – 08/2026):
- **Hyperparameters:** cross-validation prefers simple models: HGB with at most 4 leaves per tree in both cities, Random Forest with depth 8 (Hanoi) or 4 (HCMC).
  Several selected values lie at the edge of the grid (flagged in the printout). The grid was widened twice and the selected values moved to the
  simpler end each time; we stopped there and state it as a limitation. A noisy 24h-ahead target with one year of data favours simple models.
- **Naive blend:** `w` stays between 0.65 and 0.75 in Hanoi and 0.45–0.55 in HCMC across the folds, so the blend is a stable reference.
  It clearly beats Persistence (MAE 13.358 vs 13.788 in Hanoi, 9.921 vs 10.895 in HCMC).
- **ML vs the Naive blend:** all three ML models beat it on the pooled forecasts. **Hanoi:** RandomForest 12.453, Ridge 12.521, HGB 12.646 vs 13.358.
  **HCMC:** HGB 9.593, Ridge 9.630, RandomForest 9.751 vs 9.921. The selected models are **RandomForest (Hanoi)** and **HistGradientBoosting (HCMC)**,
  but the pooled MAEs of the three ML models differ by less than 2%.
- **The gain is not uniform over the year (figure above).** In Hanoi, RandomForest is far better than the blend in the winter peak
  (January 15.54 vs 19.36, February 16.31 vs 20.53) but **worse in November 2025 (17.54 vs 14.96) and April 2026 (12.12 vs 11.26)**.
  In HCMC, HGB is better than the blend in the polluted months October–January (e.g. December 12.87 vs 14.67) and in May and August,
  but worse in February (11.45 vs 10.40), March, April, June and July.
- **Why (bias table):** the ML models lag behind changes of the pollution level. When the level rises, they forecast too low:
  Hanoi RandomForest has a bias of −7.8 µg/m³ in November (mean measured PM2.5 38.4) and −7.1 in April (40.1, up from 30.3 in March);
  HCMC HGB −6.9 in October. When it falls they forecast too high (HCMC February +5.8, as the mean drops from 44.8 to 32.9).
  The Naive blend has no intercept and simply follows the current level, so its monthly bias stays within ±2.5 µg/m³.
  In November 2025 the Hanoi model had only seen July–October, the cleaner months.
""",

"CREATIVE": """
**Figure 6 caption:** On the independent Test set (01/09 – 13/11/2026), the best model of each city (chosen by walk-forward validation) reaches
**MAE 10.91 µg/m³ in Hanoi (RandomForest)** and **11.07 µg/m³ in HCMC (HistGradientBoosting)**. Against the two reference forecasts:

| City | vs Persistence | vs Naive blend (strongest naive baseline) |
|---|---|---|
| Hanoi (RandomForest) | 12.07 → 10.91 (−9.6%) | 11.53 → 10.91 (**−5.3%**) |
| HCMC (HistGradientBoosting) | 12.88 → 11.07 (−14.1%) | 11.93 → 11.07 (**−7.2%**) |

Comparing only with Persistence overstates the benefit of machine learning: part of the gain comes from simply smoothing the current value
with its 24h mean. The honest answer to the question is therefore **about 5% in Hanoi and 7% in HCMC** beyond the best naive forecast on Test
(6.8% / 3.3% on the walk-forward months). On Test, Ridge is actually the best model in Hanoi (10.55, −8.4% vs the blend), although it was not selected:
the ML models are too close for the selection to be reliable (Section 4.3). In HCMC the Test R² is low for every model (at most 0.166):
day-to-day variations in the autumn are hard to forecast from the current level and weather.

Only **measured** PM2.5 values are used as targets (no interpolated targets), and the features use a causal forward-fill only.
(Tree-model results can shift by a few hundredths of µg/m³ between scikit-learn versions; Ridge and the baselines do not change.)
""",

"BOOTSTRAP": """
**Figure 7 caption — which differences are real?** (95% CI, 3-day blocks; *significant* = the interval excludes 0)

| City | Comparison | Walk-forward (10/2025–08/2026) | Test (09–11/2026) |
|---|---|---|---|
| Hanoi | RandomForest vs Persistence | **+9.7% [+2.0, +16.5]** | **+9.6% [+1.5, +17.3]** |
| Hanoi | RandomForest vs Naive blend | +6.8% [−0.9, +13.7] | +5.3% [−3.4, +13.3] |
| Hanoi | RandomForest vs Ridge | +0.5% [−2.5, +3.8] | −3.4% [−9.3, +1.6] |
| HCMC | HGB vs Persistence | **+12.0% [+7.4, +15.8]** | **+14.1% [+8.7, +19.4]** |
| HCMC | HGB vs Naive blend | +3.3% [−1.4, +7.6] | **+7.2% [+1.7, +12.6]** |
| HCMC | HGB vs RandomForest | **+1.6% [+0.3, +3.0]** | −2.2% [−5.0, +0.5] |

- **Against Persistence**, the best model is significantly better in both cities, on both evaluation sets.
- **Against the Naive blend (the pre-specified comparisons):**
  - **HCMC:** significant on Test (+7.2%, CI +1.7% to +12.6%, positive for 1-, 3- and 7-day blocks), not significant on walk-forward (+3.3%).
  - **Hanoi:** not significant on Test (+5.3%, CI −3.4% to +13.3%). On walk-forward the result is borderline: the interval reaches just below 0 for 1- and
    3-day blocks and stays above 0 for 7-day blocks. We therefore do **not** claim a significant gain in Hanoi.
  - The point estimates are positive everywhere, so the most likely situation is a small real gain (a few %) that one year of data cannot pin down.
- **Between the ML models**, almost no difference is significant, and the only one (HCMC, HGB vs RandomForest on walk-forward, +1.6%) reverses sign on Test.
  *Which ML model is best is not established in either city.*
- **Block length:** the autocorrelation of the daily error difference is small (Hanoi 0.15 / −0.06 / 0.02 at 1 / 2 / 3 days; HCMC 0.16 / 0.20 / 0.10),
  and the conclusions above do not change between 1-, 3- and 7-day blocks except for the borderline Hanoi walk-forward case.
- **Method note:** an i.i.d. bootstrap over single hours would give much narrower intervals and declare almost everything significant;
  the blocks keep the strong hour-to-hour correlation of forecast errors.
""",

"FORECAST": """
**Figure 8 caption:** In this window the Hanoi model follows the actual series better than the Naive blend (MAE 10.99 vs 12.42 µg/m³ over 337 hours),
while in HCMC it is slightly worse (11.45 vs 10.85): two weeks are too short to judge a model, which is why Section 4.3 uses the whole sets.
In both cities the forecast is much smoother than reality: its standard deviation is less than half of the actual one
(Hanoi 7.2 vs 15.5, HCMC 5.5 vs 18.5 µg/m³), so short spikes are missed. This is the visual counterpart of the under-forecasting of polluted hours in Section 4.5.
""",

"ERROR": """
**Error analysis findings:**
- **By hour:** the error is highest for forecasts of the early morning (Hanoi 6h: 13.8 µg/m³ on both sets; HCMC walk-forward 7h: 11.1) and lowest
  in the late afternoon or evening (Hanoi 17h: 10.6 walk-forward, 8.2 Test; HCMC 19h–20h: 7.5 / 8.5), i.e. where the night-time accumulation and the
  morning rush hour make the level change fastest. The error by month is the walk-forward figure of Section 3.2.
- **By EPA concentration band (Test):** the correct band is predicted for 59.3% (Hanoi) and 60.5% (HCMC) of hours, 96.7% / 96.8% within ±1 band
  (walk-forward: 61.0% / 66.9%). The main error is **under-forecasting polluted hours**: in Hanoi only 46 of the 215 "Unhealthy" hours are predicted
  correctly and 271 of the 455 "USG" hours are predicted as "Moderate"; in HCMC only 2 of the 136 "Unhealthy" hours are correct.
  No model forecast is ever "Good" (the 132 / 37 "Good" hours are all over-forecast): the forecasts are pulled towards the mean,
  the expected behaviour of a model trained to minimize MAE on a noisy target.
- **Practical consequence:** the model is useful for following the general level, but **not reliable as an early-warning tool for severe pollution episodes**,
  where a missed alarm costs more than a false alarm.
""",

"IMPORTANCE": """
**Findings (Test set):**
- **Hanoi (RandomForest):** the current PM2.5 value dominates (`pm25_lag_0`: +2.37 µg/m³ when shuffled); the next features are weather
  (surface pressure +0.17, 24h rain +0.17, wind speed +0.12). By group: **PM2.5 history +3.82, weather +0.95, calendar −0.06**.
  A negative value means shuffling the calendar features made the error slightly *smaller*: they carry no signal on this period.
- **HCMC (HistGradientBoosting):** the current value is still first (+0.64), followed by the **hour of day** (`hour_cos` +0.24) and 24h rain (+0.10).
  By group: **PM2.5 history +1.55, calendar +0.26, weather +0.17** — HCMC's PM2.5 follows the daily traffic rhythm (EDA: 7h peak).

The drivers differ between the cities (Hanoi: recent level + weather; HCMC: recent level + time of day), which supports training one model per city.
""",

"CONCLUSION": """
**Conclusions:**
- We built a 24h-ahead PM2.5 forecasting pipeline, applied identically but **trained independently** for Hanoi and HCMC, using reference-station
  (Hanoi) and low-cost-sensor (HCMC) data from OpenAQ combined with Open-Meteo weather, over the same period for both cities (04/07/2025 – 14/11/2026).
  Features use only data available at forecast time (causal forward-fill), and models are evaluated only on measured values.
- Models are selected by **walk-forward validation** over 11 months that include the whole winter, and evaluated once on Test (09–11/2026).
- The best model lowers MAE by **5.3% in Hanoi (RandomForest)** and **7.2% in HCMC (HistGradientBoosting)** on Test compared with the
  strongest naive baseline (Naive blend), and by 9.6% / 14.1% compared with Persistence.
- Block-bootstrap intervals: the gain over Persistence is significant everywhere; the gain over the Naive blend is significant only for HCMC on Test;
  which ML model is best is not established in either city.
- The ML gain is concentrated in the stable winter peak; in months where the pollution level changes (Hanoi November and April, HCMC October and February)
  the ML models lag behind and can be worse than the Naive blend.

**Limitations:**
- **One winter only** (the Hanoi station started in 07/2025): the models never see two winters, and the walk-forward winter months are the only evidence
  about winter performance. The Test period covers the start of the polluted season, not its peak.
- One station per city; the HCMC station is a low-cost sensor (AirGradient), possibly less accurate at high humidity, and the only HCMC station with continuous data.
- Hyperparameters were tuned on the whole Development set, so the walk-forward scores of the ML models are slightly optimistic; several selected values lie at the
  edge of the grid. The Test scores are not affected.
- Weather features use observations at $t$, not weather forecasts for $t+24h$. The models under-forecast high-pollution episodes.

**Future work:**
- Re-run the walk-forward evaluation after a second winter (from 12/2026) to check whether the winter gain repeats.
- Add **weather forecasts** for the target time and more stations per city; add features that react to level changes (e.g. the change of the 24h mean over several days).
- Use quantile regression or sample weights that emphasize high-pollution hours, or a threshold classifier for early warning.
- Re-tune hyperparameters inside each walk-forward fold (nested tuning) to remove the optimism of the walk-forward scores.
""",
}
