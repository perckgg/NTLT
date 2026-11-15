# Observations written from real run results (data 2024-12-01 → 2026-09-30, random_state=42, scikit-learn 1.9.1).
NOTES = {
"EDA": """
(Every number below is printed by a cell above: the statistics table in 1.3, the monthly table under Figure 3, the printout under Figure 4,
the Spearman printout under Figure 5 and the autocorrelation table.)
- **Similar average pollution, different shapes:** mean hourly PM2.5 is 31.7 µg/m³ in Hanoi and 32.1 µg/m³ in HCMC.
  Hanoi is more volatile (std 20.6 vs 16.3; max 332 vs 171 µg/m³).
- **WHO guideline (24-hour mean ≤ 15 µg/m³):** it is exceeded on **91.3% of the 389 valid days in Hanoi** and **98.2% of the 613 valid days in HCMC**
  (valid day = at least 18 measured hours). Both cities are almost always above the guideline.
- **Seasonality (monthly means):** Hanoi has a clear winter pollution season (December 53.3, January 55.0 µg/m³) and is cleanest in August (19.4),
  almost a 3× difference. HCMC is also most polluted at the turn of the year (January 46.2) and cleanest in July (23.5), with a narrower range.
  Figure 3 shows medians, which are a few µg/m³ lower (e.g. Hanoi January: median 52.8) because the distribution is right-skewed.
- **Daily cycle (complete days only):** HCMC peaks in the morning rush hour in both half-years (7h: 45.7 µg/m³ in Nov–Apr, 35.6 in May–Oct).
  Hanoi in Nov–Apr peaks around midnight (0h: 43.4) and is lowest at 17h (38.7), consistent with a shallow night-time boundary layer
  trapping dust near the ground and daytime mixing diluting it; its daily cycle is weak compared with its seasonal cycle.
  Without the complete-day filter the Hanoi curve shows a spurious noon peak, an artefact of the missing afternoons after 11/2025.
- **Correlation with weather (Spearman):** in Hanoi PM2.5 is lower with rain (−0.29), cloud cover (−0.28) and higher temperature (−0.26), and higher with
  surface pressure (+0.28): cold, dry, high-pressure winter air lets dust accumulate. In HCMC the strongest link is wind speed (−0.38): wind disperses dust.
  These are monotonic associations, not causal effects (season drives several of them at once).
- **Autocorrelation:** 0.92 (Hanoi) / 0.88 (HCMC) at lag 1h, but only 0.56 / 0.45 at lag 24h. Forecasting 24h ahead is therefore much harder
  than forecasting the next hour and needs weather, season and daily-cycle features. (In HCMC the autocorrelation at 24h is higher than at 12h, 0.45 vs 0.27:
  the same hour of the day tends to repeat.)
- **Data quality:** Hanoi is missing 32.8% of PM2.5 hours. Until October 2025 at most 12% of each month is missing, but **from November 2025 onwards
  31–97% of every month is missing** (December 2025: 97%), and the station rarely reports in the afternoon. HCMC is missing 8.4%.
  The Validation and Test periods of Hanoi fall entirely in this degraded period, which must be kept in mind when reading the results.
""",

"VAL": """
**Validation findings** (Section 3.2):
- Copying the current value (Persistence) is a weak forecast: R² ≈ 0.01 (Hanoi) and −0.13 (HCMC). The **Naive blend**
  (≈ half current value, half 24h mean; `w` = 0.45 / 0.50 learned on Train) is clearly better: MAE 13.01 vs 13.48 (Hanoi) and 7.06 vs 7.66 (HCMC).
  This is the bar a machine learning model has to clear.
- **Hanoi:** all three ML models beat the Naive blend; RandomForest is best (MAE 12.00, ≈ 8% lower than the blend), but Ridge is very close (12.10).
- **HCMC:** only Ridge beats the Naive blend, and only marginally (7.03 vs 7.06, ≈ 0.5%). The tree models (7.48–7.50) are **worse than the naive blend**:
  with a weak signal (R² ≤ 0.15), the strongly regularized linear model (alpha = 100) overfits less.
- GridSearch (with a 24h `gap` between CV folds) picks a small learning rate (0.03) for HGB and large leaves (`min_samples_leaf` = 30) for RandomForest,
  i.e. it prefers simpler models — consistent with noisy data covering only 22 months.
- The differences between the best ML models are small (Hanoi: RF 12.00 vs Ridge 12.10) and are measured on 1,462 (Hanoi) / 2,161 (HCMC) samples.
  Section 4.3 checks with block-bootstrap confidence intervals which of these differences are larger than chance.
""",

"CREATIVE": """
**Figure 6 caption:** On the independent Test set the best model of each city (chosen on Validation) reaches
**MAE 8.08 µg/m³ in Hanoi (RandomForest)** and **8.54 µg/m³ in HCMC (Ridge)**. Against the two reference forecasts:

| City | vs Persistence | vs Naive blend (strongest naive baseline) |
|---|---|---|
| Hanoi (RandomForest) | 9.66 → 8.08 (−16.4%) | 9.23 → 8.08 (**−12.5%**) |
| HCMC (Ridge) | 9.58 → 8.54 (−10.9%) | 9.05 → 8.54 (**−5.6%**) |

Comparing only with Persistence overstates the benefit of machine learning: part of the gain comes from simply smoothing the current value
with its 24h mean. The honest answer to question 1 is therefore **≈ 12% in Hanoi and ≈ 6% in HCMC** beyond the best naive forecast.
In HCMC the tree models are only 3.1–3.6% better than the blend on Test (and worse on Validation), so the linear Ridge model is the only one that adds a consistent gain.
Section 4.3 checks whether these gains are larger than chance: the Hanoi gain over the Naive blend is, the HCMC gain is not.

Note: only **measured** PM2.5 values are used as targets (no interpolated targets), and the features use a causal forward-fill only.
The two Test periods differ (Hanoi 26/02–29/09/2026, HCMC 30/06–29/09/2026, because Hanoi has more missing data), and Hanoi's Test set has
almost no afternoon hours (the station rarely reports 14h–19h after late 2025), so absolute MAE values should not be compared between cities —
compare the relative improvement instead. Hanoi's Test MAE is lower than its Validation MAE (8.1 vs 12.0) because the Test period contains only the end of winter;
HCMC is the opposite (8.5 vs 7.0) because dust rises again in September.
(Tree-model results can shift by a few hundredths of µg/m³ between scikit-learn versions; Ridge and the baselines do not change.)
""",

"SPLIT": """
**Figure caption (split):** Train / Validation / Test have 6,884 / 1,462 / 1,481 samples in Hanoi and 10,177 / 2,161 / 2,185 in HCMC.
The mean target drops from Train to Test (Hanoi 33.5 → 30.1 → 25.5 µg/m³, HCMC 34.5 → 26.9 → 25.9 µg/m³): the later sets fall in the
cleaner season, so the models are evaluated on a distribution they saw less of. Because Hanoi has more missing data, its Validation and Test
periods start earlier (Test 26/02/2026, vs 30/06/2026 for HCMC) — another reason to compare relative gains, not absolute MAE, between cities.
""",

"FORECAST": """
**Figure 8 caption:** In Hanoi the best model clearly tracks the actual series better than the Naive blend in this window
(MAE 10.08 vs 12.98 µg/m³ over 203 hours with data). In HCMC the two are practically tied (Ridge 6.96 vs blend 6.81 over 335 hours) —
consistent with the small overall gain in HCMC (Section 4.2). In both cities the forecast is much smoother than reality:
its standard deviation is about half of the actual one (Hanoi 8.8 vs 16.3, HCMC 5.3 vs 9.5 µg/m³), so short spikes are missed.
This is the visual counterpart of the under-forecasting of polluted hours in Section 4.5.
""",

"BOOTSTRAP": """
**Figure 7 caption — which differences are real?** (95% CI, 3-day blocks; *significant* = the interval excludes 0)

| City | Comparison | Validation | Test |
|---|---|---|---|
| Hanoi | RandomForest vs Persistence | +10.9% [−5.4, +23.2] | **+16.4% [+8.3, +23.0]** |
| Hanoi | RandomForest vs Naive blend | +7.8% [−9.2, +20.4] | **+12.5% [+3.2, +20.0]** |
| Hanoi | RandomForest vs Ridge | +0.8% [−9.0, +8.9] | +3.1% [−0.3, +6.3] |
| HCMC | Ridge vs Persistence | +8.2% [−0.8, +15.4] | **+10.9% [+4.4, +16.7]** |
| HCMC | Ridge vs Naive blend | +0.5% [−8.8, +9.0] | +5.6% [−1.3, +11.8] |
| HCMC | Ridge vs RandomForest | **+6.3% [+2.6, +10.1]** | +2.6% [−1.9, +6.8] |

- **Hanoi:** the gain of RandomForest over both naive baselines on Test is real (vs the Naive blend: +12.5%, CI +3.2% to +20.0%),
  and it stays positive for 1-, 3- and 7-day blocks. But RandomForest is **not** significantly better than Ridge or HGB (CIs include 0),
  and on Validation no difference is significant at all — the Validation set (1,462 samples, 29 blocks) is too small to separate
  the models, so picking RandomForest over Ridge (+0.8%) was close to a coin flip. *Machine learning helps in Hanoi; which ML model is best is not established.*
- **HCMC:** Ridge is significantly better than Persistence on Test (+10.9%), but its gain over the Naive blend (+5.6%) is **not** significant
  for any block length (the interval always reaches below 0). *We cannot claim that machine learning beats a well-chosen naive forecast in HCMC.*
  Ridge does beat both tree models significantly on Validation (≈ +6%), and in the same direction on Test (+2–3%, not significant):
  the evidence consistently says that **tree models do not help in HCMC**.
- **Method note:** an i.i.d. bootstrap over single hours would give much narrower intervals and declare almost everything significant;
  the blocks keep the strong hour-to-hour correlation of forecast errors, so the intervals are honest about how little independent evidence 3–7 months of data contain.
""",

"ERROR": """
**Error analysis findings:**
- **By hour:** Hanoi's error is fairly even across the hours that have data (6.8–9.3 µg/m³; hours 15h, 16h and 19h have fewer than 20 Test samples
  and are not shown). HCMC's error is highest during the daytime (peak 9.8 at 14h) and lowest at the 7h rush hour (7.4).
- **By month:** the error grows towards the end of the Test period in HCMC (7.1 in July → 10.9 in September 2026), as pollution starts rising again.
  The model learned mostly from the earlier distribution, so it reacts slowly to seasonal change. Hanoi varies between 6.4 (August) and 9.4 (March).
- **By EPA concentration band:** the correct band is predicted for 80.7% (Hanoi) and 76.3% (HCMC) of hours; 98.4% / 98.6% are within ±1 band.
  The main error is **under-forecasting polluted hours**: in HCMC, 286 of 359 "USG" hours are predicted as "Moderate"; in Hanoi 148 of 229 "USG" hours,
  and none of the 40 "Unhealthy" hours in Hanoi is predicted correctly. The model never predicts "Good", i.e. it pulls forecasts towards the mean
  (the expected behaviour of a model trained to minimize MAE).
- **Practical consequence:** the model is useful for following the general trend, but **not yet reliable as an early-warning tool for severe pollution episodes** —
  a case where a missed alarm costs more than a false alarm.
""",

"IMPORTANCE": """
**Findings:**
- **HCMC (Ridge):** the current PM2.5 level (`pm25_lag_0`, +0.98 µg/m³) and **temperature** (+0.86) matter most, followed by the **hour of day**
  (`hour_cos`) and **wind speed**. By group: PM2.5 history +1.81, weather +0.90, calendar +0.47 — HCMC's dust follows the daily traffic rhythm
  and how well the wind disperses it, consistent with the EDA (7h peak, strongest correlation with wind).
- **Hanoi (RandomForest):** single-feature importances are all small (at most about 0.1 µg/m³, led by wind speed and the current PM2.5),
  because the model spreads the PM2.5 information over many correlated lag/rolling features. The grouped importance shows the real picture:
  **PM2.5 history +0.78, weather +0.47, calendar ≈ 0**. The calendar (season) carries no signal on this Test period, which covers only the end of winter.
  Small single-feature values such as `day_of_week` should be read as noise.

The drivers differ between the cities (HCMC: daily cycle + temperature/wind; Hanoi: recent PM2.5 + weather), which is further evidence that each city should
have its own model.
""",

"CONCLUSION": """
**Conclusions:**
- We built a 24h-ahead PM2.5 forecasting pipeline, applied identically but **trained independently** for Hanoi and HCMC, using real
  monitoring data from OpenAQ combined with Open-Meteo weather (12/2024 – 09/2026). Features use only data available at forecast time
  (causal forward-fill), and models are evaluated only on measured values.
- On the Test set, the best model (chosen on Validation) lowers MAE by **12.5% in Hanoi (RandomForest)** and **5.6% in HCMC (Ridge)** compared with the
  strongest naive baseline (Naive blend), or by 16.4% / 10.9% compared with Persistence, and predicts the correct EPA concentration band in 76–81% of hours.
- Block-bootstrap confidence intervals show that the Hanoi gain over the Naive blend is real (95% CI +3.2% to +20.0%), while the HCMC gain
  over the Naive blend is not significant (−1.3% to +11.8%); in HCMC, tree models are significantly worse than Ridge on Validation.
  Which ML model is best in Hanoi is not established (RandomForest vs Ridge: CI includes 0).

**Limitations:**
- Only **one station** per city. The HCMC station is a low-cost sensor (AirGradient) that may be less accurate at high humidity.
- Hanoi has a lot of missing data (32.8%, and 31–97% per month from November 2025), so its Validation/Test sets are small, contain almost no afternoon hours,
  and the Test set covers only the end of winter.
- Only 22 months of data (fewer than 2 seasonal cycles), so seasonal patterns are hard to learn.
- The Validation sets are too small for the block bootstrap to separate the ML models, so the model selection itself is uncertain (Section 4.3).
- The models under-forecast high-pollution episodes. Weather features use observations at $t$, not weather forecasts for $t+24h$.

**Future work:**
- Add **weather forecasts** for the target time and data from more stations in each city.
- Use quantile regression / sample weights that emphasize high-pollution hours, or add an "exceeds threshold or not" classifier for early warning.
- Collect a longer history so that Validation can separate the models, and use the same calendar split for both cities.
- Try dedicated time series models (SARIMAX, multi-horizon LightGBM, LSTM) once longer data is available.
""",
}
