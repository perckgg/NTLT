# Observations written from real run results (data 2024-12-01 → 2026-09-30, random_state=42, scikit-learn 1.9.1).
NOTES = {
"EDA": """
- **Similar average pollution, different shapes:** mean PM2.5 is 31.7 µg/m³ in Hanoi and 32.1 µg/m³ in HCMC.
  Hanoi is more volatile (std 20.6 vs 16.3; max 332 vs 171). Both exceed the WHO 24h guideline (15 µg/m³)
  in most hours: 56% in Hanoi, 82% in HCMC.
- **Seasonality:** Hanoi has a clear winter pollution season (December: 53, January: 55 µg/m³) and is cleanest in August (19 µg/m³),
  a difference of almost 3×. HCMC is also most polluted at the end of the dry season / start of the year (January: 46 µg/m³) and cleanest in the middle of the wet season (July: 24 µg/m³), but its range is narrower.
- **Daily cycle:** HCMC peaks in the morning at 7h (≈41 µg/m³), matching the traffic rush hour; Hanoi peaks in the evening (≈34 µg/m³ at 22h),
  consistent with the night-time temperature inversion trapping dust near the ground.
- **Correlation with weather:** in Hanoi PM2.5 correlates negatively with temperature (-0.31) and cloud cover (-0.25) and positively with pressure (+0.31),
  i.e. cold, high-pressure winter air lets dust accumulate. In HCMC the strongest driver is wind speed (-0.36): strong wind disperses dust.
- **Autocorrelation:** the lag-1h autocorrelation is very high (0.92 / 0.88), but at lag 24h it drops to 0.56 / 0.45. Forecasting 24h ahead
  is therefore much harder than forecasting the next hour, and needs weather, season and daily-cycle features.
- **Data quality:** Hanoi is missing 32.8% of PM2.5 hours. Until October 2025 at most 12% of each month is missing, but **from November 2025 onwards
  31–97% of every month is missing** (December 2025: 97%), and the station rarely reports in the afternoon (≈14h–19h). HCMC is missing 8.4%.
  The Validation and Test periods of Hanoi fall entirely in this degraded period, which must be kept in mind when reading the results.
""",

"VAL": """
**Validation findings** (Section 3.2):
- Copying the current value (Persistence) is a weak forecast: R² ≈ 0.01 (Hanoi) and −0.13 (HCMC). The **Naive blend**
  (≈ half current value, half 24h mean; `w` = 0.45 / 0.50 learned on Train) is clearly better: MAE 13.01 vs 13.48 (Hanoi) and 7.06 vs 7.66 (HCMC).
  This is the bar a machine learning model has to clear.
- **Hanoi:** all three ML models beat the Naive blend; RandomForest is best (MAE 12.00, ≈ 8% lower than the blend), but Ridge is very close (12.10).
- **HCMC:** only Ridge beats the Naive blend, and only marginally (7.03 vs 7.06, ≈ 0.5%). The tree models (7.46–7.50) are **worse than the naive blend**:
  with a weak signal (R² ≤ 0.15), the strongly regularized linear model (alpha = 100) overfits less.
- GridSearch (with a 24h `gap` between CV folds) picks a small learning rate (0.03) for HGB and large leaves (`min_samples_leaf` = 30) for RandomForest,
  i.e. it prefers simpler models — consistent with noisy data covering only 22 months.
- The differences between the best ML models are small (Hanoi: RF 12.00 vs Ridge 12.10) and are measured on 1,462 (Hanoi) / 2,161 (HCMC) samples,
  so the choice of model type should not be over-interpreted.
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
The best model type differs between cities, which supports training a separate model for each city — although in Hanoi
the gap between RandomForest and the other two ML models (8.31–8.34) is small.

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
**Figure 7 caption:** In Hanoi the best model clearly tracks the actual series better than the Naive blend in this window
(MAE 10.08 vs 12.98 µg/m³ over 203 hours with data). In HCMC the two are practically tied (Ridge 6.96 vs blend 6.81 over 335 hours) —
consistent with the small overall gain in HCMC (Section 4.2). In both cities the forecast is much smoother than reality:
its standard deviation is about half of the actual one (Hanoi 8.8 vs 16.3, HCMC 5.3 vs 9.5 µg/m³), so short spikes are missed.
This is the visual counterpart of the under-forecasting of polluted hours in Section 4.4.
""",

"ERROR": """
**Error analysis findings:**
- **By hour:** Hanoi's error is fairly even across the hours that have data (6.8–9.3 µg/m³; hours 15h, 16h and 19h have fewer than 20 Test samples
  and are not shown). HCMC's error is highest during the daytime (peak 9.8 at 14h) and lowest at the 7h rush hour (7.4).
- **By month:** the error grows towards the end of the Test period in HCMC (7.1 in July → 10.9 in September 2026), as pollution starts rising again.
  The model learned mostly from the earlier distribution, so it reacts slowly to seasonal change. Hanoi varies between 6.4 (August) and 9.4 (March).
- **By AQI level:** the correct level is predicted for 80.8% (Hanoi) and 76.4% (HCMC) of hours; 98.5% / 98.6% are within ±1 level.
  The main error is **under-forecasting polluted hours**: in HCMC, 285 of 359 "USG" hours are predicted as "Moderate"; in Hanoi 147 of 229 "USG" hours,
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
- **Hanoi (RandomForest):** single-feature importances are all small (≤ 0.11 µg/m³, led by wind speed and the current PM2.5),
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
  strongest naive baseline (Naive blend), or by 16.4% / 10.9% compared with Persistence, and predicts the correct AQI level in 76–81% of hours.
- Machine learning adds a clear gain in Hanoi; in HCMC the gain over a well-chosen naive forecast is small, and only the linear model achieves it consistently.

**Limitations:**
- Only **one station** per city. The HCMC station is a low-cost sensor (AirGradient) that may be less accurate at high humidity.
- Hanoi has a lot of missing data (32.8%, and 31–97% per month from November 2025), so its Validation/Test sets are small, contain almost no afternoon hours,
  and the Test set covers only the end of winter.
- Only 22 months of data (fewer than 2 seasonal cycles), so seasonal patterns are hard to learn.
- Differences between the best models are small and no statistical test (e.g. block bootstrap, Diebold–Mariano) has been run yet.
- The models under-forecast high-pollution episodes. Weather features use observations at $t$, not weather forecasts for $t+24h$.

**Future work:**
- Add **weather forecasts** for the target time and data from more stations in each city.
- Use quantile regression / sample weights that emphasize high-pollution hours, or add an "exceeds threshold or not" classifier for early warning.
- Add confidence intervals (block bootstrap) for the MAE differences, and use the same calendar split for both cities.
- Try dedicated time series models (SARIMAX, multi-horizon LightGBM, LSTM) once longer data is available.
""",
}
