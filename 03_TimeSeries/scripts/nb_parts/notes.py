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
- **Data quality:** Hanoi is missing 32.8% of PM2.5 hours, concentrated from late 2025 (December 2025 is 97% missing); HCMC is missing 8.4%.
  This must be handled carefully in Section 2.
""",

"VAL": """
**Validation findings:**
- In both cities every machine learning model has a lower MAE than the Persistence baseline. The gain is large in Hanoi
  (14.97 → 12.45 with RandomForest, ≈17% lower) and small in HCMC (7.71 → 7.07 with Ridge, ≈8% lower).
- The baseline has a negative R² (-0.06 / -0.15): copying today's value as tomorrow's forecast is worse than predicting the mean.
- In Hanoi the three ML models are very close (MAE 12.45–12.61). In HCMC Ridge is best: with a weak signal (R² only 0.07–0.14),
  a strongly regularized linear model (alpha = 100) overfits less than the tree models.
- GridSearch picks a small learning rate (0.03) and few leaves (15) for HGB, i.e. it prefers simpler models. This is consistent with
  noisy data that covers only 22 months.
""",

"CREATIVE": """
**Figure 6 caption:** On the independent Test set, the best model of each city (chosen on Validation) reduces MAE relative to the Persistence baseline by
**13.1% in Hanoi** (RandomForest: 8.32 → 7.23 µg/m³) and by **10.6% in HCMC** (Ridge: 9.53 → 8.52 µg/m³).
In Hanoi, HistGradientBoosting is marginally lower on Test (7.20), but the 0.03 µg/m³ gap is within noise and choosing it would mean selecting on Test;
the honest conclusion is that the two tree ensembles perform the same (≈13% gain).
The best model type differs between the cities: Hanoi benefits from non-linear tree ensembles, which capture interactions
between season, rain and recent dust levels, while in HCMC the linear Ridge model is the most stable (the tree models gain only 7.6–8.7%).
So **there is no single best model for both cities**, which is why we train each city separately instead of merging the data.

Note: the two Test periods differ (Hanoi 19/03–29/09/2026, HCMC 29/06–29/09/2026, because Hanoi has more missing data)
and both fall in the cleaner season, so absolute MAE values should not be compared between cities — compare the relative improvement over the baseline instead.
Hanoi's Test MAE is lower than its Validation MAE (7.2 vs 12.5) because the Test period has no winter; HCMC is the opposite (8.5 vs 7.1) because dust rises again in September.
(Tree-model results can shift by a few hundredths of µg/m³ between scikit-learn versions; Ridge and the baseline do not change.)
""",

"ERROR": """
**Error analysis findings:**
- **By hour:** Hanoi's error is fairly even across the day (6.1–8.6 µg/m³). HCMC's error is highest during the daytime (peak 9.9 at 14h),
  when traffic and convection make PM2.5 fluctuate most.
- **By month:** the error grows towards the end of the Test period (HCMC: 3.2 in late June → 10.9 in September 2026), as pollution starts rising again.
  The model learned mostly from the earlier distribution, so it reacts slowly to seasonal change.
- **By AQI level:** the correct level is predicted for 82.2% (Hanoi) and 76.4% (HCMC) of hours; 99.1% / 98.6% are within ±1 level.
  The main error is **under-forecasting polluted hours**: in HCMC, 281 of 359 "USG" hours are predicted as "Moderate", and 29 of 71 "Unhealthy" hours
  are also predicted as "Moderate". The model never predicts "Good", i.e. it pulls forecasts towards the mean.
- **Practical consequence:** the model is useful for following the general trend, but **not yet reliable as an early-warning tool for severe pollution episodes** —
  a case where a missed alarm costs more than a false alarm.
""",

"IMPORTANCE": """
**Findings:** in both cities **the current PM2.5 level (`pm25_lag_0`) is the most important feature**, consistent with the high autocorrelation.
The cities differ in the next features:
- **Hanoi:** the recent PM2.5 history (6h mean, value 6h ago), **rain over the last 24h** (`rain_24h`) and the **season**. Hanoi's dust depends on the seasonal regime and on rain washing it out.
- **HCMC:** **temperature**, the **hour of day** (`hour_cos`) and **wind speed**. HCMC's dust follows the daily activity/traffic rhythm and how well the wind disperses it.

This matches the EDA (Hanoi has a strong seasonal pattern; HCMC has a rush-hour peak and a strong correlation with wind)
and is further evidence that each city should have its own model.
""",

"CONCLUSION": """
**Conclusions:**
- We built a 24h-ahead PM2.5 forecasting pipeline, applied identically but **trained independently** for Hanoi and HCMC, using real
  monitoring data from OpenAQ combined with Open-Meteo weather (12/2024 – 09/2026).
- On the Test set, the best model reduces MAE by **13.1% in Hanoi (RandomForest, chosen on Validation)** and **10.6% in HCMC (Ridge)**
  compared with the Persistence baseline, and predicts the correct AQI level in 76–82% of hours.
- The two cities have different pollution mechanisms: Hanoi is driven by season and rain, HCMC by the daily cycle and wind.

**Limitations:**
- Only **one station** per city. The HCMC station is a low-cost sensor (AirGradient) that may be less accurate at high humidity.
- Hanoi has a lot of missing data (32.8%, mostly from late 2025), so its Test set contains no winter — the most polluted season.
- Only 22 months of data (fewer than 2 seasonal cycles), so seasonal patterns are hard to learn.
- The models tend to under-forecast high-pollution episodes. Weather features use observations at $t$, not weather forecasts for $t+24h$.

**Future work:**
- Add **weather forecasts** for the target time (Open-Meteo forecast API) and data from more stations in each city.
- Use a loss function / sample weights that emphasize high-pollution hours, or add an "exceeds threshold or not" classifier for early warning.
- Try dedicated time series models (SARIMAX, multi-horizon LightGBM, LSTM) once longer data is available.
""",
}
