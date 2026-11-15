# Video Script — Part 3: Time Series (≈ 7 minutes)

Suggested YouTube title: **IndexError – Time Series – Programming Foundations**
Upload as *Public* or *Unlisted*, then set `VIDEO_URL` in `report/build_report.py` and rebuild.

Show the executed notebook (`result/03_TimeSeries_Air_Quality_executed.ipynb`) or the Colab notebook while talking.
All numbers below come from the reference run and match the PDF report.

| Time | Screen | What to say |
|---|---|---|
| 0:00–0:40 | Title cell / web page | Group, member, topic: forecast PM2.5 **24 hours ahead** for Hanoi and HCMC. Two questions: how much does ML beat a naive baseline in each city, and what drives PM2.5 in each place? |
| 0:40–1:30 | Section 1.1, crawler script | Data is **self-collected** through APIs: OpenAQ (one station per city) + Open-Meteo weather, 01/12/2024 – 30/09/2026, 16,056 hours per city. Real gaps: Hanoi 32.8% missing, HCMC 8.4%. |
| 1:30–2:40 | Figures 1, 3, 4, 5 | EDA: similar mean (31.7 vs 32.1 µg/m³), 24-hour mean above the WHO guideline on 91.3% / 98.2% of valid days (WHO is a daily guideline, so compare daily means). Hanoi has a strong winter peak (monthly mean Jan 55.0 vs Aug 19.4) and, on complete days, a night-time peak (0h in Nov–Apr); HCMC has a 7h rush-hour peak and depends most on wind (Spearman −0.38). Autocorrelation drops from 0.92 (1h) to 0.56 (24h) → day-ahead is hard. |
| 2:40–3:50 | Section 2 | Cleaning; **no look-ahead**: features use a causal forward-fill ≤ 6h (32.8% → 26.0% missing), targets are always measured values (interpolation would leak the future and flatter the score); keep real peaks in the target, clip only features with the custom `IQRClipper`; 23 features (lags, rolling, weather, cyclic hour, season, wind sector); **chronological 70/15/15 split with a 24h gap**; everything in one `ColumnTransformer` + `Pipeline` fitted on Train only. |
| 3:50–4:40 | Section 3 | Three naive baselines (Persistence, 24h mean, **Naive blend** with `w` learned on Train) vs Ridge, RandomForest, HistGradientBoosting; `GridSearchCV` with `TimeSeriesSplit(gap=24)`; the `CityForecaster` class runs the same flow for both cities. Validation: the blend is much better than Persistence; in HCMC only Ridge beats it. Best model per city is **chosen on Validation** (Hanoi: RandomForest, HCMC: Ridge), so Test is only used for reporting. |
| 4:40–5:40 | Figure 6 + Test table | **Main result:** vs Persistence ML looks like −16.4% (Hanoi) / −10.9% (HCMC), but vs the Naive blend the honest gain is **−12.5% in Hanoi** (RandomForest 9.23 → 8.08) and **−5.6% in HCMC** (Ridge 9.05 → 8.54). **Forest plot (block bootstrap, 3-day blocks):** the Hanoi gain over the blend is significant (CI +3.2% to +20.0%), the HCMC one is not (−1.3% to +11.8%); Ridge > trees in HCMC is significant on Validation, Hanoi's ML models cannot be separated. Test periods differ, so compare relative gains. |
| 5:40–6:30 | Figure 8, EPA-band confusion matrix, permutation importance | Correct EPA band 80.7% / 76.3% (≈ 98.5% within ±1 band; hourly bands, not official AQI); main weakness: polluted hours are under-forecast (HCMC: 286 of 359 USG hours → Moderate). Drivers (grouped importance): recent PM2.5 first in both cities; HCMC = temperature, hour of day, wind; Hanoi = weather, calendar ≈ 0 on its end-of-winter Test period. |
| 6:30–7:00 | `predict_next` demo + Section 4.7 | Run the inference function; limitations (one station, Hanoi's patchy data from 11/2025, 22 months, Validation too small to separate the models) and future work (weather forecasts, more stations, quantile regression / exceedance classifier). |

Tips: record at 1080p, zoom the notebook to ~125%, keep each figure on screen while explaining its conclusion.
