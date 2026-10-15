# Video Script — Part 3: Time Series (≈ 7 minutes)

Suggested YouTube title: **[Group name] – Time Series – Programming Foundations**
Upload as *Public* or *Unlisted*, then set `VIDEO_URL` in `report/build_report.py` and rebuild.

Show the executed notebook (`result/03_TimeSeries_Air_Quality_executed.ipynb`) or the Colab notebook while talking.
All numbers below come from the reference run and match the PDF report.

| Time | Screen | What to say |
|---|---|---|
| 0:00–0:40 | Title cell / web page | Group, member, topic: forecast PM2.5 **24 hours ahead** for Hanoi and HCMC. Two questions: how much does ML beat a naive baseline in each city, and what drives PM2.5 in each place? |
| 0:40–1:30 | Section 1.1, crawler script | Data is **self-collected** through APIs: OpenAQ (one station per city) + Open-Meteo weather, 01/12/2024 – 30/09/2026, 16,056 hours per city. Real gaps: Hanoi 32.8% missing, HCMC 8.4%. |
| 1:30–2:40 | Figures 1, 3, 4, 5 | EDA: similar mean (31.7 vs 32.1 µg/m³), above the WHO guideline in 56% / 82% of hours. Hanoi has a strong winter peak (Jan 55 vs Aug 19) and an evening peak; HCMC has a 7h rush-hour peak and depends most on wind (−0.36). Autocorrelation drops from 0.92 (1h) to 0.56 (24h) → day-ahead is hard. |
| 2:40–3:50 | Section 2 | Cleaning; interpolate only gaps ≤ 6h (32.8% → 26.0% missing); keep real peaks in the target, clip only features with the custom `IQRClipper`; 23 features (lags, rolling, weather, cyclic hour, season, wind sector); **chronological 70/15/15 split with a 24h gap**; everything in one `ColumnTransformer` + `Pipeline` fitted on Train only. |
| 3:50–4:40 | Section 3 | Persistence baseline vs Ridge, RandomForest, HistGradientBoosting; `GridSearchCV` with `TimeSeriesSplit`; the `CityForecaster` class runs the same flow for both cities. Validation: all ML models beat the baseline; the best model per city is **chosen on Validation** (Hanoi: RandomForest, HCMC: Ridge), so Test is only used for reporting. |
| 4:40–5:40 | Figure 6 + Test table | **Main result:** Test MAE −13.1% in Hanoi (RandomForest 8.32 → 7.23; HGB is tied at 7.20) and −10.6% in HCMC (Ridge 9.53 → 8.52). The best model type differs by city → train separately. Test periods differ, so compare relative gains. |
| 5:40–6:30 | Figure 7, AQI confusion matrix, permutation importance | Correct AQI level 82.2% / 76.4% (99% within ±1 level); main weakness: polluted hours are under-forecast (HCMC: 281 of 359 USG hours → Moderate). Drivers: Hanoi = recent PM2.5, rain over 24h and season; HCMC = temperature, hour of day, wind. |
| 6:30–7:00 | `predict_next` demo + Section 4.7 | Run the inference function; limitations (one station, no winter in Hanoi's Test, 22 months) and future work (weather forecasts, more stations, exceedance classifier). |

Tips: record at 1080p, zoom the notebook to ~125%, keep each figure on screen while explaining its conclusion.
