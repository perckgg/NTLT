# Part 3 — Time Series: 24-Hour-Ahead PM2.5 Forecasting — Hanoi vs Ho Chi Minh City

Course project for **Programming Foundations for Data Analysis and Visualization** (Semester 261, Instructor: Lê Thành Sách) — time series part.

| | |
|---|---|
| **Task** | Time series regression: at hour $t$, forecast PM2.5 at $t + 24h$ (same hour tomorrow) |
| **Data** | Self-collected via API: hourly PM2.5 (OpenAQ) + weather (Open-Meteo), 01/12/2024 – 30/09/2026, 16,056 hours per city |
| **Notebook** | [`notebooks/03_TimeSeries_Air_Quality.ipynb`](notebooks/03_TimeSeries_Air_Quality.ipynb) — runs end-to-end on Colab (Run all) |
| **Executed copy (with outputs)** | [`result/03_TimeSeries_Air_Quality_executed.ipynb`](result/03_TimeSeries_Air_Quality_executed.ipynb) |
| **Result (Test)** | Best model per city chosen on Validation, scored on measured values only. Hanoi — RandomForest: **8.08 µg/m³, −12.5% vs. the Naive blend baseline** (−16.4% vs. Persistence); HCMC — Ridge: **8.54 µg/m³, −5.6% vs. Naive blend** (−10.9% vs. Persistence); correct AQI level 80.8% / 76.4% |
| **PDF report** | [`report/03_TimeSeries_Report.pdf`](report/03_TimeSeries_Report.pdf) |
| **Web page (GitHub Pages)** | [`index.html`](index.html) |

## 1. Approach

Hanoi (four seasons, winter temperature inversion) and Ho Chi Minh City (wet/dry seasons) have different pollution mechanisms.
Instead of merging the two cities, we apply **the same pipeline** to each city **independently** and compare the results to answer:

1. **How much better than a naive baseline can machine learning forecast PM2.5 one day ahead in each city?**
2. **Which city is harder to forecast, and what drives PM2.5 in each place?**

Three naive baselines set the bar: **Persistence** `ŷ(t+24) = pm25(t)` (equivalent to a 24h seasonal-naive forecast), the **24h mean**, and a
**Naive blend** `w·pm25(t) + (1−w)·mean_24h(t)` with `w` learned on Train. A model is only useful if it beats the strongest one (the Naive blend) —
comparing only with Persistence overstates the gain.

## 2. Method (following the 4 required sections)

| Section | Main content |
|---|---|
| **1. Data description & EDA** | Data sources and column table; descriptive statistics, missing rate, % of hours above the WHO 24h guideline (15 µg/m³); daily series, hourly distribution, monthly boxplots, diurnal cycle by half-year, Spearman correlation with weather |
| **2. Data preparation** | Missing report (gap lengths); cleaning invalid (≤ 0, > 1000) and "stuck sensor" values; **causal forward-fill (≤ 6h) for features only, targets are always measured values**; IQR outlier detection; feature engineering (lags, rolling stats, weather, cyclic hour, season, wind sector); **chronological 70/15/15 split with a 24h gap**; `ColumnTransformer` pipeline with a custom `IQRClipper` |
| **3. Modeling** | Three naive baselines (Persistence, 24h mean, Naive blend), Ridge, RandomForest, HistGradientBoosting; `GridSearchCV` with `TimeSeriesSplit(3, gap=24)` (MAE) for all three ML models; comparison on Validation, which also **selects the best model per city** (Test is used only for reporting); a `CityForecaster` class runs the whole flow for one city |
| **4. Summary** | Retrain on Train + Validation, evaluate **once** on Test; MAE comparison figure; forecast vs. actual; error analysis by hour, month and **AQI-level confusion matrix**; single and **grouped** permutation importance; inference function `predict_next`; conclusions, limitations, future work |

### Key decisions and evidence

| Decision | Reason |
|---|---|
| Model each city separately | Different seasonality (Hanoi winter peak ~3× August; HCMC narrower range), different daily peaks (Hanoi 22h, HCMC 7h) and different weather drivers |
| No interpolation: causal forward-fill (≤ 6h) for features, measured values only for targets | Time interpolation fills a gap with the value measured *after* it — unknown at forecast time (it would leak up to 6h of the future into ~22% of Hanoi's Test features) — and interpolated targets are easier to predict, which would flatter the Test score. Long gaps are not filled (Hanoi is 32.8% missing, 31–97% per month from 11/2025) |
| Keep PM2.5 peaks in the target, clip only the features (`IQRClipper`) | Peaks are real pollution events; removing them would make evaluation look better than it is |
| Chronological split + 24h gap, `TimeSeriesSplit(gap=24)` for tuning | Shuffling would leak the future into the past; the gap keeps a set's (and a CV fold's) targets out of the next one |
| Calendar features computed for the **target** time ($t+24h$) | The hour/season of the forecast moment is known in advance, so this does not leak |
| MAE as the main metric | Easy to interpret in µg/m³ and robust to the heavy right tail |

## 3. What is special (creativity)

1. **Two-city comparison with one shared pipeline, against an honest baseline.** The Test figure shows each model's MAE per city, labeled with the % reduction vs. the Naive blend.
   Against Persistence ML looks like −16% / −11%; against the Naive blend the real gain is **−12.5% in Hanoi (RandomForest) and only −5.6% in HCMC (Ridge)**.
   In HCMC the tree models are worse than the blend on Validation and only 3–4% better on Test → there is **no single best model for both cities**.
2. **Error analysis translated into AQI levels.** Forecasts are mapped to US EPA PM2.5 levels to build a confusion matrix:
   80.8% / 76.4% correct level, 98.5% / 98.6% within ±1 level. The main weakness is **under-forecasting polluted hours**
   (HCMC: 285 of 359 "Unhealthy for Sensitive Groups" hours predicted as "Moderate"; Hanoi: none of 40 "Unhealthy" hours correct), so the model is not yet reliable as an early-warning tool.
3. **Feature importance, single and grouped.** Correlated lag features share their importance, so groups are also shuffled together.
   Recent PM2.5 is the most important group in both cities; HCMC then depends on **temperature, hour of day and wind speed**,
   Hanoi on **weather**, with the calendar adding almost nothing on its (end-of-winter) Test period.
4. **Real, self-collected data with real gaps** (not a ready-made Kaggle file), with a reproducible crawler and an automatic fallback source.
5. **Reproducible:** the notebook is generated by a script; every observation is written from real run results (`random_state = 42`).

> Note: the two Test periods differ (Hanoi 26/02–29/09/2026, HCMC 30/06–29/09/2026, because Hanoi has more missing data), and Hanoi's Test set has almost
> no afternoon hours (the station rarely reports 14h–19h after late 2025), so we compare the **relative improvement over the baseline**, not absolute MAE between cities.

## 4. Data sources

| Component | Source | Notes |
|---|---|---|
| PM2.5 – Hanoi | [OpenAQ v3](https://openaq.org) – station *46 Luu Quang Vu* (Hanoi Air Quality Monitoring Network, sensor 7772032) | City monitoring station, has real gaps (32.8% missing) |
| PM2.5 – HCMC | OpenAQ v3 – station *CMT8* (AirGradient, sensor 11357424) | Low-cost sensor (8.4% missing) |
| Weather | [Open-Meteo Historical Weather API](https://open-meteo.com) | Temperature, humidity, precipitation, wind speed/direction, pressure, cloud cover |

The crawler builds a full hourly grid (Asia/Ho_Chi_Minh time); hours with no measurement become `NaN`.
If no OpenAQ key is available, it falls back to the Open-Meteo Air Quality API (CAMS model data) and records this in the `pm25_source` column.

## 5. Folder structure

```
03_TimeSeries/
├── README.md
├── index.html                           # GitHub Pages sub-page for this part (generated)
├── VIDEO_SCRIPT.md                      # outline for the 5–10 minute presentation video
├── data/timeseries/
│   ├── air_quality_hanoi.csv            # hourly PM2.5 + weather, Hanoi
│   └── air_quality_hcmc.csv             # hourly PM2.5 + weather, HCMC
├── notebooks/
│   └── 03_TimeSeries_Air_Quality.ipynb  # submission notebook (no outputs)
├── result/
│   └── 03_TimeSeries_Air_Quality_executed.ipynb   # executed copy with all outputs
├── report/
│   ├── build_report.py                  # builds the PDF report + index.html from the executed notebook
│   ├── 03_TimeSeries_Report.pdf         # PDF report (generated)
│   └── figures/                         # figures exported from the notebook (generated)
└── scripts/
    ├── crawl_air_quality.py             # collects data from OpenAQ + Open-Meteo
    ├── generate_timeseries_notebook.py  # generates the notebook from the parts below
    └── nb_parts/
        ├── part1_overview.py            # Sections 0–1: introduction, data description, EDA
        ├── part2_preparation.py         # Section 2: cleaning, missing, outliers, features, split, pipeline
        ├── part3_modeling.py            # Section 3: models, tuning, validation
        ├── part4_summary.py             # Section 4: test, creativity, error analysis, importance, inference
        └── notes.py                     # observations (written from real results)
```

## 6. How to run

**On Google Colab:** open the notebook and choose *Runtime → Run all*. The notebook reads the CSV files from local folders first;
if they are not found, it downloads them from `GITHUB_RAW_BASE` (set it to the group's repo).

**Locally (Windows), from the `03_TimeSeries` folder:**
```powershell
$env:OPENAQ_API_KEY = "<your-key>"           # free key from openaq.org; never commit it
py scripts\crawl_air_quality.py              # (optional) re-download the data
py scripts\generate_timeseries_notebook.py   # regenerate the notebook after editing nb_parts/
jupyter nbconvert --to notebook --execute --output-dir result --output 03_TimeSeries_Air_Quality_executed.ipynb notebooks\03_TimeSeries_Air_Quality.ipynb
py report\build_report.py                    # rebuild the PDF report and index.html (needs weasyprint)
```
Every number in the PDF report and the web page is read from the executed notebook, so they always match it.
Fill in `GROUP_NAME`, `MEMBERS` and `VIDEO_URL` at the top of `report/build_report.py` before the final build.
Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (≥ 1.2). A full run takes about 1–2 minutes.
Re-crawling may change the numbers slightly (OpenAQ can backfill data); in that case, re-run the notebook and update `notes.py`.
Tree-model results (RandomForest, HGB) can shift by a few hundredths of µg/m³ between scikit-learn versions; Ridge and the naive baselines do not change.
The reference run used scikit-learn 1.9.1 (printed in the first code cell).

## 7. Limitations and future work

- One station per city; the HCMC station is a low-cost sensor that may be less accurate at high humidity.
- Only 22 months of data (fewer than 2 seasonal cycles); Hanoi's Validation/Test periods fall in the degraded-data period (from 11/2025): small, no afternoon hours, only the end of winter.
- Differences between the best models are small and not yet tested statistically (block bootstrap / Diebold–Mariano).
- Weather features use observations at $t$, not weather forecasts for $t+24h$.
- Future work: add weather forecasts and more stations, quantile regression or an exceedance classifier for early warning, a shared calendar split for both cities,
  and try dedicated time series models (SARIMAX, multi-horizon LightGBM, LSTM) with longer data.

## 8. Before submitting

- [ ] Fill in the group name and member in the first notebook cell (`scripts/nb_parts/part1_overview.py`) and in `report/build_report.py`.
- [x] Set `GITHUB_RAW_BASE` to the group's repo (`perckgg/NTLT`, branch `main`) and push the `data/timeseries/` CSV files.
- [ ] Merge this work into `main`, then check that *Run all* works from the "Open in Google Colab" link.
- [x] Executed copy with outputs saved in `result/`.
- [x] PDF report (`report/03_TimeSeries_Report.pdf`), generated from the executed notebook.
- [x] Sub-page for GitHub Pages (`index.html`) with links to the notebook, Colab, PDF and video.
- [ ] Record the video (outline in `VIDEO_SCRIPT.md`), upload it to YouTube as Public/Unlisted, set `VIDEO_URL` and rebuild.
- [ ] Link this page from the group landing page and enable GitHub Pages for the repository.
