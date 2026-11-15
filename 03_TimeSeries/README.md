# Part 3 — Time Series: 24-Hour-Ahead PM2.5 Forecasting — Hanoi vs Ho Chi Minh City

Course project for **Programming Foundations for Data Analysis and Visualization** (Semester 261, Instructor: Lê Thành Sách) — time series part.

| | |
|---|---|
| **Task** | Time series regression: at hour $t$, forecast PM2.5 at $t + 24h$ (same hour tomorrow) |
| **Data** | Self-collected via API: hourly PM2.5 (OpenAQ) + weather (Open-Meteo), 04/07/2025 – 14/11/2026 for both cities, 11,976 hours per city |
| **Notebook** | [`notebooks/03_TimeSeries_Air_Quality.ipynb`](notebooks/03_TimeSeries_Air_Quality.ipynb) — runs end-to-end on Colab (Run all) |
| **Executed copy (with outputs)** | [`result/03_TimeSeries_Air_Quality_executed.ipynb`](result/03_TimeSeries_Air_Quality_executed.ipynb) |
| **Result (Test, 01/09 – 13/11/2026)** | Model per city chosen by monthly walk-forward validation, scored on measured values only. Hanoi — RandomForest: **10.91 µg/m³, −5.3% vs. the Naive blend baseline** (95% block-bootstrap CI −3.4% to +13.3%, **not significant**; −9.6% vs. Persistence, significant); HCMC — HistGradientBoosting: **11.07 µg/m³, −7.2% vs. Naive blend** (CI +1.7% to +12.6%, significant; −14.1% vs. Persistence, significant); correct EPA concentration band 59.3% / 60.5% |
| **PDF report** | [`report/03_TimeSeries_Report.pdf`](report/03_TimeSeries_Report.pdf) |
| **Web page (GitHub Pages)** | [`index.html`](index.html) |

## 1. Approach

Hanoi (four seasons, winter temperature inversion) and Ho Chi Minh City (wet/dry seasons) have different pollution mechanisms.
Instead of merging the two cities, we apply **the same pipeline** to each city **independently** and compare the results to answer:

1. **How much better than a naive baseline can machine learning forecast PM2.5 one day ahead in each city?**
2. **Which city is harder to forecast, and what drives PM2.5 in each place?**

Three naive baselines set the bar: **Persistence** `ŷ(t+24) = pm25(t)` (equivalent to a 24h seasonal-naive forecast), the **24h mean**, and a
**Naive blend** `w·pm25(t) + (1−w)·mean_24h(t)` with `w` learned on the training data. A model is only useful if it beats the strongest one
(the Naive blend) — comparing only with Persistence overstates the gain.

## 2. Method (following the 4 required sections)

| Section | Main content |
|---|---|
| **1. Data description & EDA** | Data sources, **station choice by coverage** (table of every candidate OpenAQ station), column table; descriptive statistics, missing rate, % of **days** whose 24-hour mean exceeds the WHO guideline (15 µg/m³; days with ≥ 18 measured hours); daily series, hourly distribution, monthly boxplots, diurnal cycle by half-year (complete days only), autocorrelation table, Spearman correlation with weather |
| **2. Data preparation** | Missing report (gap lengths, by month); cleaning invalid (≤ 0, > 1000) and "stuck sensor" values; **causal forward-fill (≤ 6h) for features only, targets are always measured values**; IQR outlier detection; feature engineering (lags, rolling stats, weather, cyclic hour, season, wind sector); **Development / Test split on the same calendar dates for both cities (24h gap)** and **11 monthly walk-forward folds**; `ColumnTransformer` pipeline with a custom `IQRClipper` |
| **3. Modeling** | Three naive baselines, Ridge, RandomForest, HistGradientBoosting; `GridSearchCV` with `TimeSeriesSplit(3, gap=24)` (MAE) on Development, with selected values at the grid edge flagged; **walk-forward validation** (10/2025 – 08/2026, expanding window, every model refitted each month) **selects the best model per city**; MAE and bias by month; a `CityForecaster` class runs the whole flow for one city |
| **4. Summary** | Refit on Development, evaluate **once** on Test; MAE comparison figure; **paired block-bootstrap confidence intervals** (3-day blocks, checked with 1 and 7 days) on the walk-forward forecasts and on Test, as a forest plot; forecast vs. actual; error by hour and **EPA-band confusion matrix**; single and **grouped** permutation importance; inference function `predict_next`; conclusions, limitations, future work |

### Key decisions and evidence

| Decision | Reason |
|---|---|
| Hanoi station *Nhân Chính* instead of *46 Lưu Quang Vũ* | Over the study period Nhân Chính measured 96.4% of hours (95.0% for its worst hour of the day); Lưu Quang Vũ measured 51.0% and only 27.9% of its worst hour, i.e. its gaps fell mostly on certain hours (not at random) and would bias the daily cycle and the errors (`scripts/check_station_coverage.py`) |
| Same calendar period and split dates for both cities | The two cities are evaluated over the same weather and season, so their results can be compared |
| Walk-forward validation instead of one Validation block | With one winter in the data, a single Validation block falls in one season; 11 monthly folds forecast every month from 10/2025 to 08/2026 exactly once, including the whole winter |
| Model each city separately | Different seasonality, different daily cycles (Hanoi dry-season peak at night, HCMC 7h rush hour) and different weather drivers |
| No interpolation: causal forward-fill (≤ 6h) for features, measured values only for targets | Time interpolation fills a gap with the value measured *after* it — unknown at forecast time — and interpolated targets are easier to predict, which would flatter the score. Long gaps are not filled |
| Keep PM2.5 peaks in the target, clip only the features (`IQRClipper`) | Peaks are real pollution events; removing them would make evaluation look better than it is |
| 24h gap before every boundary, `TimeSeriesSplit(gap=24)` for tuning | The target of a training sample is 24h in the future; the gap keeps it out of the next period |
| Calendar features computed for the **target** time ($t+24h$) | The hour/season of the forecast moment is known in advance, so this does not leak |
| MAE as the main metric | Easy to interpret in µg/m³ and robust to the heavy right tail |

## 3. What is special (creativity)

1. **Two-city comparison with one shared pipeline, against an honest baseline.** Against Persistence ML gains 9.6% / 14.1% on Test; against the Naive blend
   only **5.3% in Hanoi (RandomForest) and 7.2% in HCMC (HistGradientBoosting)**. On Test, Ridge is actually the best model in Hanoi (10.55) although it was not selected.
2. **Walk-forward MAE and bias by month.** The ML gain is concentrated in the winter peak (Hanoi January: RandomForest 15.54 vs Naive blend 19.36 µg/m³),
   while in months where the pollution level changes the ML models lag behind and lose to the blend (Hanoi November 2025: 17.54 vs 14.96, bias −7.8 µg/m³).
3. **Are the gains real? Block-bootstrap confidence intervals.** Forecast errors are autocorrelated, so whole 3-day blocks are resampled (2,000×).
   The gain over Persistence is significant in both cities on both sets; the gain over the Naive blend is significant only for HCMC on Test
   (+7.2%, CI +1.7% to +12.6%); Hanoi is borderline on walk-forward (+6.8%, CI −0.9% to +13.7%) and not significant on Test. The ML models cannot be told apart.
4. **Error analysis translated into EPA concentration bands.** Actual and forecast hourly PM2.5 are put into the US EPA 2024 breakpoint bands
   (not the official AQI, which uses 24-hour averages): 59.3% / 60.5% correct band on Test, 96.7% / 96.8% within ±1 band. The main weakness is
   **under-forecasting polluted hours** (Hanoi: 46 of 215 "Unhealthy" hours correct; HCMC: 2 of 136), so the model is not reliable as an early-warning tool.
5. **Feature importance, single and grouped.** Recent PM2.5 is the most important group in both cities; Hanoi then depends on weather, HCMC on the hour of day.
6. **Real, self-collected data** with a reproducible crawler, a station-coverage check and an automatic fallback source.
7. **Reproducible:** the notebook is generated by a script; every observation is written from real run results (`random_state = 42`), and the PDF report and web page read every number from the executed notebook.

## 4. Data sources

| Component | Source | Notes |
|---|---|---|
| PM2.5 – Hanoi | [OpenAQ v3](https://openaq.org) – station *Công viên Nhân Chính – Khuất Duy Tiến* (Hanoi Air Quality Monitoring Network, sensor 13502151) | Reference monitor, 3.6% of hours missing |
| PM2.5 – HCMC | OpenAQ v3 – station *CMT8* (AirGradient, sensor 11357424) | Low-cost sensor, 0.6% missing; the only HCMC station with continuous data |
| Weather | [Open-Meteo Historical Weather API](https://open-meteo.com) | Temperature, humidity, precipitation, wind speed/direction, pressure, cloud cover, at each station's coordinates |

The crawler builds a full hourly grid (Asia/Ho_Chi_Minh time; each timestamp is the start of the hour); hours with no measurement become `NaN`.
If no OpenAQ key is available, it falls back to the Open-Meteo Air Quality API (CAMS model data) and records this in the `pm25_source` column.
`scripts/check_station_coverage.py` compares every OpenAQ PM2.5 sensor within 25 km of each city over the same period and saves `data/timeseries/station_coverage.csv`.

## 5. Folder structure

```
03_TimeSeries/
├── README.md
├── index.html                           # GitHub Pages sub-page for this part (generated)
├── VIDEO_SCRIPT.md                      # outline for the 5–10 minute presentation video
├── GIAI_THICH.md                        # Vietnamese walkthrough of every step, formula and decision (for the defence)
├── data/timeseries/
│   ├── air_quality_hanoi.csv            # hourly PM2.5 + weather, Hanoi
│   ├── air_quality_hcmc.csv             # hourly PM2.5 + weather, HCMC
│   └── station_coverage.csv             # coverage of every candidate station (station choice)
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
    ├── check_station_coverage.py        # compares candidate stations by coverage
    ├── generate_timeseries_notebook.py  # generates the notebook from the parts below
    └── nb_parts/
        ├── part1_overview.py            # Sections 0–1: introduction, data description, EDA
        ├── part2_preparation.py         # Section 2: cleaning, missing, outliers, features, split, pipeline
        ├── part3_modeling.py            # Section 3: models, tuning, walk-forward validation
        ├── part4_summary.py             # Section 4: test, bootstrap, error analysis, importance, inference
        └── notes.py                     # observations (written from real results)
```

## 6. How to run

**On Google Colab:** open the notebook and choose *Runtime → Run all*. The notebook reads the CSV files from local folders first;
if they are not found, it downloads them from `GITHUB_RAW_BASE` (the group's repo, branch `main`).

**Locally (Windows), from the `03_TimeSeries` folder:**
```powershell
$env:OPENAQ_API_KEY = "<your-key>"           # free key from openaq.org; never commit it
py scripts\check_station_coverage.py         # (optional) re-check the candidate stations
py scripts\crawl_air_quality.py              # (optional) re-download the data
py scripts\generate_timeseries_notebook.py   # regenerate the notebook after editing nb_parts/
jupyter nbconvert --to notebook --execute --output-dir result --output 03_TimeSeries_Air_Quality_executed.ipynb notebooks\03_TimeSeries_Air_Quality.ipynb
py report\build_report.py                    # rebuild the PDF report and index.html (needs weasyprint)
```
Every number in the PDF report and the web page is read from the executed notebook, so they always match it.
`GROUP_NAME` and `MEMBERS` are filled in at the top of `report/build_report.py`; set `VIDEO_URL` there once the video is uploaded.
Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (≥ 1.2). A full run takes about 2–3 minutes.
Re-crawling may change the numbers slightly (OpenAQ can backfill data); in that case, re-run the notebook and update `notes.py`.
Tree-model results (RandomForest, HGB) can shift by a few hundredths of µg/m³ between scikit-learn versions; Ridge and the naive baselines do not change.
The reference run used scikit-learn 1.9.1 (printed in the first code cell).

## 7. Limitations and future work

- **One winter only** (the Hanoi station started reporting in 07/2025): winter performance rests on the walk-forward months, and the Test period
  (09–11/2026) is the start of the polluted season, not its peak.
- One station per city; the HCMC station is a low-cost sensor that may be less accurate at high humidity.
- Hyperparameters are tuned on the whole Development set, so walk-forward scores of the ML models are slightly optimistic; several selected values lie at the grid edge.
- Weather features use observations at $t$, not weather forecasts for $t+24h$. Polluted hours are under-forecast.
- Future work: repeat the walk-forward evaluation after a second winter; weather forecasts and more stations; features that react to level changes;
  quantile regression or an exceedance classifier for early warning; nested tuning inside each walk-forward fold.

## 8. Before submitting

- [x] Group name (IndexError) and member filled in the first notebook cell and in `report/build_report.py`.
- [x] Set `GITHUB_RAW_BASE` to the group's repo (`perckgg/NTLT`, branch `main`) and push the `data/timeseries/` CSV files.
- [ ] Merge this work into `main`, then check that *Run all* works from the "Open in Google Colab" link.
- [x] Executed copy with outputs saved in `result/`.
- [x] PDF report (`report/03_TimeSeries_Report.pdf`), generated from the executed notebook.
- [x] Sub-page for GitHub Pages (`index.html`) with links to the notebook, Colab, PDF and video.
- [ ] Record the video (outline in `VIDEO_SCRIPT.md`), upload it to YouTube as Public/Unlisted, set `VIDEO_URL` and rebuild.
- [ ] Link this page from the group landing page and enable GitHub Pages for the repository.
