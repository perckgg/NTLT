"""
Build the Part 3 (Time Series) deliverables from the executed notebook:
  - report/03_TimeSeries_Report.pdf   (PDF report for this part)
  - index.html                        (GitHub Pages sub-page for this part)
  - report/figures/*.png              (figures used by both)

Every metric is read from result/03_TimeSeries_Air_Quality_executed.ipynb, so the report
always matches the notebook. Re-run after re-executing the notebook.

Run from the 03_TimeSeries folder:  py report/build_report.py
Requires: pandas, weasyprint (for the PDF).
"""

import ast
import base64
import html
import json
import os
import re
from io import StringIO

import pandas as pd

# ---------------------------------------------------------------- group info (fill in before submitting)
GROUP_NAME = "[Group name]"          # must match the GroupRegistration sheet
MEMBERS = [                          # (full name, student ID, role in this part)
    ("[Full name]", "[Student ID]", "Data collection, notebook, report"),
]
VIDEO_URL = None                     # YouTube link (Public/Unlisted) once uploaded

# ---------------------------------------------------------------- paths and links
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXECUTED_NB = os.path.join(ROOT, "result", "03_TimeSeries_Air_Quality_executed.ipynb")
FIG_DIR = os.path.join(ROOT, "report", "figures")
PDF_PATH = os.path.join(ROOT, "report", "03_TimeSeries_Report.pdf")
PAGE_PATH = os.path.join(ROOT, "index.html")

REPO_URL = "https://github.com/perckgg/NTLT"
NB_PATH = "03_TimeSeries/notebooks/03_TimeSeries_Air_Quality.ipynb"
LINKS = {
    "repo": REPO_URL,
    "notebook": f"{REPO_URL}/blob/main/{NB_PATH}",
    "colab": f"https://colab.research.google.com/github/perckgg/NTLT/blob/main/{NB_PATH}",
    "executed": f"{REPO_URL}/blob/main/03_TimeSeries/result/03_TimeSeries_Air_Quality_executed.ipynb",
    "data": f"{REPO_URL}/tree/main/03_TimeSeries/data/timeseries",
    "pdf": "report/03_TimeSeries_Report.pdf",
    "landing": "../",
}

# Figures to export: name -> first line of the code cell that draws it
FIGURES = {
    "daily": "fig, ax = plt.subplots(figsize=(14, 5))",
    "cycle": "plot_df['half_year']",
    "split": "def time_split(",
    "mae": "MODEL_ORDER = list(make_models())",
    "forecast": "fig, axes = plt.subplots(2, 1, figsize=(14, 8))",
    "aqi": "AQI_BINS = ",
    "importance": "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\nfor ax, (city, f) in zip(axes, forecasters.items()):\n    model",
}
CITIES = ["Hanoi", "HCMC"]


# ---------------------------------------------------------------- read results from the notebook
def load_results():
    cells = json.load(open(EXECUTED_NB, encoding="utf-8"))["cells"]
    texts, htmls = [], []
    os.makedirs(FIG_DIR, exist_ok=True)
    found = set()
    for c in cells:
        src = "".join(c["source"])
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                raise RuntimeError("The executed notebook contains an error; re-run it first.")
            data = o.get("data", {})
            texts.append("".join(o.get("text") or data.get("text/plain", "")))
            htmls.append("".join(data.get("text/html", "")))
            if "image/png" in data:
                for name, start in FIGURES.items():
                    if src.startswith(start) and name not in found:
                        with open(os.path.join(FIG_DIR, f"{name}.png"), "wb") as fh:
                            fh.write(base64.b64decode(data["image/png"]))
                        found.add(name)
    missing = set(FIGURES) - found
    if missing:
        raise RuntimeError(f"Figures not found in the notebook: {missing}")
    all_text = "\n".join(texts)

    test_html = next(h for h in htmls if "MAE vs Naive blend" in h)
    test = pd.read_html(StringIO(test_html))[0]
    test.columns = ["city", "model", "MAE", "RMSE", "R2", "reduction", "vs_blend"]
    test["city"] = test["city"].ffill()
    for col in ["reduction", "vs_blend"]:
        test[col] = test[col].str.rstrip("%").astype(float)

    val_text = next(t for t in texts if t.lstrip().startswith("MAE") and "Persistence (baseline)" in t)
    val_rows = []
    for line in val_text.splitlines():
        m = re.match(r"^((?:Persistence|Mean 24h|Naive blend) \(baseline\)|Ridge|RandomForest|HistGradientBoosting)\s+(.*)$", line)
        if m:
            nums = [float(x) for x in m.group(2).split()]
            val_rows.append({"model": m.group(1), "MAE Hanoi": nums[0], "MAE HCMC": nums[1],
                             "R2 Hanoi": nums[4], "R2 HCMC": nums[5]})
    val = pd.DataFrame(val_rows)

    aqi = {c: (float(a), float(b)) for c, a, b in
           re.findall(r"(\w+): correct AQI level ([\d.]+)% \| within ±1 level: ([\d.]+)%", all_text)}
    periods = ast.literal_eval(re.search(r"Test: (\{.*\})", all_text).group(1))
    top = {c: ast.literal_eval(d) for c, d in re.findall(r"(\w+) – top 5: (\{.*\})", all_text)}
    versions = re.search(r"scikit-learn ([\d.]+)", all_text).group(1)
    best = ast.literal_eval(re.search(r"Best model \(chosen on Validation\): (\{.*\})", all_text).group(1))
    groups = ast.literal_eval(re.search(r"Grouped importance \(MAE increase, µg/m³\): (\{.*\})", all_text).group(1))
    blend_w = dict(zip(CITIES, (float(w) for w in re.findall(r"Naive blend: w = ([\d.]+)", all_text))))
    outliers = {c: (int(n), float(pct)) for c, n, pct in
                re.findall(r"(\w+): upper bound = [\d.]+ µg/m³ \| (\d+) outlier hours \(([\d.]+)%\)", all_text)}
    missing = {c: (int(n0), float(a), float(b)) for c, n0, a, b in
               re.findall(r"(\w+): (\d+) values ≤ 0 or > 1000.*?missing PM2.5: ([\d.]+)% → ([\d.]+)%", all_text)}
    err = {c: (float(lo), int(hlo), float(hi), int(hhi), ast.literal_eval(months)) for c, lo, hlo, hi, hhi, months in
           re.findall(r"(\w+): MAE by hour min ([\d.]+) \(h(\d+)\) – max ([\d.]+) \(h(\d+)\).*?by month: (\{.*\})", all_text)}
    split = {c: tuple(int(x.replace(",", "")) for x in (a, b_, c_)) + (float(m1), float(m2), float(m3))
             for c, a, b_, c_, m1, m2, m3 in re.findall(
                 r"(\w+) split: Train ([\d,]+) \| Validation ([\d,]+) \| Test ([\d,]+) samples \| "
                 r"mean target: Train ([\d.]+), Validation ([\d.]+), Test ([\d.]+)", all_text)}
    window = {c: (int(n), float(mb), m, float(mm), float(sf), float(sa)) for c, n, mb, m, mm, sf, sa in re.findall(
        r"(\w+) – first 14 days of Test \((\d+) hours\): MAE Naive blend ([\d.]+) vs (\w+) ([\d.]+) \| "
        r"std of forecast ([\d.]+) vs actual ([\d.]+)", all_text)}
    samples = {c: int(n.replace(",", "")) for c, n in re.findall(r"(\w+): ([\d,]+) samples \| 23 features", all_text)}
    return {"test": test, "val": val, "aqi": aqi, "periods": periods, "top": top, "sklearn": versions, "best": best,
            "groups": groups, "blend_w": blend_w, "outliers": outliers, "missing": missing, "samples": samples, "err": err, "split": split, "window": window}


# ---------------------------------------------------------------- HTML building blocks
def esc(s):
    return html.escape(str(s))


def table(df, cls=""):
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in df.itertuples(index=False))
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def figure(name, number, caption):
    return (f'<figure><img src="report/figures/{name}.png" alt="Figure {number}">'
            f"<figcaption><b>Figure {number}.</b> {caption}</figcaption></figure>")


def build_body(r, for_pdf):
    test, val, aqi, periods, top = r["test"], r["val"], r["aqi"], r["periods"], r["top"]
    # the best model of each city is chosen on Validation (as in the notebook); Test is only reported
    best = pd.concat([test[(test.city == c) & (test.model == m)] for c, m in r["best"].items()]).set_index("city")
    base = test[test.model == "Persistence (baseline)"].set_index("city")
    blend = test[test.model == "Naive blend (baseline)"].set_index("city")
    groups, w, outl, miss, err = r["groups"], r["blend_w"], r["outliers"], r["missing"], r["err"]
    tree = test[test.model.isin(["RandomForest", "HistGradientBoosting"])]

    def b(city, col):
        return best.loc[city, col]

    hc_blend = tree[tree.city == "HCMC"].set_index("model")["vs_blend"]

    test_tbl = test.copy()
    test_tbl["MAE"] = test_tbl["MAE"].map("{:.2f}".format)
    test_tbl["RMSE"] = test_tbl["RMSE"].map("{:.2f}".format)
    test_tbl["R2"] = test_tbl["R2"].map("{:.3f}".format)
    test_tbl["reduction"] = test_tbl["reduction"].map("{:+.1f}%".format)
    test_tbl["vs_blend"] = test_tbl["vs_blend"].map("{:+.1f}%".format)
    for i, row in test_tbl.iterrows():
        if row["model"] == best.loc[row["city"], "model"]:
            test_tbl.loc[i, ["model", "MAE"]] = [f"<b>{row['model']}</b>", f"<b>{row['MAE']}</b>"]
    test_tbl.columns = ["City", "Model", "MAE (µg/m³)", "RMSE (µg/m³)", "R²", "Gain vs Persistence", "Gain vs Naive blend"]

    val_tbl = val.copy()
    for c in val_tbl.columns[1:]:
        val_tbl[c] = val_tbl[c].map("{:.2f}".format if c.startswith("MAE") else "{:.3f}".format)
    val_tbl.columns = ["Model", "MAE Hanoi", "MAE HCMC", "R² Hanoi", "R² HCMC"]

    members = "".join(f"<tr><td>{esc(n)}</td><td>{esc(i)}</td><td>{esc(role)}</td></tr>" for n, i, role in MEMBERS)
    video = (f'<a href="{esc(VIDEO_URL)}">YouTube presentation</a>' if VIDEO_URL
             else "YouTube presentation: <i>to be added</i>")
    links = f"""
<ul class="links">
  <li><a href="{LINKS['notebook']}">Notebook on GitHub</a> · <a href="{LINKS['colab']}">Open in Google Colab</a> (Run all)</li>
  <li><a href="{LINKS['executed']}">Executed copy with all outputs</a> · <a href="{LINKS['data']}">Dataset (CSV)</a></li>
  <li><a href="{LINKS['pdf'] if not for_pdf else REPO_URL + '/blob/main/03_TimeSeries/' + LINKS['pdf']}">PDF report</a> · {video}</li>
</ul>"""

    tp = {c: " → ".join(pd.to_datetime(x).strftime("%d/%m/%Y") for x in periods[c].split(" → ")) for c in CITIES}
    top_txt = {c: ", ".join(f"<code>{esc(k)}</code> ({v:.2f})" for k, v in top[c].items()) for c in CITIES}
    grp_txt = {c: ", ".join(f"{esc(k)} +{v:.2f}" for k, v in groups[c].items()) for c in CITIES}
    hc_months = list(err["HCMC"][4].items())
    sp, win = r["split"], r["window"]
    split_txt = "; ".join(f"{c}: {sp[c][0]:,} / {sp[c][1]:,} / {sp[c][2]:,} samples, mean target {sp[c][3]:.1f} → {sp[c][4]:.1f} → {sp[c][5]:.1f} µg/m³"
                          for c in CITIES)
    win_txt = "; ".join(f"{c}: {win[c][2]} {win[c][3]:.2f} vs Naive blend {win[c][1]:.2f} ({win[c][0]} hours)" for c in CITIES)
    std_txt = ", ".join(f"{c} {win[c][4]:.1f} vs {win[c][5]:.1f}" for c in CITIES)

    return f"""
<header class="title">
  <p class="kicker">Programming Foundations for Data Analysis and Visualization · Semester 261 (2026–2027)</p>
  <h1>Part 3 — Time Series: 24-Hour-Ahead PM2.5 Forecasting, Hanoi vs Ho Chi Minh City</h1>
  <p class="meta">Ho Chi Minh City University of Technology (HCMUT) – VNU-HCM · Faculty of Computer Science and Engineering ·
  Instructor: Lê Thành Sách</p>
  <p class="meta"><b>Data type:</b> Time series · <b>Group:</b> {esc(GROUP_NAME)}</p>
</header>

<table class="members"><thead><tr><th>Member</th><th>Student ID</th><th>Contribution</th></tr></thead><tbody>{members}</tbody></table>
{links}

<div class="summary">
  <p><b>In one paragraph.</b> We collected 22 months of hourly PM2.5 (OpenAQ) and weather (Open-Meteo) data for one station in Hanoi and one in
  Ho Chi Minh City (HCMC), and forecast PM2.5 24 hours ahead using only information available at forecast time, scored on measured values only.
  On an untouched, chronologically last Test set, the best model (chosen on Validation) lowers the MAE of the strongest naive forecast
  (a blend of the current value and its 24h mean) by <b>{b('Hanoi', 'vs_blend'):.1f}% in Hanoi</b> ({b('Hanoi', 'model')}, {blend.loc['Hanoi', 'MAE']:.2f} → {b('Hanoi', 'MAE'):.2f} µg/m³)
  and by <b>{b('HCMC', 'vs_blend'):.1f}% in HCMC</b> ({b('HCMC', 'model')}, {blend.loc['HCMC', 'MAE']:.2f} → {b('HCMC', 'MAE'):.2f} µg/m³)
  — {b('Hanoi', 'reduction'):.1f}% / {b('HCMC', 'reduction'):.1f}% against plain Persistence — and predicts the correct US EPA air quality level in
  {aqi['Hanoi'][0]:.1f}% / {aqi['HCMC'][0]:.1f}% of hours. The best model type differs by city, and both models under-forecast polluted hours.</p>
</div>

<h2>1. Problem and questions</h2>
<p>At hour <i>t</i>, using the PM2.5 history and the current weather, forecast PM2.5 at <i>t</i> + 24h (the same hour tomorrow) —
a time series regression problem. Hanoi (four seasons, winter temperature inversion) and HCMC (wet/dry seasons) have different pollution
mechanisms, so we apply the <b>same pipeline to each city independently</b> and ask:</p>
<ol>
  <li>How much better than a naive baseline can machine learning forecast PM2.5 one day ahead in each city?</li>
  <li>Which city is harder to forecast, and what drives PM2.5 in each place?</li>
</ol>
<p>Three naive baselines set the bar: <b>Persistence</b> ŷ(<i>t</i>+24) = pm25(<i>t</i>) (also the 24h seasonal-naive forecast), the <b>24h mean</b>,
and a <b>Naive blend</b> w·pm25(<i>t</i>) + (1 − w)·mean<sub>24h</sub>(<i>t</i>) with w learned on Train (w = {w['Hanoi']:.2f} / {w['HCMC']:.2f}).
A model is only useful if it beats the strongest of them — the Naive blend.</p>

<h2>2. Dataset (self-collected)</h2>
<table>
<thead><tr><th>Component</th><th>Source</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>PM2.5 – Hanoi</td><td>OpenAQ v3, station <i>46 Luu Quang Vu</i> (Hanoi Air Quality Monitoring Network)</td><td>City monitoring station; 32.8% of hours missing (31–97% per month from 11/2025)</td></tr>
<tr><td>PM2.5 – HCMC</td><td>OpenAQ v3, station <i>CMT8</i> (AirGradient)</td><td>Low-cost sensor; 8.4% of hours missing</td></tr>
<tr><td>Weather</td><td>Open-Meteo Historical Weather API</td><td>Temperature, humidity, precipitation, wind speed/direction, pressure, cloud cover</td></tr>
</tbody></table>
<p>Period 01/12/2024 – 30/09/2026: <b>16,056 hourly rows × 11 columns per city</b> on a complete hourly grid (Vietnam time), where hours without a
measurement are <code>NaN</code>. The crawler (<code>scripts/crawl_air_quality.py</code>) is reproducible and falls back to Open-Meteo Air Quality (CAMS) if
no OpenAQ key is available.</p>
<p><b>Key EDA findings.</b> Mean PM2.5 is similar (31.7 vs 32.1 µg/m³) and exceeds the WHO 24h guideline (15 µg/m³) in 56% (Hanoi) and 82% (HCMC) of hours,
but the shapes differ: Hanoi is more volatile (std 20.6 vs 16.3) with a strong winter peak (January 55 vs August 19 µg/m³), and correlates with
temperature (−0.31) and pressure (+0.31); HCMC has a narrower seasonal range, a 7h rush-hour peak and its strongest link to wind speed (−0.36).
The autocorrelation drops from 0.92 / 0.88 at lag 1h to 0.56 / 0.45 at lag 24h, so a day-ahead forecast is much harder than a next-hour one.</p>
{figure('daily', 1, 'Daily mean PM2.5 against the WHO 24h guideline. Hanoi has a pronounced winter peak and patchy data from November 2025 onwards; HCMC is more stable.')}
{figure('cycle', 2, 'Mean PM2.5 by hour of day, dry/cool vs wet/hot half-year. HCMC peaks at 7h (traffic), Hanoi in the evening (≈34 µg/m³ at 22h, night-time inversion).')}

<h2>3. Data preparation</h2>
<ul>
<li><b>Cleaning:</b> values ≤ 0 or &gt; 1000 µg/m³ ({miss['Hanoi'][0]} in Hanoi, {miss['HCMC'][0]} in HCMC) and "stuck sensor" runs (≥ 6 identical hours, none found) are set to missing.</li>
<li><b>Missing values without look-ahead:</b> time interpolation would fill a gap with the value measured <i>after</i> it, which is unknown at forecast time.
So the <b>target is always a measured value</b> (never filled), and the features use a <b>causal forward-fill of at most 6h</b> — missing PM2.5 in the feature series
falls from {miss['Hanoi'][1]:.1f}% → {miss['Hanoi'][2]:.1f}% (Hanoi) and {miss['HCMC'][1]:.1f}% → {miss['HCMC'][2]:.1f}% (HCMC).
Longer gaps are not filled; samples without a current value or target are dropped ({r['samples']['Hanoi']:,} / {r['samples']['HCMC']:,} samples remain),
and remaining feature gaps use a median <code>SimpleImputer</code> fitted on Train.</li>
<li><b>Outliers:</b> the IQR rule (k = 3) flags {outl['Hanoi'][0]} ({outl['Hanoi'][1]:.2f}%) and {outl['HCMC'][0]} ({outl['HCMC'][1]:.2f}%) hours. They are real pollution events, so they stay in the <i>target</i>;
only the PM2.5 <i>features</i> are clipped by a custom <code>IQRClipper</code> transformer learned on Train.</li>
<li><b>Features (23):</b> PM2.5 lags (0–23h) and rolling statistics, weather at <i>t</i> plus 24h rain total, cyclic hour of the forecast time (numerical,
<code>StandardScaler</code>), and season, day of week and 8-sector wind direction (categorical, <code>OneHotEncoder</code>) — 39 columns after encoding.
Calendar features describe the target time <i>t</i> + 24h, which is known in advance, so they do not leak.</li>
<li><b>Split:</b> chronological 70 / 15 / 15 Train / Validation / Test with a 24h gap before each boundary, so no target leaks into the next set
(the same 24h <code>gap</code> is used between the folds of <code>TimeSeriesSplit</code> during tuning).
All preprocessing lives in one <code>ColumnTransformer</code> + <code>Pipeline</code> fitted on Train only.</li>
</ul>
{figure('split', 3, f"Chronological split per city (Train / Validation / Test — {split_txt}). The later sets fall in the cleaner season, so the models are tested on a distribution they saw less of. The Test periods differ (Hanoi {tp['Hanoi']}, HCMC {tp['HCMC']}) because Hanoi has more missing data; Hanoi's Test set has almost no afternoon hours.")}

<h2>4. Modeling</h2>
<p>Models: three naive baselines (<b>Persistence</b>, <b>24h mean</b>, <b>Naive blend</b>), <b>Ridge</b>, <b>RandomForest</b> and <b>HistGradientBoosting</b>.
The three ML models are tuned with <code>GridSearchCV</code> on <code>TimeSeriesSplit(n_splits=3, gap=24)</code> (MAE). A <code>CityForecaster</code>
class runs the whole flow for one city. Validation results (used to compare models and <b>choose the best model of each city</b>,
so the Test set is never used for selection):</p>
{table(val_tbl)}
<p>The Naive blend is clearly better than Persistence in both cities. In Hanoi all three ML models beat it; in HCMC only the strongly regularized Ridge
(alpha = 100) does, and only marginally — the tree models are worse than the blend there, because the signal is weak (R² ≤ 0.15).
Selected models: <b>{r['best']['Hanoi']}</b> for Hanoi and <b>{r['best']['HCMC']}</b> for HCMC.</p>

<h2>5. Results on the Test set</h2>
<p>Each model is retrained on Train + Validation and evaluated <b>once</b> on Test. The model chosen on Validation is shown in bold. "Gain" = MAE reduction relative to the reference (+ = better).</p>
{table(test_tbl, 'metrics')}
{figure('mae', 4, f"Test MAE per model and city (labels: MAE reduction vs the Naive blend; + = better, − = worse). The model chosen on Validation cuts MAE by {b('Hanoi', 'vs_blend'):.1f}% in Hanoi ({b('Hanoi', 'model')}) and by {b('HCMC', 'vs_blend'):.1f}% in HCMC ({b('HCMC', 'model')}) relative to the blend; in HCMC the tree models gain only {hc_blend.min():.1f}–{hc_blend.max():.1f}%. There is no single best model for both cities.")}
{figure('forecast', 5, f"First 14 days of the Test set (time axis = forecast time). MAE in this window — {win_txt}: a clear gain in Hanoi, a tie in HCMC. The forecasts are much smoother than reality (std of forecast vs actual, µg/m³: {std_txt}), so short spikes are missed.")}
<p><b>Answer to question 1.</b> Against plain Persistence, machine learning looks like a {b('Hanoi', 'reduction'):.0f}% / {b('HCMC', 'reduction'):.0f}% gain,
but part of that comes from simply smoothing the current value. Against the best naive forecast the gain is about <b>{b('Hanoi', 'vs_blend'):.0f}% in Hanoi</b>
and only <b>{b('HCMC', 'vs_blend'):.0f}% in HCMC</b>, where only the linear Ridge model adds a consistent improvement.
In Hanoi the best model is a tree ensemble, but its lead over the other ML models is small.
Absolute MAE should not be compared between cities because the Test periods differ; relative improvement over the baseline is the fair comparison.</p>

<h2>6. Error analysis and drivers</h2>
<p>PM2.5 forecasts are mapped to US EPA 2024 levels (Good ≤ 9, Moderate ≤ 35.4, USG ≤ 55.4, Unhealthy ≤ 125.4 µg/m³) to build a level confusion matrix.</p>
{figure('aqi', 6, f"AQI-level confusion matrix of the best model. Correct level: {aqi['Hanoi'][0]:.1f}% (Hanoi) and {aqi['HCMC'][0]:.1f}% (HCMC); within ±1 level: {aqi['Hanoi'][1]:.1f}% / {aqi['HCMC'][1]:.1f}%. Most errors are polluted hours forecast as 'Moderate', and 'Good' is never predicted.")}
<p>The main weakness is <b>under-forecasting polluted hours</b>: forecasts are pulled towards the mean, so the model follows the trend but is not yet a
reliable early-warning tool. The error also grows at the end of the Test period (HCMC: {hc_months[0][1]:.1f} µg/m³ in {hc_months[0][0]} → {hc_months[-1][1]:.1f} in {hc_months[-1][0]})
when pollution rises again, and HCMC's error peaks in the daytime ({err['HCMC'][2]:.1f} µg/m³ at {err['HCMC'][3]}h).</p>
{figure('importance', 7, f"Permutation importance on Test (MAE increase when a feature is shuffled). Hanoi: {top_txt['Hanoi']}. HCMC: {top_txt['HCMC']}. Correlated PM2.5 lag features share their importance, so groups are also shuffled together — Hanoi: {grp_txt['Hanoi']}; HCMC: {grp_txt['HCMC']}.")}
<p><b>Answer to question 2.</b> Recent PM2.5 is the most important group in both cities. HCMC then depends on temperature, hour of day and wind speed
(daily traffic rhythm and dispersion), matching the EDA. In Hanoi, PM2.5 history and weather matter while the calendar adds almost nothing on a Test period
that covers only the end of winter. Hanoi is the harder city in absolute terms (higher volatility and winter peaks), while HCMC gives the
smaller relative gain over the naive forecast.</p>

<h2>7. Conclusions, limitations and future work</h2>
<ul>
<li><b>Conclusion:</b> one reusable pipeline, trained independently per city, beats the strongest naive forecast by {b('Hanoi', 'vs_blend'):.1f}% (Hanoi) and
{b('HCMC', 'vs_blend'):.1f}% (HCMC) in Test MAE ({b('Hanoi', 'reduction'):.1f}% / {b('HCMC', 'reduction'):.1f}% vs Persistence) and gets the AQI level right in
{min(a[0] for a in aqi.values()):.0f}–{max(a[0] for a in aqi.values()):.0f}% of hours.</li>
<li><b>Limitations:</b> one station per city (HCMC is a low-cost sensor); Hanoi has 31–97% missing hours per month from November 2025, so its Validation/Test sets are small,
lack afternoon hours and contain only the end of winter; only 22 months of data (fewer than two seasonal cycles); differences between the best models are small and
not yet tested statistically; weather features are observations at <i>t</i>, not forecasts for <i>t</i> + 24h; polluted hours are under-forecast.</li>
<li><b>Future work:</b> add weather forecasts and more stations; quantile regression or an exceedance classifier for early warning;
block-bootstrap confidence intervals and a shared calendar split for both cities; SARIMAX, multi-horizon LightGBM or LSTM once longer data is available.</li>
</ul>
<p class="note">All numbers in this report are read automatically from the executed notebook (scikit-learn {r['sklearn']}, random_state = 42).
Tree-model results can shift by a few hundredths of µg/m³ across scikit-learn versions; Ridge and the naive baselines do not change.</p>
"""


# ---------------------------------------------------------------- page / PDF wrappers
BASE_CSS = """
:root{--bg:#ffffff;--fg:#1d2433;--muted:#5b6475;--line:#d9dee7;--accent:#b4232c;--soft:#f4f6fa;--link:#1f5fae}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#12151c;--fg:#e6e9ef;--muted:#a3abba;--line:#2c3340;--accent:#f0717a;--soft:#1b2029;--link:#7fb0f0}}
:root[data-theme="dark"]{--bg:#12151c;--fg:#e6e9ef;--muted:#a3abba;--line:#2c3340;--accent:#f0717a;--soft:#1b2029;--link:#7fb0f0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:"DejaVu Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.55}
main{max-width:920px;margin:0 auto;padding:24px 16px 64px}
a{color:var(--link)}
h1{font-size:1.75rem;line-height:1.25;margin:.2em 0 .4em}
h2{font-size:1.25rem;margin:1.8em 0 .5em;padding-bottom:.25em;border-bottom:2px solid var(--line)}
.kicker{color:var(--accent);font-weight:600;font-size:.85rem;letter-spacing:.02em;margin:0}
.meta{color:var(--muted);margin:.2em 0;font-size:.95rem}
.summary{background:var(--soft);border-left:4px solid var(--accent);padding:12px 16px;margin:16px 0;border-radius:4px}
.summary p{margin:0}
.links{padding-left:1.2em}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:.9rem;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;white-space:nowrap}
td{white-space:normal}
th{background:var(--soft)}
figure{margin:18px 0}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
figcaption{font-size:.88rem;color:var(--muted);margin-top:6px}
code{font-family:"DejaVu Sans Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.88em;background:var(--soft);padding:0 3px;border-radius:3px}
.note{font-size:.85rem;color:var(--muted)}
nav.top{font-size:.9rem;margin-bottom:12px}
"""

PDF_CSS = """
@page{size:A4;margin:16mm 15mm 18mm;@bottom-center{content:"Part 3 — Time Series · page " counter(page) " of " counter(pages);font-size:8pt;color:#777;font-family:"DejaVu Sans",sans-serif}}
body{font-size:9.6pt;background:#fff;color:#1d2433}
main{max-width:none;padding:0}
h1{font-size:16pt}
h2{font-size:12pt;margin-top:1.3em;break-after:avoid}
table{display:table;font-size:8.4pt}
th{white-space:normal}
figure{break-inside:avoid}
figure img{border:none}
"""


def write_page(body):
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PM2.5 Forecasting — Part 3</title>
<meta name="description" content="Part 3 (Time Series) of the course project: 24-hour-ahead PM2.5 forecasting for Hanoi and Ho Chi Minh City.">
<style>{BASE_CSS}</style>
</head>
<body>
<main>
<nav class="top"><a href="{LINKS['landing']}">← Group landing page</a> · <a href="{LINKS['repo']}">Repository</a></nav>
{body}
</main>
</body>
</html>
"""
    with open(PAGE_PATH, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"Wrote {PAGE_PATH}")


def write_pdf(body):
    from weasyprint import HTML  # imported here so the page can be built without weasyprint
    doc = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><style>{BASE_CSS}{PDF_CSS}</style></head><body><main>{body}</main></body></html>"
    HTML(string=doc, base_url=ROOT).write_pdf(PDF_PATH)
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    results = load_results()
    write_page(build_body(results, for_pdf=False))
    write_pdf(build_body(results, for_pdf=True))
