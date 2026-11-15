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
GROUP_NAME = "IndexError"            # must match the GroupRegistration sheet
MEMBERS = [                          # (full name, student ID, role in this part)
    ("Nguyễn Văn Hoàng Khang", "2570754", "Time series: data collection, notebook, report"),
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
    "landing": "https://perckgg.github.io/IndexError-DataViz/",
}

# Figures to export: name -> first line(s) of the code cell that draws it
FIGURES = {
    "daily": "fig, ax = plt.subplots(figsize=(14, 5))",
    "cycle": "def complete_days_only(",
    "split": "TEST_START = pd.Timestamp(",
    "wf": "fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)\nmonthly_mae",
    "mae": "MODEL_ORDER = list(make_models())",
    "boot": "# Forest plot of the bootstrap confidence intervals",
    "forecast": "fig, axes = plt.subplots(2, 1, figsize=(14, 8))",
    "aqi": "EPA_BINS = ",
    "importance": "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\nfor ax, (city, f) in zip(axes, forecasters.items()):\n    model",
}
CITIES = ["Hanoi", "HCMC"]
SETS = ["Walk-forward", "Test"]
BLEND = "Naive blend (baseline)"


# ---------------------------------------------------------------- read results from the notebook
def load_results():
    cells = json.load(open(EXECUTED_NB, encoding="utf-8"))["cells"]
    outputs = []          # ("text" | "html", content), in notebook order
    os.makedirs(FIG_DIR, exist_ok=True)
    found = set()
    for c in cells:
        src = "".join(c["source"])
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                raise RuntimeError("The executed notebook contains an error; re-run it first.")
            data = o.get("data", {})
            outputs.append(("text", "".join(o.get("text") or data.get("text/plain", ""))))
            if "text/html" in data:
                outputs.append(("html", "".join(data["text/html"])))
            if "image/png" in data:
                for name, start in FIGURES.items():
                    if src.startswith(start) and name not in found:
                        with open(os.path.join(FIG_DIR, f"{name}.png"), "wb") as fh:
                            fh.write(base64.b64decode(data["image/png"]))
                        found.add(name)
    missing_figs = set(FIGURES) - found
    if missing_figs:
        raise RuntimeError(f"Figures not found in the notebook: {missing_figs}")
    texts = [x for k, x in outputs if k == "text"]
    all_text = "\n".join(texts)

    def html_after(marker):
        """First HTML table that follows the first text output containing `marker`."""
        i = next(i for i, (k, x) in enumerate(outputs) if k == "text" and marker in x)
        h = next(x for k, x in outputs[i + 1:] if k == "html")
        return pd.read_html(StringIO(h))[0]

    def one(pattern):
        return re.search(pattern, all_text).group(1)

    test_html = next(x for k, x in outputs if k == "html" and "MAE vs Naive blend" in x)
    test = pd.read_html(StringIO(test_html))[0]
    test.columns = ["city", "model", "MAE", "RMSE", "R2", "reduction", "vs_blend"]
    test["city"] = test["city"].ffill()
    for col in ["reduction", "vs_blend"]:
        test[col] = test[col].str.rstrip("%").astype(float)

    wf_text = next(t for t in texts if t.lstrip().startswith("MAE") and "Persistence (baseline)" in t)
    wf_rows = []
    for line in wf_text.splitlines():
        m = re.match(r"^((?:Persistence|Mean 24h|Naive blend) \(baseline\)|Ridge|RandomForest|HistGradientBoosting)\s+(.*)$", line)
        if m:
            nums = [float(x) for x in m.group(2).split()]
            wf_rows.append({"model": m.group(1), "MAE Hanoi": nums[0], "MAE HCMC": nums[1],
                            "R2 Hanoi": nums[4], "R2 HCMC": nums[5]})
    wf = pd.DataFrame(wf_rows)

    def month_table(marker):
        t = html_after(marker)
        return t.set_index(t.columns[0])

    summary_html = next(x for k, x in outputs if k == "html" and "missing_rate" in x)
    summary = pd.read_html(StringIO(summary_html))[0].set_index("Unnamed: 0")
    gaps = pd.read_html(StringIO(next(x for k, x in outputs if k == "html" and "Longest gap (hours)" in x)))[0].set_index("City")
    monthly_text = next(t for t in texts if t.lstrip().startswith("mean") and "median" in t and "month" in t)
    header = re.search(r"^city\s+(.*)$", monthly_text, flags=re.M).group(1).split()
    monthly = {}
    for mo, *vals in re.findall(r"^(\d{1,2})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", monthly_text, flags=re.M):
        for city, v in zip(header[:2], vals[:2]):
            monthly.setdefault(city, {})[int(mo)] = float(v)
    acf_text = next(t for t in texts if "lag (hours)" in t)
    acf_cols = acf_text.split("\n")[0].split()
    acf = {int(l): dict(zip(acf_cols, map(float, v.split()))) for l, v in
           re.findall(r"^(\d+)\s+([\d.]+\s+[\d.]+)\s*$", acf_text, flags=re.M)}

    return {
        "test": test, "wf": wf, "summary": summary, "gaps": gaps, "monthly": monthly, "acf": acf,
        "coverage": html_after("Candidate stations"),
        "rows": {c: (int(n.replace(",", "")), d0, d1, int(days)) for c, n, d0, d1, days in re.findall(
            r"(\w+): ([\d,]+) rows × \d+ columns \| (\S+) \S+ → (\S+) \S+ \((\d+) days\)", all_text)},
        "spearman": {c: ast.literal_eval(d) for c, d in re.findall(r"(\w+) – Spearman correlation of PM2.5 with: (\{.*\})", all_text)},
        "cycle": {(c, h): (int(n), int(hmax), float(vmax), int(hmin), float(vmin)) for c, h, n, hmax, vmax, hmin, vmin in re.findall(
            r"(\w+) (May–Oct|Nov–Apr) \((\d+) complete days\): highest mean at (\d+)h \(([\d.]+) µg/m³\), lowest at (\d+)h \(([\d.]+) µg/m³\)",
            all_text)},
        "aqi": {c: (float(a), float(b)) for c, a, b in
                re.findall(r"(\w+) Test: correct EPA band ([\d.]+)% \| within ±1 band: ([\d.]+)%", all_text)},
        "aqi_wf": {c: float(a) for c, a in re.findall(r"(\w+): correct EPA band on the walk-forward forecasts ([\d.]+)%", all_text)},
        "periods": ast.literal_eval(one(r"Test: (\{.*\})")),
        "top": {c: ast.literal_eval(d) for c, d in re.findall(r"(\w+) – top 5: (\{.*\})", all_text)},
        "sklearn": one(r"scikit-learn ([\d.]+)"),
        "best": ast.literal_eval(one(r"Best model \(lowest pooled walk-forward MAE among the ML models\): (\{.*\})")),
        "groups": ast.literal_eval(one(r"Grouped importance \(MAE increase, µg/m³\): (\{.*\})")),
        "w_folds": {c: ast.literal_eval(d) for c, d in re.findall(r"(\w+) – Naive blend weight w learned in each fold: (\{.*\})", all_text)},
        "outliers": {c: (int(n), float(pct)) for c, n, pct in
                     re.findall(r"(\w+): upper bound = [\d.]+ µg/m³ \| (\d+) outlier hours \(([\d.]+)%\)", all_text)},
        "missing": {c: (int(n0), float(a), float(b)) for c, n0, a, b in
                    re.findall(r"(\w+): (\d+) values ≤ 0 or > 1000.*?missing PM2.5: ([\d.]+)% → ([\d.]+)%", all_text)},
        "samples": {c: int(n.replace(",", "")) for c, n in re.findall(r"(\w+): ([\d,]+) samples \| 23 features", all_text)},
        "err": {(c, st): (float(lo), int(hlo), float(hi), int(hhi)) for c, st, lo, hlo, hi, hhi in re.findall(
            r"(\w+) ([\w-]+): MAE by hour of the forecast time min ([\d.]+) \(h(\d+)\) – max ([\d.]+) \(h(\d+)\)", all_text)},
        "split": {c: (int(nd.replace(",", "")), d0, d1, int(nt.replace(",", "")), t0, t1, float(md), float(mt))
                  for c, nd, d0, d1, nt, t0, t1, md, mt in re.findall(
                      r"(\w+) split: Development ([\d,]+) \((\S+) → (\S+) \S+\) \| Test ([\d,]+) \((\S+) → (\S+) \S+\) samples \| "
                      r"mean target: Development ([\d.]+), Test ([\d.]+)", all_text)},
        "window": {c: (int(n), float(mb), m, float(mm), float(sf), float(sa)) for c, n, mb, m, mm, sf, sa in re.findall(
            r"(\w+) – first 14 days of Test \((\d+) hours\): MAE Naive blend ([\d.]+) vs (\w+) ([\d.]+) \| "
            r"std of forecast ([\d.]+) vs actual ([\d.]+)", all_text)},
        "boot": {(c, st, b): (float(g), float(lo), float(hi)) for c, st, b, g, lo, hi in re.findall(
            r"BOOT \| (\w+) \| ([\w-]+) \| \w+ vs ([^|]+?) \| gain ([+-][\d.]+)% \[([+-][\d.]+)%, ([+-][\d.]+)%\]", all_text)},
        "sens": {(c, st): re.findall(r"(\d)-day blocks \((\d+)\): \[([+-][\d.]+)%, ([+-][\d.]+)%\]", line)
                 for c, st, line in re.findall(r"^(\w+) ([\w-]+): (1-day blocks.*)$", all_text, flags=re.M)},
        "monthly_mae": {c: month_table(f"{c} – walk-forward MAE by month") for c in CITIES},
        "bias": {c: month_table(f"{c} – walk-forward bias by month") for c in CITIES},
    }


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


def pct(x):
    return f"{x * 100:.1f}%"


def build_body(r, for_pdf):
    test, wf, aqi, periods, top = r["test"], r["wf"], r["aqi"], r["periods"], r["top"]
    # the best model of each city is chosen by walk-forward validation (as in the notebook); Test is only reported
    best = pd.concat([test[(test.city == c) & (test.model == m)] for c, m in r["best"].items()]).set_index("city")
    blend = test[test.model == BLEND].set_index("city")
    groups, outl, miss, err = r["groups"], r["outliers"], r["missing"], r["err"]
    summ, gaps, monthly, acf, sp_ = r["summary"], r["gaps"], r["monthly"], r["acf"], r["spearman"]
    cyc, rows, split, win, bt, sens = r["cycle"], r["rows"], r["split"], r["window"], r["boot"], r["sens"]
    mm, bias = r["monthly_mae"], r["bias"]

    def b(city, col):
        return best.loc[city, col]

    def d(x):
        return pd.to_datetime(x).strftime("%d/%m/%Y")

    # ---- tables
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

    wf_tbl = wf.copy()
    for c in wf_tbl.columns[1:]:
        wf_tbl[c] = wf_tbl[c].map("{:.3f}".format)
    for c in CITIES:
        i = wf_tbl.index[wf_tbl.model == r["best"][c]][0]
        wf_tbl.loc[i, f"MAE {c}"] = f"<b>{wf_tbl.loc[i, f'MAE {c}']}</b>"
    wf_tbl.columns = ["Model", "MAE Hanoi", "MAE HCMC", "R² Hanoi", "R² HCMC"]

    cov = r["coverage"].drop(columns=[c for c in r["coverage"].columns if c.startswith("Unnamed")])
    cov_tbl = pd.DataFrame({
        "City": cov["city"], "Station (OpenAQ sensor)": [f"{esc(n)} ({i})" for n, i in zip(cov["station"], cov["sensor_id"])],
        "Type": cov["type"], "Hours measured": cov["coverage"].map(pct), "Worst hour of day": cov["worst_hour_cov"].map(pct),
        "Worst month": cov["worst_month_cov"].map(pct), "Longest gap (h)": cov["longest_gap_h"]})

    def ci(city, st, vs):
        g, lo, hi = bt[(city, st, vs)]
        txt = f"{g:+.1f}% [{lo:+.1f}, {hi:+.1f}]"
        return f"<b>{txt}</b>" if lo > 0 or hi < 0 else txt

    second_ml = {c: (wf[~wf.model.isin([r["best"][c]]) & wf.model.isin(["Ridge", "RandomForest", "HistGradientBoosting"])]
                     .sort_values(f"MAE {c}").model.iloc[0]) for c in CITIES}
    boot_rows = [(c, f"{r['best'][c]} vs {vs}", ci(c, "Walk-forward", vs), ci(c, "Test", vs))
                 for c in CITIES for vs in ("Persistence (baseline)", BLEND, second_ml[c])]
    boot_tbl = pd.DataFrame(boot_rows, columns=["City", "Comparison", "Walk-forward gain [95% CI]", "Test gain [95% CI]"])
    blocks = {st: sens[("Hanoi", st)][1][1] for st in SETS}
    sens_txt = "; ".join(f"{c} {st}: " + ", ".join(f"{L}-day [{lo}, {hi}]%" for L, _, lo, hi in sens[(c, st)])
                         for c in CITIES for st in SETS)

    members = "".join(f"<tr><td>{esc(n)}</td><td>{esc(i)}</td><td>{esc(role)}</td></tr>" for n, i, role in MEMBERS)
    video = (f'<a href="{esc(VIDEO_URL)}">YouTube presentation</a>' if VIDEO_URL
             else "YouTube presentation: <i>to be added</i>")
    links = f"""
<ul class="links">
  <li><a href="{LINKS['notebook']}">Notebook on GitHub</a> · <a href="{LINKS['colab']}">Open in Google Colab</a> (Run all)</li>
  <li><a href="{LINKS['executed']}">Executed copy with all outputs</a> · <a href="{LINKS['data']}">Dataset (CSV)</a></li>
  <li><a href="{LINKS['pdf'] if not for_pdf else REPO_URL + '/blob/main/03_TimeSeries/' + LINKS['pdf']}">PDF report</a> · {video}</li>
</ul>"""

    # ---- derived text
    top_txt = {c: ", ".join(f"<code>{esc(k)}</code> ({v:.2f})" for k, v in top[c].items()) for c in CITIES}
    grp_txt = {c: ", ".join(f"{esc(k)} {v:+.2f}" for k, v in groups[c].items()) for c in CITIES}
    w_rng = {c: (min(r["w_folds"][c].values()), max(r["w_folds"][c].values())) for c in CITIES}
    peak = {c: max(monthly[c], key=monthly[c].get) for c in CITIES}
    low = {c: min(monthly[c], key=monthly[c].get) for c in CITIES}
    month_name = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August",
                  9: "September", 10: "October", 11: "November", 12: "December"}
    cov_city = cov.groupby("city").first()   # rows are sorted by coverage, so first = chosen station
    old = cov[cov.sensor_id == 7772032].iloc[0]
    start, end, ndays = rows["Hanoi"][1], rows["Hanoi"][2], rows["Hanoi"][3]

    def month_compare(c):
        m = mm[c]
        diff = m.loc[r["best"][c]] - m.loc[BLEND]           # < 0: the ML model is better than the blend
        worse = diff[diff > 0].index.tolist()
        return diff, worse

    diff_h, worse_h = month_compare("Hanoi")
    diff_c, worse_c = month_compare("HCMC")
    bh, bc = bias["Hanoi"], bias["HCMC"]

    def mae_pair(c, month):
        return f"{mm[c].loc[r['best'][c], month]:.2f} vs {mm[c].loc[BLEND, month]:.2f}"

    def bias_txt(c, month):
        return f"{bias[c].loc[r['best'][c], month]:+.1f}"

    win_months_h = [mo for mo in ("2026-01", "2026-02") if diff_h[mo] < 0]
    win_txt = "; ".join(f"{c}: {win[c][2]} {win[c][3]:.2f} vs Naive blend {win[c][1]:.2f} ({win[c][0]} hours)" for c in CITIES)
    std_txt = ", ".join(f"{c} {win[c][4]:.1f} vs {win[c][5]:.1f}" for c in CITIES)
    sig_blend = {(c, st): bt[(c, st, BLEND)][1] > 0 for c in CITIES for st in SETS}

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
  <p><b>In one paragraph.</b> We collected {ndays} days ({d(start)} – {d(end)}) of hourly PM2.5 (OpenAQ) and weather (Open-Meteo) data for one station
  in Hanoi and one in Ho Chi Minh City (HCMC), and forecast PM2.5 24 hours ahead using only information available at forecast time, scored on measured values only.
  Models are selected by <b>monthly walk-forward validation</b> over 11 months that include the whole winter, then evaluated once on the last
  2.5 months (Test). On Test the selected model lowers the MAE of the strongest naive forecast (a blend of the current value and its 24h mean)
  by <b>{b('Hanoi', 'vs_blend'):.1f}% in Hanoi</b> ({b('Hanoi', 'model')}, {blend.loc['Hanoi', 'MAE']:.2f} → {b('Hanoi', 'MAE'):.2f} µg/m³)
  and by <b>{b('HCMC', 'vs_blend'):.1f}% in HCMC</b> ({b('HCMC', 'model')}, {blend.loc['HCMC', 'MAE']:.2f} → {b('HCMC', 'MAE'):.2f} µg/m³);
  block-bootstrap intervals say only the HCMC Test gain is clearly larger than chance. Against plain Persistence the gain is
  {b('Hanoi', 'reduction'):.1f}% / {b('HCMC', 'reduction'):.1f}% and significant in both cities. The ML gain is concentrated in the winter peak; in months
  where the pollution level changes, the ML models lag behind and can lose to the naive forecast.</p>
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
and a <b>Naive blend</b> w·pm25(<i>t</i>) + (1 − w)·mean<sub>24h</sub>(<i>t</i>) with w learned on each training window
(w = {w_rng['Hanoi'][0]:.2f}–{w_rng['Hanoi'][1]:.2f} in Hanoi, {w_rng['HCMC'][0]:.2f}–{w_rng['HCMC'][1]:.2f} in HCMC).
A model is only useful if it beats the strongest of them — the Naive blend.</p>

<h2>2. Dataset (self-collected)</h2>
<table>
<thead><tr><th>Component</th><th>Source</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>PM2.5 – Hanoi</td><td>OpenAQ v3, station <i>Công viên Nhân Chính – Khuất Duy Tiến</i> (Hanoi Air Quality Monitoring Network)</td><td>Reference monitor; {gaps.loc['Hanoi', 'Missing rate']} of hours missing</td></tr>
<tr><td>PM2.5 – HCMC</td><td>OpenAQ v3, station <i>CMT8</i> (AirGradient)</td><td>Low-cost sensor; {gaps.loc['HCMC', 'Missing rate']} of hours missing</td></tr>
<tr><td>Weather</td><td>Open-Meteo Historical Weather API, at each station's coordinates</td><td>Temperature, humidity, precipitation, wind speed/direction, pressure, cloud cover</td></tr>
</tbody></table>
<p>Period {d(start)} – {d(end)}, the same for both cities: <b>{rows['Hanoi'][0]:,} hourly rows × 11 columns per city</b> on a complete hourly grid
(Vietnam time), where hours without a measurement are <code>NaN</code>. The crawler (<code>scripts/crawl_air_quality.py</code>) is reproducible and falls back to
Open-Meteo Air Quality (CAMS) if no OpenAQ key is available.</p>
<p><b>Choice of stations.</b> <code>scripts/check_station_coverage.py</code> scores every OpenAQ PM2.5 sensor within 25 km of each city centre that
reported in the period. Besides the share of hours measured, it reports the <i>worst hour of the day</i>: a station that misses mostly certain hours
(e.g. afternoons) has gaps that are not at random, which would bias both the daily cycle and the forecast errors. The station used in an earlier
version of this project (46 Lưu Quang Vũ) measured {pct(old['coverage'])} of hours overall but only {pct(old['worst_hour_cov'])} of its worst hour of the day;
Nhân Chính measured {pct(cov_city.loc['Hanoi', 'coverage'])} ({pct(cov_city.loc['Hanoi', 'worst_hour_cov'])} for its worst hour). HCMC has a single station with continuous data.</p>
{table(cov_tbl)}
<p><b>Key EDA findings.</b> Mean hourly PM2.5 is similar ({summ.loc['Hanoi', 'mean']:.1f} vs {summ.loc['HCMC', 'mean']:.1f} µg/m³). The WHO guideline
(15 µg/m³) applies to the 24-hour mean, so it is compared with daily means of days with at least 18 measured hours: it is exceeded on
{pct(summ.loc['Hanoi', 'share of days > WHO 24h'])} of {summ.loc['Hanoi', 'valid days']} valid days in Hanoi and {pct(summ.loc['HCMC', 'share of days > WHO 24h'])}
of {summ.loc['HCMC', 'valid days']} in HCMC. Hanoi is more volatile (std {summ.loc['Hanoi', 'std']:.1f} vs {summ.loc['HCMC', 'std']:.1f}) with a strong winter peak
(monthly mean {month_name[peak['Hanoi']]} {monthly['Hanoi'][peak['Hanoi']]:.1f} vs {month_name[low['Hanoi']]} {monthly['Hanoi'][low['Hanoi']]:.1f} µg/m³) and is
associated with cloud cover (Spearman {sp_['Hanoi']['cloud_cover']:+.2f}), rain ({sp_['Hanoi']['precipitation']:+.2f}) and pressure ({sp_['Hanoi']['surface_pressure']:+.2f});
HCMC ({month_name[peak['HCMC']]} {monthly['HCMC'][peak['HCMC']]:.1f} vs {month_name[low['HCMC']]} {monthly['HCMC'][low['HCMC']]:.1f}) has a 7h rush-hour peak and its
strongest link to wind speed ({sp_['HCMC']['wind_speed_10m']:+.2f}). The autocorrelation drops from {acf[1]['Hanoi']:.2f} / {acf[1]['HCMC']:.2f} at lag 1h to
{acf[24]['Hanoi']:.2f} / {acf[24]['HCMC']:.2f} at lag 24h, so a day-ahead forecast is much harder than a next-hour one. The data covers a single winter.</p>
{figure('daily', 1, 'Daily mean PM2.5 against the WHO 24h guideline (days with at least 18 measured hours). Both cities peak in the 2025–26 winter; Hanoi has the higher and more variable peak.')}
{figure('cycle', 2, f"Mean PM2.5 by hour of day, dry/cool vs wet/hot half-year, using only days with at least 18 measured hours. HCMC peaks at the 7h rush hour in both half-years ({cyc[('HCMC', 'Nov–Apr')][2]:.1f} µg/m³ in Nov–Apr); Hanoi in Nov–Apr is highest at {cyc[('Hanoi', 'Nov–Apr')][1]}h ({cyc[('Hanoi', 'Nov–Apr')][2]:.1f}) and lowest at {cyc[('Hanoi', 'Nov–Apr')][3]}h ({cyc[('Hanoi', 'Nov–Apr')][4]:.1f}), consistent with a shallow night-time boundary layer.")}

<h2>3. Data preparation</h2>
<ul>
<li><b>Cleaning:</b> values ≤ 0 or &gt; 1000 µg/m³ and "stuck sensor" runs (≥ 6 identical hours) are set to missing (none found in this period, {miss['Hanoi'][0]} / {miss['HCMC'][0]} invalid values).
Hanoi has {gaps.loc['Hanoi', 'Gaps']} gaps ({gaps.loc['Hanoi', 'Gaps ≤ 6h']} of them ≤ 6h, longest {gaps.loc['Hanoi', 'Longest gap (hours)']}h); HCMC {gaps.loc['HCMC', 'Gaps']} (longest {gaps.loc['HCMC', 'Longest gap (hours)']}h).</li>
<li><b>Missing values without look-ahead:</b> time interpolation would fill a gap with the value measured <i>after</i> it, which is unknown at forecast time.
So the <b>target is always a measured value</b> (never filled), and the features use a <b>causal forward-fill of at most 6h</b> — missing PM2.5 in the feature series
falls from {miss['Hanoi'][1]:.1f}% → {miss['Hanoi'][2]:.1f}% (Hanoi) and {miss['HCMC'][1]:.1f}% → {miss['HCMC'][2]:.1f}% (HCMC).
Samples without a current value or target are dropped ({r['samples']['Hanoi']:,} / {r['samples']['HCMC']:,} samples remain),
and remaining feature gaps use a median <code>SimpleImputer</code> fitted on the training data.</li>
<li><b>Outliers:</b> the IQR rule (k = 3) flags {outl['Hanoi'][0]} ({outl['Hanoi'][1]:.2f}%) and {outl['HCMC'][0]} ({outl['HCMC'][1]:.2f}%) hours. They are real pollution events, so they stay in the <i>target</i>;
only the PM2.5 <i>features</i> are clipped by a custom <code>IQRClipper</code> transformer learned on the training data.</li>
<li><b>Features (23):</b> PM2.5 lags (0–23h) and rolling statistics, weather at <i>t</i> plus 24h rain total, cyclic hour of the forecast time (numerical,
<code>StandardScaler</code>), and season, day of week and 8-sector wind direction (categorical, <code>OneHotEncoder</code>) — 39 columns after encoding.
Calendar features describe the target time <i>t</i> + 24h, which is known in advance, so they do not leak.</li>
<li><b>Split (same dates for both cities):</b> <b>Test</b> = samples issued from {d(split['Hanoi'][4])} ({split['Hanoi'][3]:,} per city);
<b>Development</b> = samples issued until {d(split['Hanoi'][2])} ({split['Hanoi'][0]:,} Hanoi / {split['HCMC'][0]:,} HCMC), ending 24h before Test so that no
Development target falls in the Test period. Inside Development, <b>walk-forward validation</b>: for each month <i>m</i> from 10/2025 to 08/2026, every model
is refitted on all samples whose target is measured before <i>m</i> starts (an expanding window) and forecasts the samples issued in <i>m</i>.
Each Development hour from 10/2025 on is forecast exactly once, by a model that never saw it, and the validation forecasts cover the whole winter.
All preprocessing lives in one <code>ColumnTransformer</code> + <code>Pipeline</code>, fitted only on the training data of each fit.</li>
</ul>
{figure('split', 3, f"Development (blue: training only; orange: walk-forward validation months, dotted lines = fold starts) and Test (green). Mean target: Hanoi {split['Hanoi'][6]:.1f} (Development) vs {split['Hanoi'][7]:.1f} (Test), HCMC {split['HCMC'][6]:.1f} vs {split['HCMC'][7]:.1f} µg/m³. The Test period is the start of the polluted season, not its peak.")}

<h2>4. Modeling and model selection</h2>
<p>Models: three naive baselines (<b>Persistence</b>, <b>24h mean</b>, <b>Naive blend</b>), <b>Ridge</b>, <b>RandomForest</b> and <b>HistGradientBoosting</b>.
The three ML models are tuned with <code>GridSearchCV</code> on <code>TimeSeriesSplit(n_splits=3, gap=24)</code> over the Development set (MAE), then kept fixed.
A <code>CityForecaster</code> class runs the whole flow for one city. Pooled walk-forward results over the 11 months (the selected model of each city in bold;
the Test set is never used for selection):</p>
{table(wf_tbl)}
<p>All three ML models beat the Naive blend on the pooled walk-forward forecasts, but by little, and their MAEs are within 2% of each other.
Cross-validation prefers simple models (shallow trees, at most 4 leaves per boosting tree), and several selected values lie at the edge of the grid even after
widening it twice — a sign that the 24h-ahead signal is weak. Because the hyperparameters were tuned on the whole Development set, the walk-forward
scores of the ML models are slightly optimistic; Test is not affected.</p>
{figure('wf', 4, f"Walk-forward MAE by month for every model (bars: mean measured PM2.5). In Hanoi, {r['best']['Hanoi']} is far better than the blend in the winter peak (January {mae_pair('Hanoi', '2026-01')}, February {mae_pair('Hanoi', '2026-02')} µg/m³) but worse in {', '.join(worse_h)}. In HCMC, {r['best']['HCMC']} is worse than the blend in {', '.join(worse_c)}.")}
<p><b>Why the ML gain depends on the month.</b> The bias (mean of forecast − actual) shows that the ML models lag behind changes of the pollution level.
When the level rises they forecast too low: Hanoi {r['best']['Hanoi']} has a bias of {bias_txt('Hanoi', '2025-11')} µg/m³ in November 2025
(mean measured {bh.loc['mean measured PM2.5', '2025-11']:.1f}) and {bias_txt('Hanoi', '2026-04')} in April 2026 ({bh.loc['mean measured PM2.5', '2026-04']:.1f}, up from
{bh.loc['mean measured PM2.5', '2026-03']:.1f} in March); HCMC {r['best']['HCMC']} {bias_txt('HCMC', '2025-10')} in October 2025. When the level falls they forecast
too high (HCMC February 2026: {bias_txt('HCMC', '2026-02')}). The Naive blend has no intercept and follows the current level, so its bias stays small.
In November 2025 the Hanoi model had only seen July–October, the cleaner months.</p>

<h2>5. Results on the Test set</h2>
<p>Each model is refitted on the whole Development set and evaluated <b>once</b> on Test ({d(split['Hanoi'][4])} – {d(split['Hanoi'][5])}).
The model chosen by walk-forward validation is shown in bold. "Gain" = MAE reduction relative to the reference (+ = better).</p>
{table(test_tbl, 'metrics')}
{figure('mae', 5, f"Test MAE per model and city (labels: MAE reduction vs the Naive blend; + = better). The selected models cut MAE by {b('Hanoi', 'vs_blend'):.1f}% in Hanoi ({b('Hanoi', 'model')}) and {b('HCMC', 'vs_blend'):.1f}% in HCMC ({b('HCMC', 'model')}) relative to the blend. On Test, Ridge is the best model in Hanoi although it was not selected — the ML models are too close for the selection to be reliable.")}
<p><b>Are the differences real?</b> Forecast errors are strongly autocorrelated, so an ordinary bootstrap over hours would be far too optimistic.
We use a <b>paired non-overlapping block bootstrap</b> (Carlstein, 1986): the forecasts (pooled walk-forward, or Test) are grouped into 3-day calendar blocks
({blocks['Walk-forward']} / {blocks['Test']} blocks), whole blocks are resampled with replacement 2,000 times, and the MAE gain 1 − MAE<sub>A</sub>/MAE<sub>B</sub>
of the best model over each competitor is recomputed on the same hours; the 2.5th–97.5th percentiles give a 95% percentile interval. Bold = the interval excludes 0.
The intervals are descriptive (no multiple-comparison correction): the conclusions rest on the four comparisons fixed in advance, best model vs Naive blend
on walk-forward and on Test for each city.</p>
{table(boot_tbl)}
{figure('boot', 6, f"MAE gain of the best model of each city over every competitor, with 95% block-bootstrap intervals (hollow = walk-forward, filled = Test). Best model vs Naive blend by block length — {sens_txt}.")}
{figure('forecast', 7, f"First 14 days of the Test set (time axis = forecast time). MAE in this window — {win_txt}. The forecasts are much smoother than reality (std of forecast vs actual, µg/m³: {std_txt}), so short spikes are missed.")}
<p><b>Answer to question 1.</b> Against plain Persistence, machine learning gains {b('Hanoi', 'reduction'):.1f}% / {b('HCMC', 'reduction'):.1f}% on Test, significant in both
cities on both evaluation sets. Part of that comes from simply smoothing the current value: against the best naive forecast the gain is
<b>{b('Hanoi', 'vs_blend'):.1f}% in Hanoi</b>, which is not distinguishable from zero (walk-forward: {ci('Hanoi', 'Walk-forward', BLEND)}, borderline depending on the block length),
and <b>{b('HCMC', 'vs_blend'):.1f}% in HCMC</b>, significant on Test but not on walk-forward ({ci('HCMC', 'Walk-forward', BLEND)}). All point estimates are positive,
so the most likely situation is a small real gain of a few percent that one year of data cannot pin down. Which ML model is best is not established in either city.</p>

<h2>6. Error analysis and drivers</h2>
<p>Actual and forecast PM2.5 are put into concentration bands defined by the US EPA 2024 PM2.5 breakpoints (Good ≤ 9.0, Moderate ≤ 35.4, USG ≤ 55.4,
Unhealthy ≤ 125.4 µg/m³; values truncated to 0.1 first) to build a band confusion matrix. The official AQI uses a 24-hour average, so these hourly
bands are <b>not</b> official AQI values.</p>
{figure('aqi', 8, f"EPA-band confusion matrix of the best model on Test. Correct band: {aqi['Hanoi'][0]:.1f}% (Hanoi) and {aqi['HCMC'][0]:.1f}% (HCMC); within ±1 band: {aqi['Hanoi'][1]:.1f}% / {aqi['HCMC'][1]:.1f}% (walk-forward: {r['aqi_wf']['Hanoi']:.1f}% / {r['aqi_wf']['HCMC']:.1f}% correct). Most errors are polluted hours forecast as 'Moderate', and 'Good' is never predicted.")}
<p>The main weakness is <b>under-forecasting polluted hours</b>: forecasts are pulled towards the mean, so the model follows the general level but is not a
reliable early-warning tool. By hour of the forecast time, the error is highest in the early morning (Hanoi {err[('Hanoi', 'Test')][2]:.1f} µg/m³ at {err[('Hanoi', 'Test')][3]}h on Test,
HCMC {err[('HCMC', 'Walk-forward')][2]:.1f} at {err[('HCMC', 'Walk-forward')][3]}h on walk-forward), when night-time accumulation and the morning rush hour change the level fastest.</p>
{figure('importance', 9, f"Permutation importance on Test (MAE increase when a feature is shuffled). Hanoi: {top_txt['Hanoi']}. HCMC: {top_txt['HCMC']}. Correlated PM2.5 lag features share their importance, so groups are also shuffled together — Hanoi: {grp_txt['Hanoi']}; HCMC: {grp_txt['HCMC']}.")}
<p><b>Answer to question 2.</b> Recent PM2.5 is the most important group in both cities. Hanoi then depends on weather (pressure, rain, wind), while the
calendar adds nothing on the Test period; HCMC depends on the hour of day (daily traffic rhythm, 7h peak in the EDA). Hanoi is the harder city in absolute terms
(higher volatility and winter peaks, walk-forward MAE {wf.loc[wf.model == r['best']['Hanoi'], 'MAE Hanoi'].iloc[0]:.2f} vs {wf.loc[wf.model == r['best']['HCMC'], 'MAE HCMC'].iloc[0]:.2f} µg/m³),
and in both cities the relative gain over the naive forecast is small.</p>

<h2>7. Conclusions, limitations and future work</h2>
<ul>
<li><b>Conclusion:</b> one reusable pipeline, trained independently per city and selected by walk-forward validation, lowers the Test MAE of the strongest
naive forecast by {ci('Hanoi', 'Test', BLEND)} in Hanoi and {ci('HCMC', 'Test', BLEND)} in HCMC (gain [95% block-bootstrap CI]), and of Persistence by
{b('Hanoi', 'reduction'):.1f}% / {b('HCMC', 'reduction'):.1f}% (both significant). The gain is concentrated in the stable winter peak; in months where the level
changes, the ML models lag behind the naive forecast.</li>
<li><b>Limitations:</b> a single winter (the Hanoi station started reporting in 07/2025), so winter performance rests on the walk-forward months only and the
Test period covers the start of the polluted season, not its peak; one station per city (HCMC is a low-cost sensor and the only HCMC station with continuous data);
hyperparameters tuned on the whole Development set (walk-forward scores slightly optimistic) with several values at the grid edge; weather features are
observations at <i>t</i>, not forecasts for <i>t</i> + 24h; polluted hours are under-forecast.</li>
<li><b>Future work:</b> repeat the walk-forward evaluation after a second winter (from 12/2026); add weather forecasts and more stations; add features that react
to level changes; quantile regression or an exceedance classifier for early warning; nested tuning inside each walk-forward fold.</li>
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
