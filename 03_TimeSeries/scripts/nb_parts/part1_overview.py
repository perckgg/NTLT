# Sections 0 + 1: header, environment, data description & EDA
md("""
# COURSE PROJECT: PROGRAMMING FOUNDATIONS FOR DATA ANALYSIS AND VISUALIZATION
**Semester 261 – Academic Year 2026–2027** | **Instructor:** Lê Thành Sách

---
## PART 3: TIME SERIES DATA
### Topic: 24-Hour-Ahead PM2.5 Forecasting — Hanoi vs Ho Chi Minh City

- **Group:** IndexError
- **Member in charge:** Nguyễn Văn Hoàng Khang – 2570754

**Problem.** At time $t$, using the PM2.5 history and the current weather, forecast the PM2.5 concentration
at time $t + 24h$ (the same hour tomorrow). This is a **time series regression** problem.

**Why compare two cities?** Hanoi (four seasons, winter temperature inversion) and Ho Chi Minh City (HCMC; wet/dry seasons)
have different pollution mechanisms. We apply **the same pipeline** to each city **independently** (no merging) to answer:
*how much does machine learning improve over a naive baseline in each city, and which city is harder to forecast?*
""")

md("## 0. ENVIRONMENT SETUP")

code("""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

RANDOM_STATE = 42
HORIZON = 24            # forecast 24 hours ahead
WHO_24H = 15            # WHO 2021 guideline for 24-hour mean PM2.5 (µg/m³)
CITY_COLORS = {'Hanoi': '#d62728', 'HCMC': '#1f77b4'}
import sklearn
print(f'Environment ready! (scikit-learn {sklearn.__version__}, pandas {pd.__version__})')
""")

md("""
## 1. DATASET DESCRIPTION

### 1.1. Data sources (self-collected)
The data was **collected through public APIs** with the script `scripts/crawl_air_quality.py`:

| Component | Source | Notes |
|---|---|---|
| PM2.5 – Hanoi | [OpenAQ v3](https://openaq.org) – station *46 Luu Quang Vu* (Hanoi Air Quality Monitoring Network, sensor 7772032) | City monitoring station, **has real gaps** |
| PM2.5 – HCMC | OpenAQ v3 – station *CMT8* (AirGradient, sensor 11357424) | Low-cost sensor |
| Weather | [Open-Meteo Historical Weather API](https://open-meteo.com) | Temperature, humidity, precipitation, wind, pressure, cloud cover |

- **Period:** 01/12/2024 – 30/09/2026 (22 months, 16,056 hours per city), Vietnam local time (UTC+7).
- A complete hourly time grid is built: hours with no measurement become `NaN`.

| Column | Type | Meaning |
|---|---|---|
| `datetime` | datetime | Timestamp (local time) |
| `city` | categorical | City |
| `temperature_2m` | numerical | Temperature at 2 m (°C) |
| `relative_humidity_2m` | numerical | Relative humidity (%) |
| `precipitation` | numerical | Precipitation (mm) |
| `wind_speed_10m` | numerical | Wind speed at 10 m (km/h) |
| `wind_direction_10m` | numerical (angle) | Wind direction (degrees) → converted to a **categorical** 8-sector feature |
| `surface_pressure` | numerical | Surface pressure (hPa) |
| `cloud_cover` | numerical | Cloud cover (%) |
| `pm25` | numerical | PM2.5 concentration (µg/m³) — **the variable to forecast** |
| `pm25_source` | categorical | PM2.5 source (`openaq`, or `open-meteo` if the crawler had to fall back) |

### 1.2. Loading the data
The CSV files are read locally; on Colab, if they are not found, they are downloaded from GitHub (**set `GITHUB_RAW_BASE`** to the group's repo).
""")

code("""
GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/perckgg/NTLT/main/03_TimeSeries/data/timeseries'
CITY_FILES = {'Hanoi': 'air_quality_hanoi.csv', 'HCMC': 'air_quality_hcmc.csv'}
LOCAL_DIRS = ['data/timeseries', '../data/timeseries', '/content/data/timeseries', '/content']


def load_city_data(filename):
    for d in LOCAL_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=['datetime'])
    return pd.read_csv(f'{GITHUB_RAW_BASE}/{filename}', parse_dates=['datetime'])


raw = {city: load_city_data(f).set_index('datetime').sort_index().assign(city=city)
       for city, f in CITY_FILES.items()}
for city, df in raw.items():
    print(f'{city}: {df.shape[0]:,} rows × {df.shape[1]} columns | '
          f'{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d} | PM2.5 source: {", ".join(df.pm25_source.unique())}')
raw['Hanoi'].head()
""")

md("""
### 1.3. Descriptive statistics
The WHO guideline of 15 µg/m³ is defined for the **24-hour mean**, so it is compared with **daily means**, not with single hours.
A daily mean is computed only for days with at least 18 of 24 hours measured (75% completeness, the usual rule for daily averages);
other days are left out rather than averaged over a few hours.
""")

code("""
MIN_HOURS_PER_DAY = 18   # 75% of 24 hours


def daily_means(s, min_hours=MIN_HOURS_PER_DAY):
    \"\"\"Daily mean of an hourly series; NaN for days with fewer than `min_hours` measured hours.\"\"\"
    days = s.resample('D')
    return days.mean().where(days.count() >= min_hours)


raw['Hanoi'].info()
summary = pd.concat({city: df['pm25'].describe() for city, df in raw.items()}, axis=1).T
summary['missing_rate'] = [df['pm25'].isna().mean() for df in raw.values()]
daily = {city: daily_means(df['pm25']).dropna() for city, df in raw.items()}
summary['valid days'] = [len(d) for d in daily.values()]
summary['share of days > WHO 24h'] = [(d > WHO_24H).mean() for d in daily.values()]
display(summary.round(3))
display(pd.concat({city: df.describe().T for city, df in raw.items()}).round(2))
""")

md("""
### 1.4. Time series visualization
**Figure 1 – Daily mean PM2.5** in both cities against the WHO 24-hour guideline (15 µg/m³).
""")

code("""
fig, ax = plt.subplots(figsize=(14, 5))
for city, df in raw.items():
    d = daily_means(df['pm25'])          # NaN (gap in the line) on days with < 18 measured hours
    ax.plot(d.index, d, label=city, color=CITY_COLORS[city], lw=0.9)
ax.axhline(WHO_24H, color='k', ls='--', lw=1, label='WHO 24h guideline (15 µg/m³)')
ax.set(title='Daily mean PM2.5', ylabel='PM2.5 (µg/m³)')
ax.legend()
plt.show()
""")

md("**Figure 2 – Distribution of hourly PM2.5** and **Figure 3 – Monthly boxplots** (seasonality).")

code("""
plot_df = pd.concat([df[['pm25']].assign(city=city) for city, df in raw.items()])
plot_df['month'] = plot_df.index.month
plot_df['hour'] = plot_df.index.hour
plot_df = plot_df.reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 4))
sns.histplot(data=plot_df, x='pm25', hue='city', bins=100, element='step', stat='density',
             common_norm=False, palette=CITY_COLORS, ax=ax)
ax.set(title='Distribution of hourly PM2.5', xlabel='PM2.5 (µg/m³)', xlim=(0, 250))
plt.show()

fig, ax = plt.subplots(figsize=(14, 5))
sns.boxplot(data=plot_df, x='month', y='pm25', hue='city', palette=CITY_COLORS, showfliers=False, ax=ax)
ax.set(title='PM2.5 by month (box = median and quartiles; outliers hidden)', xlabel='Month', ylabel='PM2.5 (µg/m³)')
plt.show()

monthly = plot_df.groupby(['city', 'month'])['pm25'].agg(['mean', 'median']).round(1).unstack('city')
print('Hourly PM2.5 by calendar month (mean and median over all years, µg/m³):')
display(monthly)
""")

md("""
**Figure 4 – Daily cycle:** mean PM2.5 by hour of day, split into the dry/cool half-year (Nov–Apr) and the wet/hot half-year (May–Oct).
Only **days with at least 18 measured hours** are used. Reason: from November 2025 the Hanoi station rarely reports in the afternoon,
so a plain mean by hour would average the afternoon hours over *different days* (mostly the cleaner, earlier period) than the night hours,
and the shape of the curve would reflect *which days have data* instead of the real daily cycle.
""")

code("""
def complete_days_only(s, min_hours=MIN_HOURS_PER_DAY):
    \"\"\"Keep only the hours of days that have at least `min_hours` measured hours.\"\"\"
    measured_per_day = s.groupby(s.index.floor('D')).transform('count')
    return s[measured_per_day >= min_hours]


cycle_df = pd.concat([complete_days_only(df['pm25']).to_frame().assign(city=city) for city, df in raw.items()])
cycle_df['hour'] = cycle_df.index.hour
cycle_df['half_year'] = np.where(cycle_df.index.month.isin([11, 12, 1, 2, 3, 4]), 'Nov–Apr', 'May–Oct')

fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for ax, city in zip(axes, CITY_FILES):
    g = cycle_df[cycle_df.city == city]
    sns.lineplot(data=g, x='hour', y='pm25', hue='half_year', errorbar=None, ax=ax)
    ax.set(title=city, xlabel='Hour of day', ylabel='Mean PM2.5 (µg/m³)', xticks=range(0, 24, 3))
plt.tight_layout()
plt.show()

for city in CITY_FILES:
    for half, g in cycle_df[cycle_df.city == city].groupby('half_year'):
        by_hour = g.groupby('hour')['pm25'].mean()
        print(f'{city} {half} ({g.index.floor("D").nunique()} complete days): highest mean at {by_hour.idxmax()}h '
              f'({by_hour.max():.1f} µg/m³), lowest at {by_hour.idxmin()}h ({by_hour.min():.1f} µg/m³)')
""")

md("**Figure 5 – Spearman correlation matrix** between PM2.5 and the weather variables, per city.")

code("""
num_cols = ['pm25', 'temperature_2m', 'relative_humidity_2m', 'precipitation',
            'wind_speed_10m', 'surface_pressure', 'cloud_cover']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (city, df) in zip(axes, raw.items()):
    sns.heatmap(df[num_cols].corr(method='spearman'), annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, ax=ax, cbar=False)
    ax.set_title(f'Spearman correlation – {city}')
    print(f'{city} – Spearman correlation of PM2.5 with:',
          df[num_cols].corr(method='spearman')['pm25'].drop('pm25').round(2).to_dict())
plt.tight_layout()
plt.show()
""")

md("""
**Autocorrelation.** How much does the current value tell us about the value $k$ hours later? The lag-$k$ autocorrelation is the
Pearson correlation between the pairs $(x_t, x_{t-k})$ for all $t$ where both hours were measured
(`Series.autocorr(k)` in pandas). Lag 24h is the relevant one for a 24-hour-ahead forecast.
""")

code("""
ACF_LAGS = [1, 2, 3, 6, 12, 24, 48]
acf_table = pd.DataFrame({city: [df['pm25'].autocorr(lag) for lag in ACF_LAGS] for city, df in raw.items()},
                         index=pd.Index(ACF_LAGS, name='lag (hours)'))
display(acf_table.round(2))
""")

md("**EDA findings:**\n\n" + note("EDA"))
