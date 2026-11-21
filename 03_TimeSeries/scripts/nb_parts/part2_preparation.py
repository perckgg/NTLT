# Section 2: data preparation
md("""
## 2. DATA PREPARATION

The steps below are written as **reusable functions** and applied **identically** to each city,
but every city is processed **independently**.

### 2.1. Missing value statistics
""")

code("""
def gap_lengths(series):
    \"\"\"Length (hours) of each run of consecutive missing values.\"\"\"
    is_na = series.isna()
    groups = (is_na != is_na.shift()).cumsum()
    return is_na.groupby(groups).sum()[lambda s: s > 0]


def missing_report(raw_dict):
    rows = []
    for city, df in raw_dict.items():
        gaps = gap_lengths(df['pm25'])
        rows.append({'City': city, 'Missing hours': int(df['pm25'].isna().sum()),
                     'Missing rate': f"{df['pm25'].isna().mean():.1%}",
                     'Gaps': len(gaps), 'Gaps ≤ 6h': int((gaps <= 6).sum()),
                     'Longest gap (hours)': int(gaps.max()) if len(gaps) else 0})
    return pd.DataFrame(rows)


display(missing_report(raw))
display((pd.concat({city: df.isna().mean() for city, df in raw.items()}, axis=1) * 100)
        .round(2).rename(columns=lambda c: f'{c} (% missing)'))

fig, ax = plt.subplots(figsize=(14, 4))
for city, df in raw.items():
    avail = df['pm25'].notna().resample('MS').mean() * 100
    ax.plot(avail.index, avail, marker='o', label=city, color=CITY_COLORS[city])
ax.set(title='Share of hours with PM2.5 data, by month', ylabel='% of hours with data', ylim=(0, 105))
ax.legend()
plt.show()

missing_by_month = pd.DataFrame({city: df['pm25'].isna().resample('MS').mean() * 100 for city, df in raw.items()})
missing_by_month.index = missing_by_month.index.strftime('%Y-%m')
print('% of hours with missing PM2.5, by month:')
display(missing_by_month.round(0).astype(int).T)
""")

md("""
### 2.2. Cleaning invalid values and handling missing data

- **Invalid values:** PM2.5 ≤ 0 or > 1000 µg/m³ are sensor errors → set to `NaN`
  (a reference monitor reporting exactly 0.0 is a dropout, not clean air).
- **"Stuck" sensor:** the same value repeated for ≥ 6 consecutive hours (with PM2.5 > 0) indicates a frozen device → set to `NaN`.
- **No look-ahead when filling.** Time interpolation (`interpolate(method='time')`) fills a gap using the value measured **after**
  the gap, which is not known yet at forecast time $t$. If the features were built from interpolated values, the model would
  see up to 6 hours of the future (Section 2.4 prints how many samples this would concern). We therefore keep two versions:
  - `pm25_raw` — the measured values only. **The target is always a measured value** (never filled), so the evaluation is done on real data.
  - `pm25` — `pm25_raw` **forward-filled for at most 6 hours** (`ffill(limit=6)`): "the last known value, if it is at most 6h old".
    This is causal (uses only data ≤ $t$) and is used to build the features.
- **Long gaps (> 6 hours):** not filled (that would fabricate data) → samples whose current value or target is missing are dropped;
  remaining missing features are handled by `SimpleImputer` (median, fitted on the training set) inside the pipeline.
""")

code("""
MAX_FILL_GAP = 6
STUCK_HOURS = 6


def clean_city(df):
    df = df.copy()
    pm = df['pm25']
    invalid = (pm <= 0) | (pm > 1000)
    run_id = (pm != pm.shift()).cumsum()
    run_len = pm.groupby(run_id).transform('size')
    stuck = (run_len >= STUCK_HOURS) & (pm > 0) & pm.notna()
    df.loc[invalid | stuck, 'pm25'] = np.nan
    df['pm25_raw'] = df['pm25']                                # measured values only (used for the target)
    df['pm25'] = df['pm25_raw'].ffill(limit=MAX_FILL_GAP)      # causal fill (used for the features)
    return df, int(invalid.sum()), int(stuck.sum())


clean = {}
for city, df in raw.items():
    clean[city], n_invalid, n_stuck = clean_city(df)
    before, after = df['pm25'].isna().mean(), clean[city]['pm25'].isna().mean()
    print(f'{city}: {n_invalid} values ≤ 0 or > 1000, {n_stuck} stuck-sensor hours | '
          f'missing PM2.5: {before:.1%} → {after:.1%} after cleaning + causal forward-fill (features only)')


def first_short_gap(series, min_len=3, max_len=MAX_FILL_GAP):
    is_na = series.isna()
    groups = (is_na != is_na.shift()).cumsum()
    sizes = is_na.groupby(groups).transform('sum')
    mask = is_na & (sizes >= min_len) & (sizes <= max_len)
    return series.index[mask][0] if mask.any() else None


for city in clean:
    gap_time = first_short_gap(clean[city]['pm25_raw'])
    if gap_time is not None:
        break
if gap_time is not None:
    window = clean[city].loc[gap_time - pd.Timedelta('36h'): gap_time + pd.Timedelta('36h')]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(window.index, window['pm25'], where='post', ls='--', color='orange', label='Forward-filled (feature input)')
    ax.plot(window.index, window['pm25_raw'], 'o-', color=CITY_COLORS[city], label='Measured (pm25_raw, target)')
    ax.set(title=f'Causal forward-fill of a short gap – {city}', ylabel='PM2.5 (µg/m³)')
    ax.legend()
    plt.show()
""")

md("""
### 2.3. Outlier detection (IQR)
IQR rule: values outside $[Q1 - 3\\cdot IQR,\\ Q3 + 3\\cdot IQR]$ are treated as extreme outliers.
""")

code("""
def iqr_bounds(s, k=3.0):
    q1, q3 = s.quantile([0.25, 0.75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, (city, df) in zip(axes, clean.items()):
    lo, hi = iqr_bounds(df['pm25_raw'])
    out = df.loc[df['pm25_raw'] > hi, 'pm25_raw']
    top_days = out.resample('D').count().nlargest(3).index.strftime('%Y-%m-%d').tolist()
    print(f'{city}: upper bound = {hi:.1f} µg/m³ | {len(out)} outlier hours '
          f'({len(out) / df.pm25_raw.notna().sum():.2%}) | days with most outliers: {top_days}')
    ax.plot(df.index, df['pm25_raw'], lw=0.4, color=CITY_COLORS[city])
    ax.scatter(out.index, out, s=6, color='k', label='Outlier (IQR, k=3)')
    ax.axhline(hi, color='k', ls='--', lw=1)
    ax.set(title=city, ylabel='PM2.5 (µg/m³)')
    ax.legend()
plt.tight_layout()
plt.show()
""")

md("""
**Outlier decision:** PM2.5 peaks are **real events** (winter pollution episodes, straw burning, fireworks...),
so they are **not removed** from the target (removing them would make the evaluation look better than it is). Instead, the PM2.5
**features** (lags, rolling statistics) go through a custom `IQRClipper` transformer (bounds learned **on the training set only**)
to reduce the influence of extreme values on the linear model.
""")

md("""
### 2.4. Feature engineering

Each sample at time $t$ has target `y = pm25_raw(t + 24h)` (a **measured** value). The features are built from the forward-filled
series and use only information **available at $t$** (no future leakage):

| Group | Features | Type |
|---|---|---|
| PM2.5 history | `pm25_lag_0, 1, 2, 3, 6, 12, 23` (value at $t-k$), `pm25_roll_mean_6/24`, `pm25_roll_std_24`, `pm25_diff_24` | numerical |
| Weather at $t$ | temperature, humidity, precipitation, wind speed, pressure, cloud cover, `rain_24h` (total rain over the last 24h) | numerical |
| Calendar of the forecast time | `hour_sin/cos` (cyclic hour encoding) | numerical |
| | `season` (climate season of each city), `day_of_week`, `wind_sector` (8 wind directions) | **categorical** |

> `pm25_lag_0` is the value at the **same hour on the previous day** relative to the forecast time, so the
> **Persistence** baseline (`ŷ(t+24) = pm25(t)`) is also the 24h *seasonal naive* baseline.
""")

code("""
SEASONS = {
    'Hanoi': {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn'},
    'HCMC': {m: ('Dry season' if m in (12, 1, 2, 3, 4) else 'Wet season') for m in range(1, 13)},
}
WIND_SECTORS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
LAGS = [0, 1, 2, 3, 6, 12, 23]
WEATHER_RAW = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m',
               'surface_pressure', 'cloud_cover']
WEATHER_FEATURES = WEATHER_RAW + ['rain_24h']
PM_FEATURES = [f'pm25_lag_{k}' for k in LAGS] + ['pm25_roll_mean_6', 'pm25_roll_mean_24',
                                                 'pm25_roll_std_24', 'pm25_diff_24']
CYCLIC_FEATURES = ['hour_sin', 'hour_cos']
CAT_FEATURES = ['season', 'day_of_week', 'wind_sector']
FEATURES = PM_FEATURES + WEATHER_FEATURES + CYCLIC_FEATURES + CAT_FEATURES
TARGET = 'target'


def build_features(df, city, horizon=HORIZON):
    \"\"\"Build the features at each time t (using only data ≤ t).\"\"\"
    pm = df['pm25']
    X = pd.DataFrame(index=df.index)
    for k in LAGS:
        X[f'pm25_lag_{k}'] = pm.shift(k)
    X['pm25_roll_mean_6'] = pm.rolling(6, min_periods=3).mean()
    X['pm25_roll_mean_24'] = pm.rolling(24, min_periods=12).mean()
    X['pm25_roll_std_24'] = pm.rolling(24, min_periods=12).std()
    X['pm25_diff_24'] = pm - pm.shift(24)
    X[WEATHER_RAW] = df[WEATHER_RAW]
    X['rain_24h'] = df['precipitation'].rolling(24, min_periods=1).sum()

    target_time = df.index + pd.Timedelta(hours=horizon)
    X['hour_sin'] = np.sin(2 * np.pi * target_time.hour / 24)
    X['hour_cos'] = np.cos(2 * np.pi * target_time.hour / 24)
    X['season'] = pd.Series(target_time.month, index=df.index).map(SEASONS[city])
    X['day_of_week'] = target_time.day_name()
    sector_idx = ((df['wind_direction_10m'] + 22.5) % 360) // 45
    X['wind_sector'] = sector_idx.map(lambda i: WIND_SECTORS[int(i)] if pd.notna(i) else np.nan)
    return X


def make_dataset(df, city, horizon=HORIZON):
    \"\"\"Add target = measured pm25_raw(t + horizon); keep only samples with a target and a current value.\"\"\"
    X = build_features(df, city, horizon)
    X[TARGET] = df['pm25_raw'].shift(-horizon)
    return X.dropna(subset=[TARGET, 'pm25_lag_0'])


datasets = {city: make_dataset(df, city) for city, df in clean.items()}
for city, d in datasets.items():
    filled_now = clean[city]['pm25_raw'].reindex(d.index).isna().mean()   # current value comes from the forward-fill
    print(f'{city}: {len(d):,} samples | {len(FEATURES)} features '
          f'({len(FEATURES) - len(CAT_FEATURES)} numerical + {len(CAT_FEATURES)} categorical) | '
          f'{filled_now:.1%} of samples have a forward-filled current value '
          f'(with two-sided interpolation these would have used future measurements)')
datasets['Hanoi'].head()
""")

md("""
### 2.5. Chronological split: Development / Test, and walk-forward folds
A time series **must not be shuffled** (that would leak the future into the past), so every split follows **time order** and uses
the **same calendar dates for both cities**, so the two cities are compared over the same weather and season.

- **Test** = samples issued from **01/09/2026** to the end of the data (the last 2.5 months). It is used **once**, at the very end (Section 4).
- **Development** = all samples issued **before** 01/09/2026 − 24h. The extra `HORIZON` = 24 hours makes sure that the **target**
  ($t + 24h$) of every Development sample is measured before the first Test forecast is issued.

The data covers only one winter, and a single fixed Validation month (or 15% block) would fall in one season: a model would then be
chosen on, say, clean summer hours and judged on the polluted season. Model selection therefore uses **walk-forward validation**
(also called evaluation on a rolling forecasting origin, Tashman 2000; Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, §5.10):
for every month $m$ from **10/2025 to 08/2026** (11 folds), with $s_m$ the first hour of month $m$,

$$\\text{Train}_m = \\{t \\in \\text{Dev} : t < s_m - 24h\\}, \\qquad \\text{Val}_m = \\{t \\in \\text{Dev} : s_m \\le t < s_{m+1}\\}.$$

Each model is refitted on $\\text{Train}_m$ (an **expanding** window: everything known before the month) and forecasts every sample
of $\\text{Val}_m$. The 11 validation months do not overlap, so each Development hour from 10/2025 onwards is forecast **exactly once**,
by a model that never saw it, and the validation set covers the whole winter 2025–26. The first three months (07–09/2025) are only
used for training, so the first fold has a minimum amount of history.
""")

code("""
TEST_START = pd.Timestamp('2026-09-01')
WALK_FORWARD_MONTHS = pd.period_range('2025-10', '2026-08', freq='M')


def dev_test_split(d, test_start=TEST_START, gap_hours=HORIZON):
    \"\"\"Development = samples whose target is measured before test_start; Test = samples issued from test_start.\"\"\"
    dev = d[d.index < test_start - pd.Timedelta(hours=gap_hours)]
    test = d[d.index >= test_start]
    return dev, test


def walk_forward_folds(dev, months=WALK_FORWARD_MONTHS, gap_hours=HORIZON):
    \"\"\"For each month m: (m, train = samples whose target is known before m starts, val = samples issued in m).\"\"\"
    folds = []
    for m in months:
        start, end = m.start_time, (m + 1).start_time
        train = dev[dev.index < start - pd.Timedelta(hours=gap_hours)]
        val = dev[(dev.index >= start) & (dev.index < end)]
        folds.append((m, train, val))
    return folds


splits = {city: dev_test_split(d) for city, d in datasets.items()}

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
for ax, (city, (dev, te)) in zip(axes, splits.items()):
    pre_wf = dev[dev.index < WALK_FORWARD_MONTHS[0].start_time]
    wf = dev[dev.index >= WALK_FORWARD_MONTHS[0].start_time]
    for part, name, color in [(pre_wf, 'Development – training only', 'tab:blue'),
                              (wf, 'Development – walk-forward validation months', 'tab:orange'),
                              (te, 'Test', 'tab:green')]:
        daily = part[TARGET].resample('D').mean()
        ax.plot(daily.index, daily, color=color, lw=0.8,
                label=f'{name}: {len(part):,} samples ({part.index.min():%d/%m/%Y} – {part.index.max():%d/%m/%Y})')
    for m in WALK_FORWARD_MONTHS:
        ax.axvline(m.start_time, color='gray', lw=0.5, ls=':')
    ax.set(title=f'{city} (dotted lines: start of each walk-forward fold)', ylabel='Daily mean target PM2.5 (µg/m³)')
    ax.legend(loc='upper left', fontsize=9)
    print(f'{city} split: Development {len(dev):,} ({dev.index.min():%Y-%m-%d} → {dev.index.max():%Y-%m-%d %H:%M}) | '
          f'Test {len(te):,} ({te.index.min():%Y-%m-%d} → {te.index.max():%Y-%m-%d %H:%M}) samples | '
          f'mean target: Development {dev[TARGET].mean():.1f}, Test {te[TARGET].mean():.1f} µg/m³')
plt.tight_layout()
plt.show()

rows = []
for city, (dev, te) in splits.items():
    for m, tr, va in walk_forward_folds(dev):
        rows.append({'city': city, 'month': str(m), 'train': len(tr), 'validation': len(va)})
fold_sizes = pd.DataFrame(rows).pivot(index='month', columns='city', values=['train', 'validation'])
print('Walk-forward folds: number of samples in the training window and in the validation month')
display(fold_sizes)
""")

md(note("SPLIT"))

md("""
### 2.6. Preprocessing pipeline (imputation, outlier clipping, encoding, scaling)

| Column group | Steps |
|---|---|
| PM2.5 lags/rolling | `SimpleImputer(median)` → `IQRClipper(k=3)` → `StandardScaler` |
| Weather + hour cycle | `SimpleImputer(median)` → `StandardScaler` |
| Categorical | `SimpleImputer(most_frequent)` → `OneHotEncoder` |

Everything lives inside a `ColumnTransformer` + `Pipeline`, so all parameters (medians, IQR bounds, mean/std, categories)
are **learned on the data the pipeline is fitted on** (the training window of each fold, or the whole Development set
for the final model), never on the data it is evaluated on → no data leakage.
""")

code("""
class IQRClipper(BaseEstimator, TransformerMixin):
    \"\"\"Clip (winsorize) each column to [Q1 - k·IQR, Q3 + k·IQR], bounds learned on the fitted data.\"\"\"

    def __init__(self, k=3.0):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        self.lower_, self.upper_ = q1 - self.k * (q3 - q1), q3 + self.k * (q3 - q1)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor():
    return ColumnTransformer([
        ('pm', Pipeline([('impute', SimpleImputer(strategy='median')),
                         ('clip', IQRClipper(k=3.0)),
                         ('scale', StandardScaler())]), PM_FEATURES),
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), WEATHER_FEATURES + CYCLIC_FEATURES),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CAT_FEATURES),
    ], sparse_threshold=0)


tr = splits['Hanoi'][0]   # Development set of Hanoi
pre = build_preprocessor().fit(tr[FEATURES])
Xt = pd.DataFrame(pre.transform(tr[FEATURES]), columns=pre.get_feature_names_out(), index=tr.index)
print(f'After preprocessing: {Xt.shape[1]} columns (from {len(FEATURES)} original columns, one-hot expands the categoricals)')
print('Missing values left after the pipeline:', int(Xt.isna().sum().sum()))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(tr['pm25_lag_0'], bins=80, ax=axes[0], color='gray')
axes[0].set(title='pm25_lag_0 – before preprocessing (Hanoi, Development)', xlabel='µg/m³')
sns.histplot(Xt['pm__pm25_lag_0'], bins=80, ax=axes[1], color='teal')
axes[1].set(title='After IQRClipper + StandardScaler', xlabel='z-score')
plt.tight_layout()
plt.show()
""")
