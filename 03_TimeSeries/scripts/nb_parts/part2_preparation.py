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
""")

md("""
### 2.2. Cleaning invalid values and handling missing data

- **Invalid values:** negative PM2.5 or > 1000 µg/m³ are sensor errors → set to `NaN`.
- **"Stuck" sensor:** the same value repeated for ≥ 6 consecutive hours (with PM2.5 > 0) indicates a frozen device → set to `NaN`.
- **Short gaps (≤ 6 hours):** time-based linear interpolation (`interpolate(method='time', limit=6)`) —
  reasonable because PM2.5 changes smoothly over a few hours.
- **Long gaps (> 6 hours):** **not** interpolated (that would fabricate data) → samples whose *target* is missing are dropped;
  remaining missing features are handled by `SimpleImputer` (median, fitted on the training set) inside the pipeline.
""")

code("""
MAX_INTERP_GAP = 6
STUCK_HOURS = 6


def clean_city(df):
    df = df.copy()
    pm = df['pm25']
    invalid = (pm < 0) | (pm > 1000)
    run_id = (pm != pm.shift()).cumsum()
    run_len = pm.groupby(run_id).transform('size')
    stuck = (run_len >= STUCK_HOURS) & (pm > 0) & pm.notna()
    df.loc[invalid | stuck, 'pm25'] = np.nan
    df['pm25_raw'] = df['pm25']
    df['pm25'] = df['pm25'].interpolate(method='time', limit=MAX_INTERP_GAP, limit_area='inside')
    return df, int(invalid.sum()), int(stuck.sum())


clean = {}
for city, df in raw.items():
    clean[city], n_invalid, n_stuck = clean_city(df)
    before, after = df['pm25'].isna().mean(), clean[city]['pm25'].isna().mean()
    print(f'{city}: {n_invalid} negative/>1000 values, {n_stuck} stuck-sensor hours | '
          f'missing PM2.5: {before:.1%} → {after:.1%} after cleaning + interpolation')


def first_short_gap(series, min_len=3, max_len=MAX_INTERP_GAP):
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
    ax.plot(window.index, window['pm25'], 'o--', color='orange', label='After interpolation')
    ax.plot(window.index, window['pm25_raw'], 'o-', color=CITY_COLORS[city], label='Original data')
    ax.set(title=f'Short-gap interpolation example – {city}', ylabel='PM2.5 (µg/m³)')
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
    lo, hi = iqr_bounds(df['pm25'])
    out = df.loc[df['pm25'] > hi, 'pm25']
    top_days = out.resample('D').count().nlargest(3).index.strftime('%Y-%m-%d').tolist()
    print(f'{city}: upper bound = {hi:.1f} µg/m³ | {len(out)} outlier hours '
          f'({len(out) / df.pm25.notna().sum():.2%}) | days with most outliers: {top_days}')
    ax.plot(df.index, df['pm25'], lw=0.4, color=CITY_COLORS[city])
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

Each sample at time $t$ has target `y = pm25(t + 24h)`. Only information **available at $t$** is used (no future leakage):

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
    \"\"\"Add target = pm25(t + horizon); keep only samples with a target and a current value.\"\"\"
    X = build_features(df, city, horizon)
    X[TARGET] = df['pm25'].shift(-horizon)
    return X.dropna(subset=[TARGET, 'pm25_lag_0'])


datasets = {city: make_dataset(df, city) for city, df in clean.items()}
for city, d in datasets.items():
    print(f'{city}: {len(d):,} samples | {len(FEATURES)} features '
          f'({len(FEATURES) - len(CAT_FEATURES)} numerical + {len(CAT_FEATURES)} categorical)')
datasets['Hanoi'].head()
""")

md("""
### 2.5. Chronological Train / Validation / Test split
A time series **must not be shuffled** (that would leak the future into the past). The split follows **time order**:
first 70% → Train, next 15% → Validation (model selection), last 15% → Test (final evaluation).
An extra `HORIZON` = 24 hours is dropped before each boundary so that the **targets** of one set do not fall inside the next set's period.
""")

code("""
def time_split(d, train_frac=0.70, val_frac=0.15, gap_hours=HORIZON):
    n = len(d)
    t1, t2 = d.index[int(n * train_frac)], d.index[int(n * (train_frac + val_frac))]
    gap = pd.Timedelta(hours=gap_hours)
    train = d[d.index < t1 - gap]
    val = d[(d.index >= t1) & (d.index < t2 - gap)]
    test = d[d.index >= t2]
    return train, val, test


splits = {city: time_split(d) for city, d in datasets.items()}

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
for ax, (city, (tr, va, te)) in zip(axes, splits.items()):
    for part, name, color in [(tr, 'Train', 'tab:blue'), (va, 'Validation', 'tab:orange'), (te, 'Test', 'tab:green')]:
        daily = part[TARGET].resample('D').mean()
        ax.plot(daily.index, daily, color=color, lw=0.8,
                label=f'{name}: {len(part):,} samples ({part.index.min():%d/%m/%Y} – {part.index.max():%d/%m/%Y})')
    ax.set(title=city, ylabel='PM2.5 (µg/m³)')
    ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.show()
""")

md("""
### 2.6. Preprocessing pipeline (imputation, outlier clipping, encoding, scaling)

| Column group | Steps |
|---|---|
| PM2.5 lags/rolling | `SimpleImputer(median)` → `IQRClipper(k=3)` → `StandardScaler` |
| Weather + hour cycle | `SimpleImputer(median)` → `StandardScaler` |
| Categorical | `SimpleImputer(most_frequent)` → `OneHotEncoder` |

Everything lives inside a `ColumnTransformer` + `Pipeline`, so all parameters (medians, IQR bounds, mean/std, categories)
are **learned on the training set only** → no data leakage.
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


tr = splits['Hanoi'][0]
pre = build_preprocessor().fit(tr[FEATURES])
Xt = pd.DataFrame(pre.transform(tr[FEATURES]), columns=pre.get_feature_names_out(), index=tr.index)
print(f'After preprocessing: {Xt.shape[1]} columns (from {len(FEATURES)} original columns, one-hot expands the categoricals)')
print('Missing values left after the pipeline:', int(Xt.isna().sum().sum()))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(tr['pm25_lag_0'], bins=80, ax=axes[0], color='gray')
axes[0].set(title='pm25_lag_0 – before preprocessing (Hanoi, train)', xlabel='µg/m³')
sns.histplot(Xt['pm__pm25_lag_0'], bins=80, ax=axes[1], color='teal')
axes[1].set(title='After IQRClipper + StandardScaler', xlabel='z-score')
plt.tight_layout()
plt.show()
""")
