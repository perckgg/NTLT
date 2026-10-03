# Phần 2: Chuẩn bị dữ liệu
md("""
## 2. CHUẨN BỊ DỮ LIỆU

Các bước dưới đây được viết thành **hàm tái sử dụng** và áp dụng **giống hệt nhau** cho từng thành phố,
nhưng mỗi thành phố được xử lý **độc lập**.

### 2.1. Thống kê missing values
""")

code("""
def gap_lengths(series):
    \"\"\"Độ dài (giờ) của từng đoạn missing liên tiếp.\"\"\"
    is_na = series.isna()
    groups = (is_na != is_na.shift()).cumsum()
    return is_na.groupby(groups).sum()[lambda s: s > 0]


def missing_report(raw_dict):
    rows = []
    for city, df in raw_dict.items():
        gaps = gap_lengths(df['pm25'])
        rows.append({'Thành phố': city, 'Số giờ thiếu': int(df['pm25'].isna().sum()),
                     'Tỉ lệ thiếu': f"{df['pm25'].isna().mean():.1%}",
                     'Số đoạn thiếu': len(gaps), 'Đoạn ≤ 6h': int((gaps <= 6).sum()),
                     'Đoạn dài nhất (giờ)': int(gaps.max()) if len(gaps) else 0})
    return pd.DataFrame(rows)


display(missing_report(raw))
display((pd.concat({city: df.isna().mean() for city, df in raw.items()}, axis=1) * 100)
        .round(2).rename(columns=lambda c: f'{c} (% missing)'))

fig, ax = plt.subplots(figsize=(14, 4))
for city, df in raw.items():
    avail = df['pm25'].notna().resample('MS').mean() * 100
    ax.plot(avail.index, avail, marker='o', label=city, color=CITY_COLORS[city])
ax.set(title='Tỉ lệ giờ có dữ liệu PM2.5 theo tháng', ylabel='% giờ có dữ liệu', ylim=(0, 105))
ax.legend()
plt.show()
""")

md("""
### 2.2. Làm sạch giá trị lỗi và xử lý missing

- **Giá trị không hợp lệ:** PM2.5 âm hoặc > 1000 µg/m³ là lỗi cảm biến → gán `NaN`.
- **Cảm biến bị "kẹt":** cùng một giá trị lặp lại ≥ 6 giờ liên tiếp (khi PM2.5 > 0) là dấu hiệu treo thiết bị → gán `NaN`.
- **Khoảng trống ngắn (≤ 6 giờ):** nội suy tuyến tính theo thời gian (`interpolate(method='time', limit=6)`) —
  hợp lý vì PM2.5 biến đổi liên tục trong vài giờ.
- **Khoảng trống dài (> 6 giờ):** **không** nội suy (dễ bịa dữ liệu) → mẫu có *target* bị thiếu sẽ bị loại;
  feature còn thiếu được `SimpleImputer` (median, học trên tập train) xử lý bên trong pipeline.
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
    print(f'{city}: {n_invalid} giá trị âm/>1000, {n_stuck} giờ cảm biến kẹt | '
          f'missing PM2.5: {before:.1%} → {after:.1%} sau làm sạch + nội suy')


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
    ax.plot(window.index, window['pm25'], 'o--', color='orange', label='Sau nội suy')
    ax.plot(window.index, window['pm25_raw'], 'o-', color=CITY_COLORS[city], label='Dữ liệu gốc')
    ax.set(title=f'Minh họa nội suy khoảng trống ngắn – {city}', ylabel='PM2.5 (µg/m³)')
    ax.legend()
    plt.show()
""")

md("""
### 2.3. Phát hiện outliers (IQR)
Quy tắc IQR: giá trị ngoài $[Q1 - 3\\cdot IQR,\\ Q3 + 3\\cdot IQR]$ được coi là outlier cực trị.
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
    print(f'{city}: ngưỡng trên = {hi:.1f} µg/m³ | {len(out)} giờ outlier '
          f'({len(out) / df.pm25.notna().sum():.2%}) | ngày nhiều outlier nhất: {top_days}')
    ax.plot(df.index, df['pm25'], lw=0.4, color=CITY_COLORS[city])
    ax.scatter(out.index, out, s=6, color='k', label='Outlier (IQR, k=3)')
    ax.axhline(hi, color='k', ls='--', lw=1)
    ax.set(title=city, ylabel='PM2.5 (µg/m³)')
    ax.legend()
plt.tight_layout()
plt.show()
""")

md("""
**Quyết định xử lý outlier:** các đỉnh PM2.5 là **sự kiện có thật** (đợt ô nhiễm mùa đông, đốt rơm rạ, pháo hoa...)
nên **không xóa** khỏi target (xóa sẽ khiến đánh giá lạc quan giả tạo). Thay vào đó, trên các **feature** PM2.5
(lag, rolling) dùng transformer tự viết `IQRClipper` (học ngưỡng **chỉ trên tập train**) để giảm ảnh hưởng của
giá trị cực trị tới mô hình tuyến tính.
""")

md("""
### 2.4. Tạo đặc trưng (Feature Engineering)

Mỗi mẫu tại thời điểm $t$ có target `y = pm25(t + 24h)`. Chỉ dùng thông tin **có sẵn tại $t$** (tránh rò rỉ tương lai):

| Nhóm | Feature | Loại |
|---|---|---|
| Lịch sử PM2.5 | `pm25_lag_0, 1, 2, 3, 6, 12, 23` (giá trị tại $t-k$), `pm25_roll_mean_6/24`, `pm25_roll_std_24`, `pm25_diff_24` | numerical |
| Thời tiết tại $t$ | nhiệt độ, độ ẩm, mưa, tốc độ gió, áp suất, mây, `rain_24h` (tổng mưa 24h qua) | numerical |
| Lịch của thời điểm cần dự báo | `hour_sin/cos` (mã hóa vòng tròn giờ) | numerical |
| | `season` (mùa theo khí hậu từng thành phố), `day_of_week`, `wind_sector` (8 hướng gió) | **categorical** |

> `pm25_lag_0` là giá trị **cùng giờ ngày hôm trước** so với thời điểm cần dự báo, nên baseline
> **Persistence** (`ŷ(t+24) = pm25(t)`) cũng chính là baseline *seasonal naive* 24h.
""")

code("""
SEASONS = {
    'Hà Nội': {12: 'Đông', 1: 'Đông', 2: 'Đông', 3: 'Xuân', 4: 'Xuân', 5: 'Xuân',
               6: 'Hạ', 7: 'Hạ', 8: 'Hạ', 9: 'Thu', 10: 'Thu', 11: 'Thu'},
    'TP.HCM': {m: ('Mùa khô' if m in (12, 1, 2, 3, 4) else 'Mùa mưa') for m in range(1, 13)},
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
    \"\"\"Tạo feature tại mỗi thời điểm t (chỉ dùng dữ liệu ≤ t).\"\"\"
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
    \"\"\"Thêm target = pm25(t + horizon); chỉ giữ mẫu có target và có giá trị hiện tại.\"\"\"
    X = build_features(df, city, horizon)
    X[TARGET] = df['pm25'].shift(-horizon)
    return X.dropna(subset=[TARGET, 'pm25_lag_0'])


datasets = {city: make_dataset(df, city) for city, df in clean.items()}
for city, d in datasets.items():
    print(f'{city}: {len(d):,} mẫu | {len(FEATURES)} feature '
          f'({len(FEATURES) - len(CAT_FEATURES)} numerical + {len(CAT_FEATURES)} categorical)')
datasets['Hà Nội'].head()
""")

md("""
### 2.5. Chia tập Train / Validation / Test theo thời gian
Chuỗi thời gian **không được xáo trộn ngẫu nhiên** (sẽ rò rỉ tương lai vào quá khứ). Chia **theo thứ tự thời gian**:
70% đầu → Train, 15% kế tiếp → Validation (chọn mô hình), 15% cuối → Test (đánh giá cuối cùng).
Bỏ thêm `HORIZON` = 24 giờ trước mỗi ranh giới để **target** của tập trước không rơi vào khoảng thời gian của tập sau.
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
                label=f'{name}: {len(part):,} mẫu ({part.index.min():%d/%m/%Y} – {part.index.max():%d/%m/%Y})')
    ax.set(title=city, ylabel='PM2.5 (µg/m³)')
    ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.show()
""")

md("""
### 2.6. Pipeline tiền xử lý (Imputation, Outlier clipping, Encoding, Scaling)

| Nhóm cột | Các bước |
|---|---|
| PM2.5 lag/rolling | `SimpleImputer(median)` → `IQRClipper(k=3)` → `StandardScaler` |
| Thời tiết + chu kỳ giờ | `SimpleImputer(median)` → `StandardScaler` |
| Categorical | `SimpleImputer(most_frequent)` → `OneHotEncoder` |

Tất cả nằm trong `ColumnTransformer` + `Pipeline` nên các tham số (median, ngưỡng IQR, mean/std, danh mục)
**chỉ được học trên tập train** → không rò rỉ dữ liệu.
""")

code("""
class IQRClipper(BaseEstimator, TransformerMixin):
    \"\"\"Cắt (winsorize) mỗi cột về [Q1 - k·IQR, Q3 + k·IQR], ngưỡng học trên dữ liệu fit.\"\"\"

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


tr = splits['Hà Nội'][0]
pre = build_preprocessor().fit(tr[FEATURES])
Xt = pd.DataFrame(pre.transform(tr[FEATURES]), columns=pre.get_feature_names_out(), index=tr.index)
print(f'Sau tiền xử lý: {Xt.shape[1]} cột (từ {len(FEATURES)} cột gốc, one-hot mở rộng categorical)')
print('Missing còn lại sau pipeline:', int(Xt.isna().sum().sum()))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(tr['pm25_lag_0'], bins=80, ax=axes[0], color='gray')
axes[0].set(title='pm25_lag_0 – trước xử lý (Hà Nội, train)', xlabel='µg/m³')
sns.histplot(Xt['pm__pm25_lag_0'], bins=80, ax=axes[1], color='teal')
axes[1].set(title='Sau IQRClipper + StandardScaler', xlabel='z-score')
plt.tight_layout()
plt.show()
""")
