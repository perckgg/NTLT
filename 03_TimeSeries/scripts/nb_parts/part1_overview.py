# Phần 0 + 1: Header, môi trường, mô tả dữ liệu & EDA
md("""
# BÀI TẬP LỚN: NỀN TẢNG LẬP TRÌNH CHO PHÂN TÍCH VÀ TRỰC QUAN DỮ LIỆU
**Học kỳ 261 – Năm học 2026–2027** | **Giảng viên:** Lê Thành Sách

---
## BÀI 3: DỮ LIỆU CHUỖI THỜI GIAN (TIME SERIES)
### Đề tài: Dự báo nồng độ bụi mịn PM2.5 trước 24 giờ — So sánh Hà Nội và TP.HCM

- **Nhóm:** [Tên nhóm]
- **Thành viên phụ trách:** [Họ tên – MSSV]

**Bài toán.** Tại thời điểm $t$, dựa vào lịch sử PM2.5 và thời tiết hiện tại, dự báo nồng độ PM2.5
tại thời điểm $t + 24h$ (cùng giờ ngày mai). Đây là bài toán **hồi quy chuỗi thời gian**.

**Vì sao so sánh 2 thành phố?** Hà Nội (4 mùa, mùa đông nghịch nhiệt) và TP.HCM (2 mùa mưa/khô) có cơ chế
ô nhiễm khác nhau. Áp dụng **cùng một pipeline** cho từng thành phố **độc lập** (không gộp dữ liệu) để trả lời:
*mô hình học máy cải thiện bao nhiêu so với baseline ở mỗi nơi, và nơi nào khó dự báo hơn?*
""")

md("## 0. KHỞI TẠO MÔI TRƯỜNG")

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
HORIZON = 24            # dự báo trước 24 giờ
WHO_24H = 15            # ngưỡng khuyến nghị WHO 2021 cho PM2.5 trung bình 24h (µg/m³)
CITY_COLORS = {'Hà Nội': '#d62728', 'TP.HCM': '#1f77b4'}
print('Môi trường đã sẵn sàng!')
""")

md("""
## 1. MÔ TẢ TẬP DỮ LIỆU

### 1.1. Nguồn dữ liệu (tự thu thập)
Dữ liệu được **tự thu thập qua API** bằng script `scripts/crawl_air_quality.py`:

| Thành phần | Nguồn | Ghi chú |
|---|---|---|
| PM2.5 – Hà Nội | [OpenAQ v3](https://openaq.org) – trạm *Số 46 Lưu Quang Vũ* (Hanoi Air Quality Monitoring Network, sensor 7772032) | Trạm quan trắc của thành phố, **có khoảng trống thật** |
| PM2.5 – TP.HCM | OpenAQ v3 – trạm *CMT8* (AirGradient, sensor 11357424) | Cảm biến chi phí thấp |
| Thời tiết | [Open-Meteo Historical Weather API](https://open-meteo.com) | Nhiệt độ, độ ẩm, mưa, gió, áp suất, mây |

- **Giai đoạn:** 01/12/2024 – 30/09/2026 (22 tháng, 16,056 giờ mỗi thành phố), giờ Việt Nam (UTC+7).
- Lưới thời gian chuẩn hóa đầy đủ theo giờ: giờ nào trạm không đo → `NaN`.

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `datetime` | datetime | Thời điểm (giờ địa phương) |
| `city` | categorical | Thành phố |
| `temperature_2m` | numerical | Nhiệt độ 2m (°C) |
| `relative_humidity_2m` | numerical | Độ ẩm tương đối (%) |
| `precipitation` | numerical | Lượng mưa (mm) |
| `wind_speed_10m` | numerical | Tốc độ gió 10m (km/h) |
| `wind_direction_10m` | numerical (góc) | Hướng gió (độ) → chuyển thành **categorical** 8 hướng |
| `surface_pressure` | numerical | Áp suất bề mặt (hPa) |
| `cloud_cover` | numerical | Độ che phủ mây (%) |
| `pm25` | numerical | Nồng độ PM2.5 (µg/m³) — **biến cần dự báo** |
| `pm25_source` | categorical | Nguồn PM2.5 (`openaq`, hoặc `open-meteo` nếu crawler phải fallback) |

### 1.2. Nạp dữ liệu
Đọc CSV local; nếu chạy trên Colab không có file thì tải từ GitHub (**sửa `GITHUB_RAW_BASE`** theo repo của nhóm).
""")

code("""
GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/perckgg/NTLT/main/03_TimeSeries/data/timeseries'
CITY_FILES = {'Hà Nội': 'air_quality_hanoi.csv', 'TP.HCM': 'air_quality_hcmc.csv'}
LOCAL_DIRS = ['data/timeseries', '../data/timeseries', '/content/data/timeseries', '/content']


def load_city_data(filename):
    for d in LOCAL_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=['datetime'])
    return pd.read_csv(f'{GITHUB_RAW_BASE}/{filename}', parse_dates=['datetime'])


raw = {city: load_city_data(f).set_index('datetime').sort_index() for city, f in CITY_FILES.items()}
for city, df in raw.items():
    print(f'{city}: {df.shape[0]:,} dòng × {df.shape[1]} cột | '
          f'{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d} | nguồn PM2.5: {", ".join(df.pm25_source.unique())}')
raw['Hà Nội'].head()
""")

md("### 1.3. Thống kê mô tả")

code("""
raw['Hà Nội'].info()
summary = pd.concat({city: df['pm25'].describe() for city, df in raw.items()}, axis=1).T
summary['missing_rate'] = [df['pm25'].isna().mean() for df in raw.values()]
summary['% giờ > WHO'] = [(df['pm25'] > WHO_24H).mean() for df in raw.values()]
display(summary.round(3))
display(pd.concat({city: df.describe().T for city, df in raw.items()}).round(2))
""")

md("""
### 1.4. Trực quan hóa chuỗi thời gian
**Hình 1 – PM2.5 trung bình ngày** của hai thành phố so với ngưỡng WHO 24h (15 µg/m³).
""")

code("""
fig, ax = plt.subplots(figsize=(14, 5))
for city, df in raw.items():
    daily = df['pm25'].resample('D').mean()
    ax.plot(daily.index, daily, label=city, color=CITY_COLORS[city], lw=0.9)
ax.axhline(WHO_24H, color='k', ls='--', lw=1, label='Ngưỡng WHO 24h (15 µg/m³)')
ax.set(title='PM2.5 trung bình ngày', ylabel='PM2.5 (µg/m³)')
ax.legend()
plt.show()
""")

md("**Hình 2 – Phân phối PM2.5 theo giờ** và **Hình 3 – Boxplot theo tháng** (tính mùa vụ).")

code("""
plot_df = pd.concat([df[['pm25']].assign(city=city) for city, df in raw.items()])
plot_df['month'] = plot_df.index.month
plot_df['hour'] = plot_df.index.hour
plot_df = plot_df.reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 4))
sns.histplot(data=plot_df, x='pm25', hue='city', bins=100, element='step', stat='density',
             common_norm=False, palette=CITY_COLORS, ax=ax)
ax.set(title='Phân phối PM2.5 theo giờ', xlabel='PM2.5 (µg/m³)', xlim=(0, 250))
plt.show()

fig, ax = plt.subplots(figsize=(14, 5))
sns.boxplot(data=plot_df, x='month', y='pm25', hue='city', palette=CITY_COLORS, showfliers=False, ax=ax)
ax.set(title='PM2.5 theo tháng (ẩn outlier để dễ nhìn)', xlabel='Tháng', ylabel='PM2.5 (µg/m³)')
plt.show()
""")

md("""
**Hình 4 – Chu kỳ ngày:** PM2.5 trung bình theo giờ, tách nửa năm khô/lạnh (tháng 11–4) và mưa/nóng (tháng 5–10).
""")

code("""
plot_df['half_year'] = np.where(plot_df['month'].isin([11, 12, 1, 2, 3, 4]), 'Tháng 11–4', 'Tháng 5–10')
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for ax, city in zip(axes, CITY_FILES):
    g = plot_df[plot_df.city == city]
    sns.lineplot(data=g, x='hour', y='pm25', hue='half_year', errorbar=None, ax=ax)
    ax.set(title=city, xlabel='Giờ trong ngày', ylabel='PM2.5 trung bình (µg/m³)', xticks=range(0, 24, 3))
plt.tight_layout()
plt.show()
""")

md("**Hình 5 – Ma trận tương quan (Spearman)** giữa PM2.5 và các biến thời tiết, từng thành phố.")

code("""
num_cols = ['pm25', 'temperature_2m', 'relative_humidity_2m', 'precipitation',
            'wind_speed_10m', 'surface_pressure', 'cloud_cover']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (city, df) in zip(axes, raw.items()):
    sns.heatmap(df[num_cols].corr(method='spearman'), annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, ax=ax, cbar=False)
    ax.set_title(f'Tương quan Spearman – {city}')
plt.tight_layout()
plt.show()
""")

md("**Nhận xét EDA:**\n\n" + note("EDA"))
