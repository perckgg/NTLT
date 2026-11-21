# Giải thích chi tiết notebook Time Series (để bảo vệ bài)

File này đi theo **đúng thứ tự notebook** `notebooks/03_TimeSeries_Air_Quality.ipynb`
(code nằm ở `scripts/nb_parts/part1…part4`). Mỗi mục gồm: code làm gì → công thức → vì sao chọn như vậy →
câu thầy có thể hỏi. Mọi con số trong markdown của notebook đều do một cell in ra; nếu bị hỏi "số này ở đâu",
hãy chỉ đúng cell đó.

Ký hiệu: $y_t$ = PM2.5 đo được lúc $t$ (µg/m³), $H = 24$ = tầm dự báo (giờ), $\hat y$ = giá trị dự báo.

---

## 0. Bài toán

- **Input** tại giờ $t$: lịch sử PM2.5 đến $t$, thời tiết lúc $t$, giờ/mùa của thời điểm cần dự báo.
- **Output**: $\hat y_{t+24}$. Đây là **hồi quy** (regression), không phải phân loại.
- Mỗi thành phố một mô hình riêng, cùng pipeline. Lý do (có số trong EDA): mùa và chu kỳ ngày khác nhau,
  yếu tố thời tiết liên quan khác nhau (Hà Nội: mây/mưa/áp suất; HCMC: gió).

**Câu hỏi có thể gặp:** *Sao không gộp hai thành phố?* → Hai cơ chế ô nhiễm khác nhau; gộp lại thì mô hình phải
học thêm biến "thành phố" và kết quả bị trộn lẫn. Mục tiêu bài là **so sánh** hai nơi.

---

## 1. Mô tả dữ liệu & EDA (`part1_overview.py`)

### 1.1 Thu thập (`scripts/crawl_air_quality.py`)
- `fetch_pm25_openaq`: gọi `GET /v3/sensors/{id}/hours` theo từng năm, mỗi trang 1000 bản ghi, ngủ 1,1 s giữa các
  trang (giới hạn 60 request/phút). Lấy `period.datetimeFrom.local` (= **đầu giờ**) làm thời gian, `value` làm PM2.5.
  Trạm Hà Nội đo nhiều lần mỗi giờ; `value` là **trung bình theo giờ do OpenAQ tính**.
- `fetch_weather`: Open-Meteo archive, theo giờ, múi giờ `Asia/Ho_Chi_Minh`, **tại toạ độ của trạm đo PM2.5**.
- `build_city_dataset`: tạo **lưới giờ đầy đủ** `pd.date_range(..., freq="h")` rồi `reindex` → giờ không có số đo
  thành `NaN`. Đây là điểm quan trọng: nếu không có lưới đầy đủ, `shift(k)` sẽ lùi $k$ **dòng** chứ không phải $k$ **giờ**.
- Nếu không có API key → fallback sang Open-Meteo Air Quality (dữ liệu mô hình CAMS) và ghi vào cột `pm25_source`.
  Dữ liệu nộp bài là `openaq` (cell 1.2 in ra nguồn và tên trạm).
- **Giai đoạn:** 04/07/2025 – 14/11/2026, **giống nhau cho cả hai thành phố**. 04/07/2025 là ngày đầy đủ đầu tiên của trạm Nhân Chính
  (trạm bắt đầu báo lúc 22:40 ngày 03/07/2025); 14/11/2026 là ngày đầy đủ cuối cùng trước khi crawl. 499 ngày = 11.976 giờ.

### 1.1b Chọn trạm (`scripts/check_station_coverage.py`)
Với mọi sensor PM2.5 của OpenAQ trong bán kính 25 km quanh mỗi thành phố có báo số liệu trong giai đoạn, script tính:

| Cột | Định nghĩa |
|---|---|
| `coverage` | tỉ lệ giờ có số đo trong lưới giờ của giai đoạn |
| `worst_hour_cov` | $\min_{h=0..23}$ (coverage của các giờ có giờ-trong-ngày = $h$) |
| `worst_month_cov` | coverage thấp nhất trong các tháng |
| `longest_gap_h` | đoạn thiếu liên tiếp dài nhất (hàm `longest_run`, cùng mẹo `cumsum` ở 2.1) |

- **Vì sao cần `worst_hour_cov`:** nếu trạm thiếu chủ yếu vào một số giờ cố định (ví dụ buổi chiều), dữ liệu thiếu **không ngẫu nhiên**
  (MNAR – missing not at random): giờ bị thiếu khác hệ thống so với giờ có đo → chu kỳ ngày và sai số dự báo đều bị lệch.
- **Kết quả (bảng in ở 1.2):** Nhân Chính 96,4% (giờ tệ nhất 95,0%) → chọn. Trạm cũ 46 Lưu Quang Vũ chỉ 51,0%, giờ tệ nhất **27,9%**
  → đó là lý do đổi trạm. HCMC: CMT8 là trạm duy nhất có dữ liệu liên tục (99,4%).

**Hỏi:** *Sao không ghép trạm cũ (có mùa đông 2024–25) với trạm mới để có hai mùa đông?* → Hai trạm khác vị trí, khác thiết bị; ghép lại tạo
bước nhảy giả trong chuỗi đúng ở điểm nối, và mô hình học cả sự khác biệt giữa hai trạm. Bài chọn một trạm sạch, chấp nhận chỉ có một mùa đông
và nói rõ hạn chế đó.

### 1.2 `load_csv`
Đọc CSV ở các thư mục cục bộ; không có thì tải từ `GITHUB_RAW_BASE` (để Colab "Run all" được). Tham số `parse_dates` để dùng chung cho
file dữ liệu (cột `datetime`) và bảng coverage (cột `first`, `last`).

### 1.3 Thống kê mô tả, `daily_means`
- `describe()`: count, mean, std, min, quartiles, max.
- **Ngưỡng WHO 15 µg/m³ là cho trung bình 24 giờ.** Vì vậy so với **trung bình ngày**, không so từng giờ.
- `daily_means(s, min_hours=18)`: trung bình ngày chỉ tính cho ngày có ≥ 18/24 giờ đo (75%); ngày ít hơn → `NaN`.
  Lý do: trung bình của 3–4 giờ không đại diện cho cả ngày. 75% là quy tắc đầy đủ dữ liệu thường dùng cho trung bình ngày.
- Kết quả: vượt ngưỡng ở 92,2% số ngày hợp lệ (Hà Nội, 476 ngày) và 97,6% (HCMC, 498 ngày).

### 1.4 Hình 1–3
- Hình 1: `daily_means` theo ngày; đường đứt quãng ở những ngày không đủ 18 giờ.
- Hình 3 là **boxplot = trung vị và tứ phân vị**; bảng in ngay dưới có cả **mean** và **median** theo tháng.
  Notes dùng mean (ghi rõ), và nói median thấp hơn vài µg/m³ vì phân phối lệch phải.
- Dữ liệu chỉ có **một mùa đông** → mỗi tháng mùa đông chỉ là một lần quan sát của tháng đó.

### 1.5 Hình 4 – chu kỳ ngày, `complete_days_only`
- Chỉ giữ giờ của những ngày có ≥ 18 giờ đo: `s.groupby(s.index.floor('D')).transform('count') >= 18`
  (`transform` trả về số giờ đo của ngày đó cho **mỗi** dòng, rồi lọc).
- **Vì sao lọc:** để mọi giờ trong ngày được lấy trung bình trên (gần như) cùng một tập ngày. Nếu một giờ hay thiếu hơn các giờ khác,
  trung bình theo giờ sẽ trộn các ngày khác nhau → đường cong phản ánh *ngày nào có dữ liệu*. Với trạm Nhân Chính, giờ tệ nhất vẫn có
  95,0% số đo (`worst_hour_cov`), gần bằng coverage chung 96,4%, nên lọc thay đổi rất ít; giữ lại như một biện pháp an toàn (với trạm cũ thì bắt buộc).

### 1.6 Hình 5 – Spearman
- $\rho_S$ = hệ số Pearson tính trên **thứ hạng** (rank) của hai biến → đo quan hệ **đơn điệu**, ít nhạy với ngoại lai,
  không cần quan hệ tuyến tính. Phù hợp vì PM2.5 lệch phải mạnh.
- Đây là **tương quan, không phải nhân quả**: mùa đông vừa lạnh, vừa áp suất cao, vừa ít mưa → các hệ số lẫn nhau.

### 1.7 Bảng tự tương quan (autocorrelation)
- $r(k) = \mathrm{corr}(y_t, y_{t-k})$ trên các cặp mà cả hai giờ đều có số đo. Code: `Series.autocorr(k)`
  (pandas tính bằng `corr` với chính chuỗi dịch $k$ dòng; vì lưới giờ đầy đủ nên $k$ dòng = $k$ giờ).
- 0,95 (Hà Nội) / 0,88 (HCMC) ở lag 1h nhưng 0,59 / 0,47 ở lag 24h → dự báo 24h khó hơn nhiều so với 1h.
- HCMC: $r(24) = 0{,}47 > r(12) = 0{,}27$ vì cùng giờ trong ngày có xu hướng lặp lại (chu kỳ ngày).

---

## 2. Chuẩn bị dữ liệu (`part2_preparation.py`)

### 2.1 `gap_lengths`, `missing_report`
```python
is_na = series.isna()
groups = (is_na != is_na.shift()).cumsum()
```
- `is_na != is_na.shift()` là True tại mỗi điểm **đổi trạng thái** (có số ↔ thiếu); `cumsum()` biến nó thành mã nhóm
  tăng dần → mỗi đoạn liên tiếp cùng trạng thái có một mã. `is_na.groupby(groups).sum()` = số giờ thiếu của mỗi đoạn;
  lọc `> 0` để chỉ giữ các đoạn thiếu → độ dài từng khoảng trống.
- Bảng % thiếu theo tháng: `isna().resample('MS').mean() * 100` (`MS` = đầu tháng). Hà Nội thiếu 3,6% (tệ nhất 06/2026: 28%), HCMC 0,6%.

### 2.2 `clean_city`
1. **Giá trị sai:** `pm <= 0` hoặc `pm > 1000` → `NaN`. Trạm chuẩn báo đúng 0,0 là mất tín hiệu, không phải không khí sạch.
2. **Cảm biến bị kẹt:** cùng mẹo `cumsum` ở trên với `pm != pm.shift()` để tìm các đoạn **cùng giá trị**;
   `transform('size')` = độ dài đoạn; nếu ≥ 6 giờ và > 0 → `NaN`. (`NaN != NaN` là True nên mỗi NaN là một đoạn riêng,
   không bị tính là "kẹt".) Kết quả: không có giờ nào bị kẹt, không có giá trị sai.
3. Giữ hai cột:
   - `pm25_raw` = chỉ giá trị đo → dùng làm **target**.
   - `pm25` = `pm25_raw.ffill(limit=6)` → dùng làm **đặc trưng**. `ffill(limit=6)` lấp **tối đa 6 giờ liên tiếp**
     bằng giá trị đo gần nhất trước đó (kể cả 6 giờ đầu của một khoảng trống dài hơn).

**Vì sao không dùng `interpolate`:** nội suy tuyến tính giữa $y_a$ và $y_b$ ($a < t < b$):
$\tilde y_t = y_a + (y_b - y_a)\frac{t-a}{b-a}$ — dùng $y_b$ là số đo **sau** $t$, lúc dự báo thật chưa có →
**rò rỉ tương lai** (look-ahead leakage). `ffill` chỉ dùng $y_a$ ($a < t$) → **nhân quả** (causal).
Cell 2.4 in ra tỉ lệ mẫu có giá trị hiện tại là giá trị đã lấp (1,0% Hà Nội, 0,6% HCMC).

**Vì sao target không lấp:** target lấp là giá trị "bịa", lại dễ đoán hơn (đường thẳng) → điểm Test đẹp giả.

### 2.3 Ngoại lai, `iqr_bounds`
- $Q_1, Q_3$ = phân vị 25% và 75%; $IQR = Q_3 - Q_1$; biên $[Q_1 - k\cdot IQR,\ Q_3 + k\cdot IQR]$ với $k = 3$
  (“outer fence” của Tukey — ngoại lai cực đoan; $k = 1{,}5$ là “inner fence”).
- Quyết định: **giữ** đỉnh trong target (sự kiện ô nhiễm thật), chỉ **cắt (clip)** ở đặc trưng (xem `IQRClipper`).

### 2.4 Đặc trưng, `build_features`, `make_dataset`
Với mỗi giờ $t$ (chỉ dùng dữ liệu ≤ $t$):

| Đặc trưng | Công thức |
|---|---|
| `pm25_lag_k` (k = 0,1,2,3,6,12,23) | $y_{t-k}$ (từ chuỗi đã ffill) — `shift(k)` |
| `pm25_roll_mean_6` | trung bình 6 giờ $t-5 … t$, cần ≥ 3 giá trị (`min_periods=3`) |
| `pm25_roll_mean_24`, `pm25_roll_std_24` | trung bình / độ lệch chuẩn 24 giờ, cần ≥ 12 giá trị |
| `pm25_diff_24` | $y_t - y_{t-24}$ |
| thời tiết | giá trị lúc $t$; `rain_24h` = tổng mưa 24 giờ đến $t$ |
| `hour_sin`, `hour_cos` | $\sin(2\pi h/24)$, $\cos(2\pi h/24)$ với $h$ = **giờ của $t+24$** |
| `season`, `day_of_week` | của thời điểm $t + 24$ (categorical) |
| `wind_sector` | $\lfloor ((\text{hướng} + 22{,}5) \bmod 360) / 45 \rfloor$ → 0…7 → N, NE, …, NW |

- **Vì sao sin/cos:** nếu để giờ là số 0…23 thì 23h và 0h cách nhau 23 đơn vị dù thực ra kề nhau. Điểm
  $(\cos, \sin)$ nằm trên vòng tròn đơn vị nên khoảng cách 23h–0h bằng khoảng cách 0h–1h. Cần **cả hai**: chỉ dùng sin thì
  3h và 9h trùng nhau ($\sin(\pi/4) = \sin(3\pi/4) \approx 0{,}707$); cặp (sin, cos) khác nhau cho mỗi giờ.
- **Vì sao giờ/mùa của $t+24$ không rò rỉ:** lịch là biết trước.
- **+22,5 rồi chia 45:** mỗi hướng chiếm cung 45°, hướng N là $[337{,}5°, 22{,}5°)$; cộng 22,5 để cung này bắt đầu từ 0.
- `pm25_lag_0` = giá trị cùng giờ của **ngày trước** so với thời điểm dự báo → Persistence $\hat y_{t+24} = y_t$ chính là
  **seasonal naive** với chu kỳ 24.
- `make_dataset`: `target = pm25_raw.shift(-24)` (giá trị **đo** lúc $t+24$); bỏ dòng thiếu target hoặc thiếu `pm25_lag_0`.

### 2.5 Chia tập: `dev_test_split`, `walk_forward_folds`
**Development / Test (cùng ngày cho cả hai thành phố):**
- Test: mẫu phát hành (issue time) $t \ge$ `TEST_START` = 01/09/2026.
- Development: $t <$ `TEST_START` − 24h.
- **Vì sao trừ 24h:** mẫu Development cuối cùng có $t < s - 24h$ nên target ở $t + 24 < s$ → mọi target của Development đã **đo xong
  trước** dự báo Test đầu tiên. Nếu không bỏ, mô hình cuối sẽ được train trên giá trị đo **sau** thời điểm phát hành dự báo Test đầu tiên.

**Walk-forward (rolling forecasting origin; Tashman 2000; Hyndman & Athanasopoulos, *FPP* §5.10):**
với mỗi tháng $m$ từ 10/2025 đến 08/2026, $s_m$ = giờ đầu tiên của tháng:

$$\text{Train}_m = \{t \in \text{Dev} : t < s_m - 24h\}, \qquad \text{Val}_m = \{t \in \text{Dev} : s_m \le t < s_{m+1}\}$$

- **Cửa sổ mở rộng** (expanding window): Train$_m$ là *mọi thứ đã biết* trước tháng $m$ — giống cách dùng thật (mỗi tháng train lại với toàn bộ lịch sử).
- 11 tháng validation không chồng nhau → mỗi giờ Development từ 10/2025 được dự báo **đúng một lần**, bởi mô hình chưa từng thấy nó.
- 07–09/2025 chỉ dùng để train (fold đầu có tối thiểu 3 tháng lịch sử: 2.075 mẫu Hà Nội / 2.100 HCMC).
- **Vì sao không dùng một tập Validation cố định:** dữ liệu chỉ có một mùa đông. Một khối Validation (ví dụ 15% cuối) rơi vào một mùa
  (vd. mùa hè sạch) → chọn mô hình ở mùa sạch rồi chấm ở mùa ô nhiễm. Walk-forward đánh giá **mọi tháng**, kể cả cả mùa đông.
- Không xáo trộn (shuffle): xáo trộn thì mô hình học từ tương lai để đoán quá khứ.
- Đặc trưng của Val$_m$ dùng PM2.5 trong thời gian Train$_m$ — đó là **quá khứ**, hợp lệ.

**Hỏi:** *Mẫu Val$_m$ cuối tháng có target rơi sang tháng sau, có rò rỉ không?* → Không: target đó chỉ được dùng để **chấm** fold $m$;
fold $m+1$ train trên mẫu có $t < s_{m+1} - 24h$, tức là cùng điều kiện "target đã đo trước khi tháng bắt đầu".

### 2.6 `IQRClipper`, `build_preprocessor`
```python
class IQRClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):      # học biên trên dữ liệu fit
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        self.lower_, self.upper_ = q1 - k*(q3-q1), q3 + k*(q3-q1)
    def transform(self, X):        # x' = min(max(x, lower), upper)
        return np.clip(X, self.lower_, self.upper_)
```
- Kế thừa `BaseEstimator` (có `get_params`/`set_params`, dùng được trong `GridSearchCV`, `clone`) và `TransformerMixin`
  (có `fit_transform`). Thuộc tính có dấu `_` cuối = học từ dữ liệu (quy ước sklearn).
- `get_feature_names_out` trả lại tên cột để `ColumnTransformer` đặt tên đầu ra.
- `ColumnTransformer`:
  - PM2.5: `SimpleImputer(median)` → `IQRClipper(3)` → `StandardScaler`
  - thời tiết + sin/cos: `SimpleImputer(median)` → `StandardScaler`
  - categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`
- `StandardScaler`: $z = (x - \mu)/\sigma$, $\mu, \sigma$ tính trên dữ liệu fit (σ là độ lệch chuẩn tổng thể, ddof = 0).
- `OneHotEncoder`: mỗi giá trị → một cột 0/1. `handle_unknown='ignore'`: giá trị chưa gặp khi fit → toàn 0, không lỗi.
  Ví dụ thật: fold 12/2025 của Hà Nội train trên 07–11/2025, chưa có `season = Winter` → cột Winter toàn 0 lúc dự báo.
- 23 cột gốc → 39 cột sau one-hot (cell in ra).
- **Không rò rỉ:** mọi tham số (median, biên IQR, μ, σ, danh mục) học bằng `fit` trên **cửa sổ train của từng lần fit**
  (Train$_m$ trong walk-forward, cả Development cho mô hình cuối); dữ liệu đánh giá chỉ `transform`.

---

## 3. Mô hình hoá (`part3_modeling.py`)

### 3.1 Baselines
| Tên | Công thức |
|---|---|
| `PersistenceBaseline` | $\hat y_{t+24} = y_t$ |
| `MeanBaseline` | $\hat y_{t+24} = \bar y_{t-23..t}$ (thiếu thì dùng $y_t$) |
| `NaiveBlend` | $\hat y = w\,y_t + (1-w)\,\bar y_{24}$, $\ w^* = \arg\min_{w \in \{0, 0{,}05, …, 1\}} \frac1n\sum \lvert y - \hat y_w\rvert$ trên **dữ liệu train** |

- `w` học lại ở **mỗi fold** như mọi tham số khác: 0,65–0,75 ở Hà Nội, 0,45–0,55 ở HCMC (cell 3.2 in ra từng fold) → baseline ổn định.
  `argmin` trên lưới 21 giá trị; nếu hoà thì lấy giá trị đầu.
- **Vì sao cần Naive blend:** Persistence yếu. Nếu chỉ so với nó, mô hình học máy trông tốt hơn thực tế.

### 3.2 Chỉ số đánh giá
- $\mathrm{MAE} = \frac1n\sum|y_i - \hat y_i|$ (cùng đơn vị µg/m³, ít nhạy ngoại lai) — **chỉ số chính**.
- $\mathrm{RMSE} = \sqrt{\frac1n\sum(y_i - \hat y_i)^2}$ (phạt lỗi lớn mạnh hơn).
- $R^2 = 1 - \frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}$, $\bar y$ = trung bình của **chính tập đang đánh giá**.
  $R^2 < 0$ nghĩa là dự báo tệ hơn việc luôn đoán $\bar y$ (Persistence của HCMC trên Test: −0,160).
- **Bias** (bảng mới ở 3.2): $\frac{1}{|\text{Val}_m|}\sum_{i\in \text{Val}_m}(\hat y_i - y_i)$; âm = dự báo thấp trung bình.

**Hỏi:** *Sao chọn theo MAE mà không theo RMSE?* → Dễ hiểu (sai trung bình bao nhiêu µg/m³), và với phân phối lệch phải
RMSE bị vài đỉnh chi phối. Hệ quả: tối thiểu MAE khiến mô hình học **trung vị có điều kiện** → kéo dự báo về giữa,
dự báo thấp giờ ô nhiễm (thấy rõ ở 4.5).

### 3.3 Ba mô hình
- **Ridge:** $\min_{w,b} \|y - Xw - b\|_2^2 + \alpha\|w\|_2^2$ (intercept $b$ không bị phạt). Nghiệm đóng
  $w = (X^\top X + \alpha I)^{-1}X^\top y$ (trên dữ liệu đã trừ trung bình). $\alpha$ lớn → hệ số nhỏ, ít overfit.
  Cần chuẩn hoá trước để phạt công bằng giữa các đặc trưng. Grid $\alpha \in \{0{,}01; 0{,}1; 1; 10; 100; 1000\}$ → Hà Nội 0,01, HCMC 10.
  $\alpha = 0{,}01$ nằm ở biên grid nhưng không đáng lo: khi $\alpha \to 0$, Ridge tiến về bình phương tối thiểu thường (OLS).
- **RandomForest:** 200 cây; mỗi cây học trên một **bootstrap sample** (rút $n$ dòng có hoàn lại); tại mỗi nút chỉ xét
  ngẫu nhiên 50% số cột (`max_features=0.5`, tính trên 39 cột sau one-hot) và chọn phép chia giảm MSE nhiều nhất;
  dự báo = trung bình dự báo của 200 cây (bagging → giảm phương sai). Grid `max_depth` ∈ {4, 6, 8, 12}, `min_samples_leaf` ∈ {5, 10, 30, 100}.
- **HistGradientBoosting (HGB):** boosting — cây thứ $m$ học phần dư (gradient âm của sai số bình phương) của tổng các cây trước,
  $F_m = F_{m-1} + \eta\, h_m$ với $\eta$ = `learning_rate`. "Hist": mỗi đặc trưng được chia sẵn thành ≤ 255 bin → tìm phép chia nhanh.
  Giá trị lá $= -G/(H + \lambda)$ ($G$, $H$ = tổng gradient/hessian trong lá, $\lambda$ = `l2_regularization`).
  `max_iter` = số cây, `max_leaf_nodes` = số lá tối đa mỗi cây. Grid: lr {0,01; 0,03; 0,1}, max_iter {100, 200, 400},
  max_leaf_nodes {4, 7, 15}, l2 {1, 10, 100}.
  **`early_stopping=False`**: mặc định `'auto'` bật early stopping khi > 10.000 mẫu và khi đó giữ lại **10% dòng ngẫu nhiên**
  (không theo thời gian) để dừng sớm. Development HCMC có 10.076 mẫu → nếu để mặc định, HGB ở mô hình cuối sẽ dừng sớm còn ở CV và
  walk-forward (< 10.000 mẫu) thì không → hai cách chạy khác nhau. Tắt đi để số cây luôn đúng bằng `max_iter` đã tune.

### 3.4 `GridSearchCV` + `TimeSeriesSplit(n_splits=3, gap=24)` trên Development
- `TimeSeriesSplit`: chia Development thành 4 phần bằng nhau theo thứ tự; fold $i$ train trên phần đầu, validate trên phần kế tiếp
  (cửa sổ mở rộng). Cụ thể: `test_size` $= \lfloor n/4 \rfloor$; fold validate bắt đầu ở $n - (3-i)\cdot$`test_size`;
  train = các mẫu **trước** điểm đó trừ đi `gap` = 24 mẫu.
- **Vì sao `gap=24`:** 24 mẫu cuối của fold train có target nằm trong khoảng thời gian của fold validate. Vì mỗi giờ có tối đa
  1 mẫu, 24 mẫu trải ít nhất 24 giờ → đủ khoảng đệm.
- `scoring='neg_mean_absolute_error'`: sklearn luôn **tối đa hoá** điểm nên dùng −MAE.
- `refit=True` (mặc định): fit lại bộ tham số tốt nhất trên toàn Development.
- **Cảnh báo biên grid:** cell in `at the edge of the grid` cho tham số nằm ở đầu/cuối grid, và CV MAE tốt nhất/tệ nhất trên grid.
  Đã mở rộng grid hai lần; mỗi lần giá trị chọn lại dời về phía **mô hình đơn giản hơn** (ít lá, cây nông) → dừng và ghi là hạn chế:
  tín hiệu 24h yếu nên mô hình đơn giản thắng.
- **Hạn chế (đã ghi):** hyperparameter tune trên cả Development — mà Development chứa các tháng walk-forward → điểm walk-forward của
  ML **hơi lạc quan**. Nested tuning (tune lại trong từng fold) loại bỏ được nhưng tốn thời gian ×11. Test không bị ảnh hưởng.

### 3.5 `CityForecaster`
- `tune()`: chạy GridSearch cho Ridge, RF, HGB trên Development; lưu `best_params` và khoảng CV MAE.
- `walk_forward()`: với mỗi fold, `clone(model)` (cùng hyperparameter, **chưa fit**) → fit trên Train$_m$ → dự báo Val$_m$;
  ghép 11 tháng thành `wf_predictions` (có cột `fold`). Điểm **gộp** (pooled):
  $\mathrm{MAE} = \sum_m \sum_{i \in \text{Val}_m} |y_i - \hat y_i| \,/\, \sum_m |\text{Val}_m|$ (tháng nhiều mẫu nặng hơn).
- `evaluate_on_test()`: fit lại trên **toàn Development**, dự báo Test **một lần**.
- Chọn mô hình tốt nhất = MAE walk-forward gộp nhỏ nhất **trong 3 mô hình học máy** → RandomForest (Hà Nội), HGB (HCMC). Không dùng Test để chọn.

**Hỏi:** *Vì sao dùng `clone`?* → `clone` tạo bản sao **chưa fit** với cùng tham số; không có nó, mô hình đã fit ở fold trước bị fit đè, và
mô hình cuối (`self.models`) bị thay bằng mô hình của fold 08/2026.

**Kết quả quan trọng (bảng tháng + bias):** ML thắng rõ ở đỉnh mùa đông (Hà Nội tháng 1: RF 15,54 vs blend 19,36) nhưng **thua** khi mức nền
thay đổi (Hà Nội 11/2025: 17,54 vs 14,96, bias −7,8; HCMC 02/2026: bias +5,8). Giải thích: mô hình học quan hệ trung bình trên dữ liệu
quá khứ nên phản ứng chậm khi mức ô nhiễm tăng/giảm; Naive blend không có intercept, chỉ bám mức hiện tại → bias nhỏ (trong ±2,5).
Đây là quan sát từ bảng bias; nguyên nhân chính xác là **suy luận**, cần mùa đông thứ hai để kiểm chứng.

---

## 4. Tổng kết & mở rộng (`part4_summary.py`)

### 4.1–4.2 Bảng Test, Hình 6
- "MAE vs X" $= 1 - \mathrm{MAE}_{\text{mô hình}}/\mathrm{MAE}_X$; dương = tốt hơn X.
- Test = 01/09 – 13/11/2026, 1.775 mẫu mỗi thành phố — **đầu** mùa ô nhiễm, không phải đỉnh.
- Ridge là mô hình tốt nhất trên Test ở Hà Nội (10,55) dù không được chọn → các mô hình ML quá sát nhau để việc chọn đáng tin.

### 4.3 Block bootstrap (`block_bootstrap_gain`)
Mục tiêu: khoảng tin cậy 95% cho $g = 1 - \mathrm{MAE}_A/\mathrm{MAE}_B$ ($A$ = mô hình tốt nhất, $B$ = đối thủ), trên hai tập:
**dự báo walk-forward gộp** (111–112 khối) và **Test** (25 khối).

1. Gán mỗi mẫu vào khối 3 ngày theo thời điểm dự báo:
   `codes = factorize((t - t.min()) // 3 ngày)` → $k$ khối **không chồng nhau** (khối không có mẫu nào thì không tồn tại).
2. Tổng lỗi mỗi khối: $S^A_j = \sum_{i\in j}|y_i - \hat y^A_i|$ — `np.bincount(codes, weights=...)`.
3. Rút lại $k$ khối có hoàn lại = vector đếm $c \sim \mathrm{Multinomial}(k;\ 1/k,…,1/k)$ (`rng.multinomial`), lặp 2.000 lần.
4. $g^* = 1 - \sum_j c_j S^A_j / \sum_j c_j S^B_j$ (`counts @ abs_err_a`). Số giờ triệt tiêu vì $A$ và $B$ được chấm trên **cùng** giờ (paired).
5. CI 95% = phân vị 2,5% và 97,5% của 2.000 giá trị $g^*$ (**percentile interval**).

- **Vì sao theo khối:** sai số của giờ liền nhau tương quan mạnh. Bootstrap từng giờ (i.i.d.) coi hàng nghìn giờ là quan sát độc lập
  → CI quá hẹp.
- **Vì sao 3 ngày:** cell in tự tương quan của chênh lệch sai số trung bình ngày trên walk-forward: 0,15 / −0,06 / 0,02 (Hà Nội) và
  0,16 / 0,20 / 0,10 (HCMC) ở lag 1/2/3 ngày → nhỏ. Kiểm tra lại với khối 1 và 7 ngày.
- **Diễn giải:** CI không chứa 0 ≈ bác bỏ "không khác nhau" ở mức 5% (kiểm định hai phía, xấp xỉ).
- **4 so sánh định trước:** mô hình tốt nhất vs Naive blend, trên walk-forward và Test, mỗi thành phố. Kết quả:
  - HCMC Test: +7,2% [+1,7; +12,6] → **có ý nghĩa** (cả khối 1, 3, 7 ngày).
  - HCMC walk-forward: +3,3% [−1,4; +7,6] → không.
  - Hà Nội Test: +5,3% [−3,4; +13,3] → không.
  - Hà Nội walk-forward: +6,8% [−0,9; +13,7] → **sát biên** (khối 7 ngày: [+0,4; +12,1] loại 0; khối 1 và 3 ngày thì không) → không khẳng định.
- So với Persistence: có ý nghĩa ở mọi thành phố, mọi tập.
- **Nguồn:** non-overlapping block bootstrap — Carlstein (1986); ý tưởng bootstrap theo khối cho dữ liệu phụ thuộc — Künsch (1989)
  (Künsch dùng khối **chồng nhau**, *moving block*; bài này dùng khối không chồng).
- **Hạn chế (đã ghi trong notebook):** Test chỉ 25 khối → CI xấp xỉ; 16 khoảng không hiệu chỉnh đa so sánh → chỉ mang tính mô tả;
  giả định chuỗi lỗi gần dừng (11 tháng có mùa đông ở giữa → chỉ gần đúng); trên walk-forward, mô hình tốt nhất được chọn trên chính các dự báo đó
  và tham số tune trên Development → gain walk-forward hơi lạc quan; CI của Test là khoảng không thiên lệch.

### 4.4 Dự báo vs thực tế (Hình 8)
MAE trong 14 ngày đầu Test và so sánh độ lệch chuẩn của dự báo với thực tế (Hà Nội 7,2 vs 15,5; HCMC 5,5 vs 18,5 → dự báo trơn, bỏ sót đỉnh).

### 4.5 Phân tích lỗi
- MAE theo giờ của thời điểm dự báo, vẽ cho walk-forward (rỗng) và Test (đặc); giờ có < 20 mẫu không vẽ. MAE theo tháng = hình walk-forward ở 3.2.
- `to_epa_band`: cắt cụt về 0,1 µg/m³ ($\lfloor 10x \rfloor/10$, đúng quy tắc EPA) rồi `pd.cut` với các khoảng **đóng bên phải**
  $(-\infty, 9{,}0], (9{,}0; 35{,}4], (35{,}4; 55{,}4], (55{,}4; 125{,}4], (125{,}4; \infty)$.
- **Không gọi là AQI:** AQI chính thức của EPA tính trên trung bình 24 giờ (hoặc NowCast); ở đây áp ngưỡng cho giá trị theo **giờ**
  → gọi là "dải nồng độ EPA".
- Ma trận nhầm lẫn: `pd.crosstab(thực, dự báo)`; tô màu theo tỉ lệ mỗi hàng, ghi số đếm; dòng `Measured hours per band` in tổng mỗi hàng.

### 4.6 Permutation importance
- $I_j = \frac1R\sum_{r=1}^{R}\big[\mathrm{MAE}(X^{(j,r)}) - \mathrm{MAE}(X)\big]$, $X^{(j,r)}$ = Test với cột $j$ bị xáo trộn lần $r$, $R = 5$.
  Mô hình **không** train lại. Giá trị âm (Calendar ở Hà Nội: −0,06) = xáo trộn còn làm lỗi nhỏ đi → nhóm đó không mang tín hiệu.
- **Vì sao không dùng `feature_importances_` của cây:** đó là importance theo impurity, tính trên dữ liệu train, thiên vị cột nhiều giá trị,
  và không áp dụng được cho Ridge.
- **Theo nhóm (`group_importance`):** xáo trộn **cùng một hoán vị** cho mọi cột trong nhóm → giữ quan hệ giữa các cột trong nhóm,
  phá quan hệ với target. Cần vì các lag PM2.5 tương quan mạnh: xáo một cột thì cột khác "bù" lại → importance từng cột bị chia nhỏ.

### 4.7 `predict_next`
Nhận lịch sử theo giờ đến $t$ → `build_features` → lấy dòng cuối → mô hình tốt nhất dự báo $t+24$ → dải EPA.
Demo dùng lịch sử kết thúc 24h trước giá trị đo cuối cùng để so được với thực tế.

---

## 5. Những điểm yếu nên tự nói ra trước (thầy hỏi thì đã có câu trả lời)
1. **Chỉ một mùa đông** (trạm Nhân Chính bắt đầu 07/2025). Kết luận về mùa đông chỉ dựa trên các tháng walk-forward; Test là đầu mùa.
2. Mỗi thành phố một trạm; HCMC là cảm biến giá rẻ (và là trạm duy nhất có dữ liệu liên tục).
3. Hà Nội: chưa chứng minh ML hơn Naive blend (Test CI [−3,4%; +13,3%]; walk-forward sát biên).
4. Không phân biệt được các mô hình ML ở cả hai thành phố; Ridge thậm chí tốt nhất trên Test ở Hà Nội.
5. Hyperparameter tune trên cả Development → walk-forward hơi lạc quan; nhiều tham số ở biên grid.
6. ML phản ứng chậm khi mức ô nhiễm đổi (bias theo tháng); thời tiết dùng giá trị đo tại $t$, không phải dự báo cho $t+24$.
7. Tối thiểu MAE → dự báo thấp giờ ô nhiễm (Hà Nội: 46/215 giờ "Unhealthy" đúng dải); chưa dùng được để cảnh báo sớm.

## 6. Câu hỏi nhanh – trả lời một câu
| Câu hỏi | Trả lời |
|---|---|
| Vì sao đổi trạm Hà Nội? | Trạm cũ chỉ đo 51,0% số giờ và 27,9% ở giờ tệ nhất → thiếu không ngẫu nhiên; Nhân Chính đo 96,4% (giờ tệ nhất 95,0%). |
| Vì sao không shuffle khi chia tập? | Shuffle cho mô hình thấy tương lai khi đoán quá khứ. |
| `gap` để làm gì? | Để target 24h của tập trước không rơi vào khoảng thời gian của tập sau. |
| Vì sao walk-forward? | Một khối Validation chỉ rơi vào một mùa; walk-forward chấm cả 11 tháng, kể cả mùa đông, đúng như cách dùng thật. |
| Vì sao fit preprocessing trong Pipeline? | Để median/IQR/μ/σ chỉ học trên dữ liệu train của từng lần fit (kể cả từng fold CV và walk-forward). |
| Vì sao `ffill` mà không nội suy? | Nội suy dùng giá trị sau thời điểm dự báo → rò rỉ tương lai. |
| Vì sao ML thua blend ở tháng chuyển mùa? | Bias âm khi mức tăng, dương khi mức giảm: mô hình bám quan hệ trung bình quá khứ; blend bám mức hiện tại. |
| Vì sao $R^2$ âm? | Dự báo tệ hơn việc luôn đoán trung bình của chính tập đó. |
| Vì sao 18 giờ/ngày? | 75% số giờ — quy tắc đầy đủ dữ liệu cho trung bình ngày; ít hơn thì trung bình không đại diện. |
| Vì sao k = 3 cho IQR? | Biên ngoài của Tukey (ngoại lai cực đoan); chỉ cắt giá trị rất xa, không đụng đỉnh vừa. |
| Vì sao 2.000 lần bootstrap? | Đủ để phân vị 2,5%/97,5% ổn định (mỗi đuôi ~50 giá trị); tăng thêm chỉ đổi số lẻ. |
| Vì sao block 3 ngày? | Tự tương quan của chênh lệch lỗi theo ngày nhỏ; đã thử 1 và 7 ngày, chỉ trường hợp sát biên (Hà Nội walk-forward) đổi. |
| Tham số ở biên grid có sao không? | Đã mở rộng hai lần, mỗi lần dời về phía đơn giản hơn; ghi là hạn chế. Khi chạy thử ba phiên bản grid, gain ML vs blend gần như không đổi (quan sát khi phát triển, không có trong notebook). |
