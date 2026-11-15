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
  yếu tố thời tiết liên quan khác nhau (Hà Nội: mưa/mây/áp suất; HCMC: gió).

**Câu hỏi có thể gặp:** *Sao không gộp hai thành phố?* → Hai cơ chế ô nhiễm khác nhau; gộp lại thì mô hình phải
học thêm biến "thành phố" và kết quả bị trộn lẫn giai đoạn Test khác nhau. Mục tiêu bài là **so sánh** hai nơi.

---

## 1. Mô tả dữ liệu & EDA (`part1_overview.py`)

### 1.1 Thu thập (`scripts/crawl_air_quality.py`)
- `fetch_pm25_openaq`: gọi `GET /v3/sensors/{id}/hours` theo từng năm, mỗi trang 1000 bản ghi, ngủ 1,1 s giữa các
  trang (giới hạn 60 request/phút). Lấy `period.datetimeFrom.local` làm thời gian, `value` làm PM2.5.
- `fetch_weather`: Open-Meteo archive, theo giờ, múi giờ `Asia/Ho_Chi_Minh`.
- `build_city_dataset`: tạo **lưới giờ đầy đủ** `pd.date_range(..., freq="h")` rồi `reindex` → giờ không có số đo
  thành `NaN`. Đây là điểm quan trọng: nếu không có lưới đầy đủ, `shift(k)` sẽ lùi $k$ **dòng** chứ không phải $k$ **giờ**.
- Nếu không có API key → fallback sang Open-Meteo Air Quality (dữ liệu mô hình CAMS) và ghi vào cột `pm25_source`.
  Dữ liệu nộp bài là `openaq` (cell 1.2 in ra nguồn).

### 1.2 `load_city_data`
Đọc CSV ở các thư mục cục bộ; không có thì tải từ `GITHUB_RAW_BASE` (để Colab "Run all" được).

### 1.3 Thống kê mô tả, `daily_means`
- `describe()`: count, mean, std, min, quartiles, max.
- **Ngưỡng WHO 15 µg/m³ là cho trung bình 24 giờ.** Vì vậy so với **trung bình ngày**, không so từng giờ.
- `daily_means(s, min_hours=18)`: trung bình ngày chỉ tính cho ngày có ≥ 18/24 giờ đo (75%); ngày ít hơn → `NaN`.
  Lý do: trung bình của 3–4 giờ không đại diện cho cả ngày. 75% là quy tắc đầy đủ dữ liệu thường dùng cho trung bình ngày.
- Kết quả: vượt ngưỡng ở 91,3% số ngày hợp lệ (Hà Nội, 389 ngày) và 98,2% (HCMC, 613 ngày).

**Hỏi:** *Sao Hà Nội chỉ có 389 ngày hợp lệ trên ~669 ngày?* → Vì từ 11/2025 trạm thiếu rất nhiều (bảng % thiếu theo tháng ở 2.1).

### 1.4 Hình 1–3
- Hình 1: `daily_means` theo ngày; đường đứt quãng ở những ngày không đủ 18 giờ.
- Hình 3 là **boxplot = trung vị và tứ phân vị**; bảng in ngay dưới có cả **mean** và **median** theo tháng.
  Notes dùng mean (ghi rõ), và nói median thấp hơn vài µg/m³ vì phân phối lệch phải.

### 1.5 Hình 4 – chu kỳ ngày, `complete_days_only`
- Chỉ giữ giờ của những ngày có ≥ 18 giờ đo: `s.groupby(s.index.floor('D')).transform('count') >= 18`
  (`transform` trả về số giờ đo của ngày đó cho **mỗi** dòng, rồi lọc).
- **Vì sao bắt buộc lọc:** sau 11/2025 trạm Hà Nội gần như không báo buổi chiều. Nếu lấy trung bình theo giờ trên
  mọi ngày, giờ 15h chủ yếu đến từ giai đoạn cũ, giờ 0h đến từ cả hai giai đoạn → đường cong phản ánh *ngày nào có dữ liệu*
  chứ không phải chu kỳ thật (trước khi lọc, Hà Nội mùa khô hiện một đỉnh giả lúc 12h). Đây là dữ liệu thiếu **không ngẫu nhiên**
  (MNAR – missing not at random).

### 1.6 Hình 5 – Spearman
- $\rho_S$ = hệ số Pearson tính trên **thứ hạng** (rank) của hai biến → đo quan hệ **đơn điệu**, ít nhạy với ngoại lai,
  không cần quan hệ tuyến tính. Phù hợp vì PM2.5 lệch phải mạnh.
- Đây là **tương quan, không phải nhân quả**: mùa đông vừa lạnh, vừa áp suất cao, vừa ít mưa → các hệ số lẫn nhau.

### 1.7 Bảng tự tương quan (autocorrelation)
- $r(k) = \mathrm{corr}(y_t, y_{t-k})$ trên các cặp mà cả hai giờ đều có số đo. Code: `Series.autocorr(k)`
  (pandas tính bằng `corr` với chính chuỗi dịch $k$ dòng; vì lưới giờ đầy đủ nên $k$ dòng = $k$ giờ).
- 0,92 ở lag 1h nhưng 0,56 (Hà Nội) / 0,45 (HCMC) ở lag 24h → dự báo 24h khó hơn nhiều so với 1h.
- HCMC: $r(24) = 0{,}45 > r(12) = 0{,}27$ vì cùng giờ trong ngày có xu hướng lặp lại (chu kỳ ngày).

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
- Bảng % thiếu theo tháng: `isna().resample('MS').mean() * 100` (`MS` = đầu tháng).

### 2.2 `clean_city`
1. **Giá trị sai:** `pm <= 0` hoặc `pm > 1000` → `NaN`. Trạm chuẩn báo đúng 0,0 là mất tín hiệu, không phải không khí sạch.
2. **Cảm biến bị kẹt:** cùng mẹo `cumsum` ở trên với `pm != pm.shift()` để tìm các đoạn **cùng giá trị**;
   `transform('size')` = độ dài đoạn; nếu ≥ 6 giờ và > 0 → `NaN`. (`NaN != NaN` là True nên mỗi NaN là một đoạn riêng,
   không bị tính là "kẹt".) Kết quả: không có giờ nào bị kẹt.
3. Giữ hai cột:
   - `pm25_raw` = chỉ giá trị đo → dùng làm **target**.
   - `pm25` = `pm25_raw.ffill(limit=6)` → dùng làm **đặc trưng**. `ffill(limit=6)` lấp **tối đa 6 giờ liên tiếp**
     bằng giá trị đo gần nhất trước đó (kể cả 6 giờ đầu của một khoảng trống dài hơn).

**Vì sao không dùng `interpolate`:** nội suy tuyến tính giữa $y_a$ và $y_b$ ($a < t < b$):
$\tilde y_t = y_a + (y_b - y_a)\frac{t-a}{b-a}$ — dùng $y_b$ là số đo **sau** $t$, lúc dự báo thật chưa có →
**rò rỉ tương lai** (look-ahead leakage). `ffill` chỉ dùng $y_a$ ($a < t$) → **nhân quả** (causal).
Cell 2.4 in ra tỉ lệ mẫu có giá trị hiện tại là giá trị đã lấp (2,8% Hà Nội, 0,9% HCMC).

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

### 2.5 `time_split`
- $n$ mẫu, $t_1$ = thời điểm của mẫu thứ $\lfloor 0{,}70n \rfloor$, $t_2$ = của mẫu thứ $\lfloor 0{,}85n \rfloor$.
- Train: $t < t_1 - 24h$; Validation: $t_1 \le t < t_2 - 24h$; Test: $t \ge t_2$.
- **Vì sao bỏ 24h:** mẫu train cuối cùng ở $t < t_1 - 24h$ có target ở $t + 24 < t_1$ → không target nào của Train rơi vào
  thời gian của Validation (tương tự giữa Validation và Test).
- **Không xáo trộn (shuffle):** xáo trộn thì mô hình học từ tương lai để đoán quá khứ.
- Đặc trưng của Validation dùng các giá trị PM2.5 trong thời gian Train — đó là **quá khứ**, hợp lệ.

### 2.6 `IQRClipper`, `build_preprocessor`
```python
class IQRClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):      # học biên trên dữ liệu fit (chỉ Train)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        self.lower_, self.upper_ = q1 - k*(q3-q1), q3 + k*(q3-q1)
    def transform(self, X):        # x' = min(max(x, lower), upper)
        return np.clip(X, self.lower_, self.upper_)
```
- Kế thừa `BaseEstimator` (có `get_params`/`set_params`, dùng được trong `GridSearchCV`) và `TransformerMixin`
  (có `fit_transform`). Thuộc tính có dấu `_` cuối = học từ dữ liệu (quy ước sklearn).
- `get_feature_names_out` trả lại tên cột để `ColumnTransformer` đặt tên đầu ra.
- `ColumnTransformer`:
  - PM2.5: `SimpleImputer(median)` → `IQRClipper(3)` → `StandardScaler`
  - thời tiết + sin/cos: `SimpleImputer(median)` → `StandardScaler`
  - categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`
- `StandardScaler`: $z = (x - \mu)/\sigma$, $\mu, \sigma$ tính trên Train (σ là độ lệch chuẩn tổng thể, ddof = 0).
- `OneHotEncoder`: mỗi giá trị → một cột 0/1. `handle_unknown='ignore'`: giá trị chưa gặp ở Train → toàn 0, không lỗi.
- 23 cột gốc → 39 cột sau one-hot (cell in ra).
- **Không rò rỉ:** mọi tham số (median, biên IQR, μ, σ, danh mục) học bằng `fit` trên Train; Validation/Test chỉ `transform`.

---

## 3. Mô hình hoá (`part3_modeling.py`)

### 3.1 Baselines
| Tên | Công thức |
|---|---|
| `PersistenceBaseline` | $\hat y_{t+24} = y_t$ |
| `MeanBaseline` | $\hat y_{t+24} = \bar y_{t-23..t}$ (thiếu thì dùng $y_t$) |
| `NaiveBlend` | $\hat y = w\,y_t + (1-w)\,\bar y_{24}$, $\ w^* = \arg\min_{w \in \{0, 0{,}05, …, 1\}} \frac1n\sum \lvert y - \hat y_w\rvert$ trên **Train** |

- `w` học trên Train như mọi tham số khác (0,45 Hà Nội, 0,50 HCMC). `argmin` trên lưới 21 giá trị; nếu hoà thì lấy giá trị đầu.
- **Vì sao cần Naive blend:** Persistence rất yếu ($R^2 \approx 0$). Nếu chỉ so với nó, mô hình học máy trông tốt hơn thực tế.

### 3.2 Chỉ số đánh giá
- $\mathrm{MAE} = \frac1n\sum|y_i - \hat y_i|$ (cùng đơn vị µg/m³, ít nhạy ngoại lai) — **chỉ số chính**.
- $\mathrm{RMSE} = \sqrt{\frac1n\sum(y_i - \hat y_i)^2}$ (phạt lỗi lớn mạnh hơn).
- $R^2 = 1 - \frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}$, $\bar y$ = trung bình của **chính tập đang đánh giá**.
  $R^2 < 0$ nghĩa là dự báo tệ hơn việc luôn đoán $\bar y$ (Persistence của HCMC trên Validation: −0,129).

**Hỏi:** *Sao chọn theo MAE mà không theo RMSE?* → Dễ hiểu (sai trung bình bao nhiêu µg/m³), và với phân phối lệch phải
RMSE bị vài đỉnh chi phối. Hệ quả: tối thiểu MAE khiến mô hình học **trung vị có điều kiện** → kéo dự báo về giữa,
dự báo thấp giờ ô nhiễm (thấy rõ ở 4.5).

### 3.3 Ba mô hình
- **Ridge:** $\min_{w,b} \|y - Xw - b\|_2^2 + \alpha\|w\|_2^2$ (intercept $b$ không bị phạt). Nghiệm đóng
  $w = (X^\top X + \alpha I)^{-1}X^\top y$ (trên dữ liệu đã trừ trung bình). $\alpha$ lớn → hệ số nhỏ, ít overfit.
  Cần chuẩn hoá trước để phạt công bằng giữa các đặc trưng. Grid: $\alpha \in \{0{,}1; 1; 10; 100\}$ → chọn 100 ở cả hai thành phố.
- **RandomForest:** 200 cây; mỗi cây học trên một **bootstrap sample** (rút $n$ dòng có hoàn lại); tại mỗi nút chỉ xét
  ngẫu nhiên 50% số cột (`max_features=0.5`, tính trên 39 cột sau one-hot) và chọn phép chia giảm MSE nhiều nhất;
  dự báo = trung bình dự báo của 200 cây (bagging → giảm phương sai). `max_depth` ∈ {8, 12}, `min_samples_leaf` ∈ {10, 30} (tune).
- **HistGradientBoosting (HGB):** boosting — cây thứ $m$ học phần dư (gradient âm của sai số bình phương) của tổng các cây trước,
  $F_m = F_{m-1} + \eta\, h_m$ với $\eta$ = `learning_rate`. "Hist": mỗi đặc trưng được chia sẵn thành ≤ 255 bin → tìm phép chia nhanh.
  Giá trị lá $= -G/(H + \lambda)$ ($G$, $H$ = tổng gradient/hessian trong lá, $\lambda$ = `l2_regularization`).
  `max_iter` = số cây, `max_leaf_nodes` = số lá tối đa mỗi cây.
  **`early_stopping=False`**: mặc định `'auto'` bật early stopping khi > 10.000 mẫu và khi đó giữ lại **10% dòng ngẫu nhiên**
  (không theo thời gian) để dừng sớm. Train HCMC có 10.177 dòng → nếu để mặc định, HGB chạy khác nhau trong CV (< 10.000 dòng) và lúc fit cuối.
  Tắt đi để số cây luôn đúng bằng `max_iter` đã tune.

### 3.4 `GridSearchCV` + `TimeSeriesSplit(n_splits=3, gap=24)`
- `TimeSeriesSplit`: chia Train thành 4 phần bằng nhau theo thứ tự; fold $i$ train trên phần đầu, validate trên phần kế tiếp
  (cửa sổ mở rộng). Cụ thể: `test_size` $= \lfloor n/4 \rfloor$; fold validate bắt đầu ở $n - (3-i)\cdot$`test_size`;
  train = các mẫu **trước** điểm đó trừ đi `gap` = 24 mẫu.
- **Vì sao `gap=24`:** 24 mẫu cuối của fold train có target nằm trong khoảng thời gian của fold validate. Vì mỗi giờ có tối đa
  1 mẫu, 24 mẫu trải ít nhất 24 giờ → đủ khoảng đệm.
- `scoring='neg_mean_absolute_error'`: sklearn luôn **tối đa hoá** điểm nên dùng −MAE.
- `refit=True` (mặc định): fit lại bộ tham số tốt nhất trên toàn Train.

### 3.5 `CityForecaster`
- `tune()`: chạy GridSearch cho Ridge, RF, HGB.
- `evaluate_on_validation()`: fit trên Train, dự báo Validation, lưu `val_predictions` (để bootstrap).
- `evaluate_on_test()`: fit lại trên **Train + Validation** (tham số giữ nguyên), dự báo Test **một lần**.
- Chọn mô hình tốt nhất = MAE Validation nhỏ nhất **trong 3 mô hình học máy** (không dùng Test để chọn).

---

## 4. Tổng kết & mở rộng (`part4_summary.py`)

### 4.1–4.2 Bảng Test, Hình 6
- "MAE vs X" $= 1 - \mathrm{MAE}_{\text{mô hình}}/\mathrm{MAE}_X$; dương = tốt hơn X.

### 4.3 Block bootstrap (`block_bootstrap_gain`)
Mục tiêu: khoảng tin cậy 95% cho $g = 1 - \mathrm{MAE}_A/\mathrm{MAE}_B$ ($A$ = mô hình tốt nhất, $B$ = đối thủ).

1. Gán mỗi mẫu vào khối 3 ngày theo thời điểm dự báo:
   `codes = factorize((t - t.min()) // 3 ngày)` → $k$ khối **không chồng nhau** (khối không có mẫu nào thì không tồn tại).
2. Tổng lỗi mỗi khối: $S^A_j = \sum_{i\in j}|y_i - \hat y^A_i|$ — `np.bincount(codes, weights=...)`.
3. Rút lại $k$ khối có hoàn lại = vector đếm $c \sim \mathrm{Multinomial}(k;\ 1/k,…,1/k)$ (`rng.multinomial`), lặp 2.000 lần.
4. $g^* = 1 - \sum_j c_j S^A_j / \sum_j c_j S^B_j$ (`counts @ abs_err_a`). Số giờ triệt tiêu vì $A$ và $B$ được chấm trên **cùng** giờ (paired).
5. CI 95% = phân vị 2,5% và 97,5% của 2.000 giá trị $g^*$ (**percentile interval**).

- **Vì sao theo khối:** sai số của giờ liền nhau tương quan mạnh. Bootstrap từng giờ (i.i.d.) coi 2.000 giờ là 2.000 quan sát độc lập
  → CI quá hẹp. (Kiểm tra riêng, ngoài notebook: trên dữ liệu giả có tự tương quan, CI tính theo từng giờ hẹp hơn khoảng 4,6 lần CI theo khối.)
- **Vì sao 3 ngày:** cell in tự tương quan của chênh lệch sai số trung bình ngày trên **Validation**: 0,12 / 0,02 / 0,10 (Hà Nội) và
  0,20 / 0,12 / 0,14 (HCMC) ở lag 1/2/3 ngày → nhỏ sau 1 ngày. Kiểm tra lại với khối 1 và 7 ngày: kết luận không đổi.
- **Diễn giải:** CI không chứa 0 ≈ bác bỏ "không khác nhau" ở mức 5% (kiểm định hai phía, xấp xỉ).
- **Nguồn:** non-overlapping block bootstrap — Carlstein (1986); ý tưởng bootstrap theo khối cho dữ liệu phụ thuộc — Künsch (1989)
  (Künsch dùng khối **chồng nhau**, *moving block*; bài này dùng khối không chồng).
- **Hạn chế (đã ghi trong notebook):** 13–31 khối → CI xấp xỉ; 16 khoảng không hiệu chỉnh đa so sánh → chỉ mang tính mô tả;
  kết luận dựa trên **2 so sánh định trước** (mô hình tốt nhất vs Naive blend trên Test, mỗi thành phố một); giả định chuỗi lỗi gần dừng.

### 4.4 Dự báo vs thực tế (Hình 8)
MAE trong 14 ngày đầu và so sánh độ lệch chuẩn của dự báo với thực tế (dự báo trơn hơn → bỏ sót đỉnh).

### 4.5 Phân tích lỗi
- MAE theo giờ/tháng; giờ/tháng có < 20 mẫu Test không vẽ (quá nhiễu; Hà Nội thiếu buổi chiều).
- `to_epa_band`: cắt cụt về 0,1 µg/m³ ($\lfloor 10x \rfloor/10$, đúng quy tắc EPA) rồi `pd.cut` với các khoảng **đóng bên phải**
  $(-\infty, 9{,}0], (9{,}0; 35{,}4], (35{,}4; 55{,}4], (55{,}4; 125{,}4], (125{,}4; \infty)$.
- **Không gọi là AQI:** AQI chính thức của EPA tính trên trung bình 24 giờ (hoặc NowCast); ở đây áp ngưỡng cho giá trị theo **giờ**
  → gọi là "dải nồng độ EPA".
- Ma trận nhầm lẫn: `pd.crosstab(thực, dự báo)`; tô màu theo tỉ lệ mỗi hàng, ghi số đếm.

### 4.6 Permutation importance
- $I_j = \frac1R\sum_{r=1}^{R}\big[\mathrm{MAE}(X^{(j,r)}) - \mathrm{MAE}(X)\big]$, $X^{(j,r)}$ = Test với cột $j$ bị xáo trộn lần $r$, $R = 5$.
  Mô hình **không** train lại.
- **Vì sao không dùng `feature_importances_` của cây:** đó là importance theo impurity, tính trên Train, thiên vị cột nhiều giá trị,
  và không áp dụng được cho Ridge.
- **Theo nhóm (`group_importance`):** xáo trộn **cùng một hoán vị** cho mọi cột trong nhóm → giữ quan hệ giữa các cột trong nhóm,
  phá quan hệ với target. Cần vì các lag PM2.5 tương quan mạnh: xáo một cột thì cột khác "bù" lại → importance từng cột bị chia nhỏ.

### 4.7 `predict_next`
Nhận lịch sử theo giờ đến $t$ → `build_features` → lấy dòng cuối → mô hình tốt nhất dự báo $t+24$ → dải EPA.
Demo dùng lịch sử kết thúc 24h trước giá trị đo cuối cùng để so được với thực tế.

---

## 5. Những điểm yếu nên tự nói ra trước (thầy hỏi thì đã có câu trả lời)
1. Mỗi thành phố một trạm; HCMC là cảm biến giá rẻ.
2. Trạm Hà Nội suy giảm từ 11/2025 (31–97% thiếu mỗi tháng, thiếu buổi chiều) → Validation/Test của Hà Nội nhỏ và lệch về ban đêm.
3. Validation quá nhỏ để phân biệt các mô hình học máy → việc chọn RandomForest ở Hà Nội chưa chắc chắn (RF vs Ridge: +0,8%, CI [−9,0%; +8,9%]).
4. HCMC: chưa chứng minh học máy hơn Naive blend (CI [−1,3%; +11,8%]).
5. Thời tiết dùng giá trị đo tại $t$, không phải dự báo thời tiết cho $t+24$.
6. Tối thiểu MAE → dự báo thấp giờ ô nhiễm; chưa dùng được để cảnh báo sớm.

## 6. Câu hỏi nhanh – trả lời một câu
| Câu hỏi | Trả lời |
|---|---|
| Vì sao không shuffle khi chia tập? | Shuffle cho mô hình thấy tương lai khi đoán quá khứ. |
| `gap` để làm gì? | Để target 24h của tập trước không rơi vào khoảng thời gian của tập sau. |
| Vì sao fit preprocessing trong Pipeline? | Để median/IQR/μ/σ chỉ học trên Train, kể cả trong từng fold CV. |
| Vì sao `ffill` mà không nội suy? | Nội suy dùng giá trị sau thời điểm dự báo → rò rỉ tương lai. |
| Vì sao Ridge thắng ở HCMC? | Tín hiệu yếu ($R^2 \le 0{,}15$); mô hình tuyến tính bị phạt mạnh ($\alpha = 100$) ít overfit hơn cây. |
| Vì sao $R^2$ âm? | Dự báo tệ hơn việc luôn đoán trung bình của chính tập đó. |
| Vì sao 18 giờ/ngày? | 75% số giờ — quy tắc đầy đủ dữ liệu cho trung bình ngày; ít hơn thì trung bình không đại diện. |
| Vì sao k = 3 cho IQR? | Biên ngoài của Tukey (ngoại lai cực đoan); chỉ cắt giá trị rất xa, không đụng đỉnh vừa. |
| Vì sao 2.000 lần bootstrap? | Đủ để phân vị 2,5%/97,5% ổn định (mỗi đuôi ~50 giá trị); tăng thêm chỉ đổi số lẻ. |
| Vì sao block 3 ngày? | Tự tương quan của chênh lệch lỗi theo ngày nhỏ sau lag 1; đã thử 1 và 7 ngày, kết luận giữ nguyên. |
