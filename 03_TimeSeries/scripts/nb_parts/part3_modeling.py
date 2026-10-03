# Phần 3: Phân tích và mô hình hóa
md("""
## 3. PHÂN TÍCH VÀ MÔ HÌNH HÓA

### 3.1. Các mô hình so sánh
| Mô hình | Vai trò | Lý do chọn |
|---|---|---|
| **Persistence (baseline)** | `ŷ(t+24) = pm25(t)` | Baseline chuẩn cho chuỗi thời gian: mô hình ML phải **thắng** được nó mới có giá trị |
| **Ridge Regression** | Tuyến tính có regularization | Đơn giản, dễ giải thích; kiểm tra mức độ quan hệ tuyến tính |
| **Random Forest** | Ensemble cây (bagging) | Bắt được quan hệ phi tuyến, tương tác thời tiết × giờ |
| **HistGradientBoosting** | Ensemble cây (boosting) | Mạnh và nhanh trên dữ liệu bảng lớn |

**Tối ưu siêu tham số:** `GridSearchCV` với `TimeSeriesSplit(n_splits=3)` trên tập train (fold validation luôn nằm
**sau** fold train), tiêu chí `MAE`. Random Forest dùng cấu hình cố định (giới hạn `max_depth`, `min_samples_leaf`
để chống overfit) nhằm giữ thời gian chạy hợp lý trên Colab.

Lớp `CityForecaster` đóng gói toàn bộ quy trình cho **một** thành phố, để chạy lặp lại cho cả hai.
""")

code("""
class PersistenceBaseline(BaseEstimator, RegressorMixin):
    \"\"\"Dự báo PM2.5 sau HORIZON giờ bằng giá trị hiện tại.\"\"\"

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X['pm25_lag_0'], dtype=float)


def regression_metrics(y_true, y_pred):
    return {'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)}


def make_models():
    return {
        'Persistence (baseline)': PersistenceBaseline(),
        'Ridge': Pipeline([('pre', build_preprocessor()), ('model', Ridge())]),
        'RandomForest': Pipeline([('pre', build_preprocessor()), ('model', RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=10, max_features=0.5,
            n_jobs=-1, random_state=RANDOM_STATE))]),
        'HistGradientBoosting': Pipeline([('pre', build_preprocessor()),
                                          ('model', HistGradientBoostingRegressor(random_state=RANDOM_STATE))]),
    }


PARAM_GRIDS = {
    'Ridge': {'model__alpha': [0.1, 1, 10, 100]},
    'HistGradientBoosting': {'model__learning_rate': [0.03, 0.1],
                             'model__max_iter': [200, 400],
                             'model__max_leaf_nodes': [15, 31],
                             'model__l2_regularization': [0.0, 1.0]},
}


class CityForecaster:
    def __init__(self, city, train, val, test):
        self.city, self.train, self.val, self.test = city, train, val, test
        self.models = make_models()
        self.best_params = {}

    def tune(self):
        for name, grid in PARAM_GRIDS.items():
            search = GridSearchCV(self.models[name], grid, cv=TimeSeriesSplit(n_splits=3),
                                  scoring='neg_mean_absolute_error', n_jobs=-1)
            search.fit(self.train[FEATURES], self.train[TARGET])
            self.models[name] = search.best_estimator_
            self.best_params[name] = search.best_params_
        return self

    def evaluate_on_validation(self):
        rows = []
        for name, m in self.models.items():
            m.fit(self.train[FEATURES], self.train[TARGET])
            rows.append({'city': self.city, 'model': name,
                         **regression_metrics(self.val[TARGET], m.predict(self.val[FEATURES]))})
        return pd.DataFrame(rows)

    def evaluate_on_test(self):
        \"\"\"Huấn luyện lại trên Train + Validation rồi đánh giá một lần trên Test.\"\"\"
        full = pd.concat([self.train, self.val])
        rows, self.test_predictions = [], pd.DataFrame({'actual': self.test[TARGET]})
        for name, m in self.models.items():
            m.fit(full[FEATURES], full[TARGET])
            pred = m.predict(self.test[FEATURES])
            self.test_predictions[name] = pred
            rows.append({'city': self.city, 'model': name, **regression_metrics(self.test[TARGET], pred)})
        return pd.DataFrame(rows)


forecasters = {city: CityForecaster(city, *parts).tune() for city, parts in splits.items()}
for city, f in forecasters.items():
    print(f'{city} – siêu tham số tốt nhất:')
    for name, params in f.best_params.items():
        print(f'   {name}: {params}')
""")

md("### 3.2. So sánh trên tập Validation")

code("""
val_results = pd.concat([f.evaluate_on_validation() for f in forecasters.values()], ignore_index=True)
display(val_results.pivot(index='model', columns='city', values=['MAE', 'RMSE', 'R2']).round(3))
""")

md(note("VAL"))
