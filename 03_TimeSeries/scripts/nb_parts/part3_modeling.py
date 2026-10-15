# Section 3: analysis and modeling
md("""
## 3. ANALYSIS AND MODELING

### 3.1. Models compared
| Model | Role | Why |
|---|---|---|
| **Persistence (baseline)** | `ŷ(t+24) = pm25(t)` | Standard time series baseline: an ML model is only useful if it **beats** it |
| **Ridge Regression** | Regularized linear model | Simple and interpretable; tests how linear the relationship is |
| **Random Forest** | Tree ensemble (bagging) | Captures non-linear effects and weather × hour interactions |
| **HistGradientBoosting** | Tree ensemble (boosting) | Strong and fast on large tabular data |

**Hyperparameter tuning:** `GridSearchCV` with `TimeSeriesSplit(n_splits=3)` on the training set (the validation fold always comes
**after** the training fold), scored by `MAE`. Random Forest uses a fixed configuration (limited `max_depth` and `min_samples_leaf`
against overfitting) to keep the run time reasonable on Colab.

The `CityForecaster` class wraps the whole workflow for **one** city so that it can be repeated for both.
""")

code("""
class PersistenceBaseline(BaseEstimator, RegressorMixin):
    \"\"\"Forecast PM2.5 HORIZON hours ahead with the current value.\"\"\"

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
        \"\"\"Retrain on Train + Validation, then evaluate once on Test.\"\"\"
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
    print(f'{city} – best hyperparameters:')
    for name, params in f.best_params.items():
        print(f'   {name}: {params}')
""")

md("### 3.2. Comparison on the Validation set")

code("""
val_results = pd.concat([f.evaluate_on_validation() for f in forecasters.values()], ignore_index=True)
val_table = val_results.pivot(index='model', columns='city', values=['MAE', 'RMSE', 'R2'])
display(val_table.reindex(columns=list(CITY_FILES), level=1).reindex(list(make_models())).round(3))
""")

md(note("VAL"))
