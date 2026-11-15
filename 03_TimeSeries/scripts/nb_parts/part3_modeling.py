# Section 3: analysis and modeling
md("""
## 3. ANALYSIS AND MODELING

### 3.1. Models compared
| Model | Role | Why |
|---|---|---|
| **Persistence (baseline)** | `ŷ(t+24) = pm25(t)` | Standard time series baseline (= 24h seasonal naive) |
| **Mean 24h (baseline)** | `ŷ(t+24) = mean(pm25[t-23 … t])` | Smooths out the noise of a single hour |
| **Naive blend (baseline)** | `ŷ = w·pm25(t) + (1−w)·mean_24h(t)` | The **strongest naive baseline**: `w` is learned on the training data (grid 0, 0.05, …, 1, minimizing MAE). An ML model is only really useful if it beats **this** one, not just Persistence |
| **Ridge Regression** | Regularized linear model | Simple and interpretable; tests how linear the relationship is |
| **Random Forest** | Tree ensemble (bagging) | Captures non-linear effects and weather × hour interactions |
| **HistGradientBoosting** | Tree ensemble (boosting) | Strong and fast on large tabular data |

**Hyperparameter tuning:** `GridSearchCV` with `TimeSeriesSplit(n_splits=3, gap=24)` on the **Development** set, scored by `MAE`.
The validation fold always comes **after** the training fold, and the `gap` drops 24 samples between them so that the targets
(24h ahead) of the last training samples do not fall inside the validation fold. All three ML models are tuned
(Random Forest with a small grid to keep the run time reasonable on Colab). The printout flags any selected value that lies at
the smallest or largest end of its grid, because the best value could then lie outside the grid, and shows the best and worst
cross-validated MAE over the grid, i.e. how much the choice of hyperparameters matters at all for each model. (For Ridge, the smallest `alpha` is not a concern: as `alpha` → 0 Ridge tends to ordinary least squares.) The hyperparameters are then **fixed** for every
walk-forward fold and for the final model.

**Model selection:** walk-forward validation (Section 2.5). Every model, baselines included, is refitted on each training window
$\\text{Train}_m$ (so the blend weight $w$ is also re-learned in every fold) and forecasts month $m$. The scores are computed on the
**pooled** forecasts of the 11 months, e.g. $\\mathrm{MAE} = \\sum_m \\sum_{i \\in \\text{Val}_m} |y_i - \\hat y_i| \\, / \\, \\sum_m |\\text{Val}_m|$,
and the ML model with the lowest pooled MAE is the **best model** of the city.

*Stated limitation:* the hyperparameters were tuned on the whole Development set, which contains the walk-forward months, so the
walk-forward errors of the three ML models are **slightly optimistic** (the baselines have no hyperparameters). Re-tuning inside each
of the 11 folds (nested tuning) would remove this bias but multiplies the run time by about 11. The Test set (Section 4) is not
affected: it was never used for tuning or selection.

The `CityForecaster` class wraps the whole workflow for **one** city so that it can be repeated for both.
""")

code("""
class PersistenceBaseline(BaseEstimator, RegressorMixin):
    \"\"\"Forecast PM2.5 HORIZON hours ahead with the current value.\"\"\"

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X['pm25_lag_0'], dtype=float)


class MeanBaseline(BaseEstimator, RegressorMixin):
    \"\"\"Forecast PM2.5 HORIZON hours ahead with the mean of the last 24 hours.\"\"\"

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X['pm25_roll_mean_24'].fillna(X['pm25_lag_0']), dtype=float)


class NaiveBlend(BaseEstimator, RegressorMixin):
    \"\"\"w·(current value) + (1 − w)·(24h mean); w is chosen on the fitted data to minimize MAE.\"\"\"

    def __init__(self, weights=tuple(np.round(np.linspace(0, 1, 21), 2))):
        self.weights = weights

    @staticmethod
    def _parts(X):
        now = np.asarray(X['pm25_lag_0'], dtype=float)
        return now, np.asarray(X['pm25_roll_mean_24'].fillna(X['pm25_lag_0']), dtype=float)

    def fit(self, X, y):
        now, mean24 = self._parts(X)
        maes = [np.mean(np.abs(np.asarray(y) - (w * now + (1 - w) * mean24))) for w in self.weights]
        self.w_ = float(self.weights[int(np.argmin(maes))])
        return self

    def predict(self, X):
        now, mean24 = self._parts(X)
        return self.w_ * now + (1 - self.w_) * mean24


BASELINES = ['Persistence (baseline)', 'Mean 24h (baseline)', 'Naive blend (baseline)']


def regression_metrics(y_true, y_pred):
    return {'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)}


def make_models():
    return {
        'Persistence (baseline)': PersistenceBaseline(),
        'Mean 24h (baseline)': MeanBaseline(),
        'Naive blend (baseline)': NaiveBlend(),
        'Ridge': Pipeline([('pre', build_preprocessor()), ('model', Ridge())]),
        'RandomForest': Pipeline([('pre', build_preprocessor()), ('model', RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=10, max_features=0.5,
            n_jobs=-1, random_state=RANDOM_STATE))]),
        # early_stopping=False: the default 'auto' switches early stopping on above 10,000 samples and then holds out a
        # *random* 10% of the rows (not a time-ordered split), so the model would behave differently in the CV folds
        # (< 10,000 rows) and in the final fit; with False the number of trees is always exactly max_iter.
        'HistGradientBoosting': Pipeline([('pre', build_preprocessor()),
                                          ('model', HistGradientBoostingRegressor(early_stopping=False,
                                                                                  random_state=RANDOM_STATE))]),
    }


PARAM_GRIDS = {
    'Ridge': {'model__alpha': [0.01, 0.1, 1, 10, 100, 1000]},
    'RandomForest': {'model__max_depth': [4, 6, 8, 12],
                     'model__min_samples_leaf': [5, 10, 30, 100]},
    'HistGradientBoosting': {'model__learning_rate': [0.01, 0.03, 0.1],
                             'model__max_iter': [100, 200, 400],
                             'model__max_leaf_nodes': [4, 7, 15],
                             'model__l2_regularization': [1.0, 10.0, 100.0]},
}


class CityForecaster:
    def __init__(self, city, dev, test):
        self.city, self.dev, self.test = city, dev, test
        self.models = make_models()
        self.best_params, self.cv_mae_range = {}, {}

    def tune(self):
        for name, grid in PARAM_GRIDS.items():
            search = GridSearchCV(self.models[name], grid, cv=TimeSeriesSplit(n_splits=3, gap=HORIZON),
                                  scoring='neg_mean_absolute_error', n_jobs=-1)
            search.fit(self.dev[FEATURES], self.dev[TARGET])
            self.models[name] = search.best_estimator_
            self.best_params[name] = search.best_params_
            self.cv_mae_range[name] = (-search.cv_results_['mean_test_score'].max(),
                                       -search.cv_results_['mean_test_score'].min())
        return self

    def walk_forward(self):
        \"\"\"Refit every model on each expanding training window and forecast the next month (Section 2.5).\"\"\"
        parts = []
        for month, train, val in walk_forward_folds(self.dev):
            p = pd.DataFrame({'actual': val[TARGET], 'fold': str(month)})
            for name, model in self.models.items():
                m = clone(model)                      # same hyperparameters, fitted from scratch
                m.fit(train[FEATURES], train[TARGET])
                p[name] = m.predict(val[FEATURES])
            parts.append(p)
        self.wf_predictions = pd.concat(parts)
        return pd.DataFrame([{'city': self.city, 'model': name,
                              **regression_metrics(self.wf_predictions['actual'], self.wf_predictions[name])}
                             for name in self.models])

    def evaluate_on_test(self):
        \"\"\"Refit on the whole Development set, then evaluate once on Test.\"\"\"
        rows, self.test_predictions = [], pd.DataFrame({'actual': self.test[TARGET]})
        for name, m in self.models.items():
            m.fit(self.dev[FEATURES], self.dev[TARGET])
            pred = m.predict(self.test[FEATURES])
            self.test_predictions[name] = pred
            rows.append({'city': self.city, 'model': name, **regression_metrics(self.test[TARGET], pred)})
        return pd.DataFrame(rows)


forecasters = {city: CityForecaster(city, *parts).tune() for city, parts in splits.items()}
for city, f in forecasters.items():
    print(f'{city} – best hyperparameters:')
    for name, params in f.best_params.items():
        # a value at the smallest or largest end of its grid means the optimum may lie outside the grid
        at_edge = [k.replace('model__', '') for k, v in params.items()
                   if len(PARAM_GRIDS[name][k]) > 1 and v in (min(PARAM_GRIDS[name][k]), max(PARAM_GRIDS[name][k]))]
        best_cv, worst_cv = f.cv_mae_range[name]
        print(f'   {name}: {params} | at the edge of the grid: {at_edge if at_edge else "none"} | '
              f'CV MAE best {best_cv:.2f}, worst {worst_cv:.2f} over the grid')
    blend = clone(f.models['Naive blend (baseline)']).fit(f.dev[FEATURES], f.dev[TARGET])
    print(f'   Naive blend: w = {blend.w_:.2f} (weight of the current value, learned on the whole Development set)')
""")

md("""
### 3.2. Walk-forward validation (model selection)
Pooled scores over the 11 monthly folds (10/2025 – 08/2026), then the MAE of each model in each month.
""")

code("""
wf_results = pd.concat([f.walk_forward() for f in forecasters.values()], ignore_index=True)
wf_table = wf_results.pivot(index='model', columns='city', values=['MAE', 'RMSE', 'R2'])
display(wf_table.reindex(columns=list(CITY_FILES), level=1).reindex(list(make_models())).round(3))

ml_wf = wf_results[~wf_results.model.isin(BASELINES)]
best_model = ml_wf.loc[ml_wf.groupby('city')['MAE'].idxmin()].set_index('city')['model']
print('Best model (lowest pooled walk-forward MAE among the ML models):', best_model.to_dict())

for city, f in forecasters.items():
    w_per_fold = {}
    for month, train, val in walk_forward_folds(f.dev):
        w_per_fold[str(month)] = clone(f.models['Naive blend (baseline)']).fit(train[FEATURES], train[TARGET]).w_
    print(f'{city} – Naive blend weight w learned in each fold:', w_per_fold)
""")

md("""
**Figure – Walk-forward MAE by month** for every model. Each point is the MAE of the forecasts issued in that month by a model trained
only on data known before the month started. The bars show the mean measured PM2.5 (target) of the month, for context.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
monthly_mae = {}
for ax, (city, f) in zip(axes, forecasters.items()):
    p = f.wf_predictions
    abs_err = p[list(f.models)].sub(p['actual'], axis=0).abs()
    monthly_mae[city] = abs_err.groupby(p['fold']).mean()
    level = p.groupby('fold')['actual'].mean()
    ax2 = ax.twinx()
    ax2.bar(range(len(level)), level, color='lightgray', alpha=0.6, width=0.7)
    ax2.set_ylabel('Mean measured PM2.5 (µg/m³)', color='gray')
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    for name in f.models:
        style = dict(ls='--', lw=1.2) if name in BASELINES else dict(lw=2, marker='o', ms=4)
        ax.plot(range(len(level)), monthly_mae[city][name].reindex(level.index), label=name, **style)
    ax.set_xticks(range(len(level)), level.index, rotation=45)
    ax.set(title=f'{city} – walk-forward MAE by month', ylabel='MAE (µg/m³)')
axes[0].legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.show()

for city in CITY_FILES:
    print(f'{city} – walk-forward MAE by month (µg/m³):')
    display(monthly_mae[city].T.reindex(list(make_models())).round(2))
""")

md(note("VAL"))
