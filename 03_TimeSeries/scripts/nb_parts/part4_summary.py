# Section 4: summary and extensions
md("""
## 4. SUMMARY AND EXTENSIONS

### 4.1. Results on the independent Test set
Each model is retrained on Train + Validation (with its hyperparameters fixed) and evaluated **once** on Test.
The **best model of each city is chosen on the Validation set** (lowest Validation MAE, Section 3.2), not on Test,
so the Test set is used only for reporting and the error analysis below is not biased by model selection.
""")

code("""
test_results = pd.concat([f.evaluate_on_test() for f in forecasters.values()], ignore_index=True)
for ref, col in [('Persistence (baseline)', 'MAE vs Persistence'), ('Naive blend (baseline)', 'MAE vs Naive blend')]:
    ref_mae = test_results[test_results.model == ref].set_index('city')['MAE']
    test_results[col] = test_results.apply(lambda r: 1 - r['MAE'] / ref_mae[r['city']], axis=1)
display(test_results.set_index(['city', 'model']).style.format(
    {'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'R2': '{:.3f}', 'MAE vs Persistence': '{:+.1%}', 'MAE vs Naive blend': '{:+.1%}'}))
print('Test:', {c: f.test.index.min().strftime('%Y-%m-%d') + ' → ' + f.test.index.max().strftime('%Y-%m-%d')
                for c, f in forecasters.items()})
ml_val = val_results[~val_results.model.isin(BASELINES)]
best_model = ml_val.loc[ml_val.groupby('city')['MAE'].idxmin()].set_index('city')['model']
print('Best model (chosen on Validation):', best_model.to_dict())
""")

md("""
### 4.2. Question: *How much better than the baseline is machine learning in each city?*
**Figure 6 – Test MAE** of each model, per city. Bar labels: % MAE reduction vs. the **Naive blend** baseline
(the strongest naive forecast), so a negative label means "worse than the naive blend".
""")

code("""
MODEL_ORDER = list(make_models())
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=test_results, x='model', y='MAE', hue='city', order=MODEL_ORDER,
            hue_order=list(CITY_FILES), palette=CITY_COLORS, ax=ax)
for container, city in zip(ax.containers, CITY_FILES):
    sub = test_results[test_results.city == city].set_index('model')
    ax.bar_label(container, labels=[f"{sub.loc[m, 'MAE vs Naive blend']:+.1%}" for m in MODEL_ORDER],
                 fontsize=8, padding=2)
ax.set(title='Test MAE (lower is better). Labels: MAE reduction vs. Naive blend (+ = better)', xlabel='', ylabel='MAE (µg/m³)')
ax.tick_params(axis='x', labelrotation=15)
ax.set_ylim(0, test_results['MAE'].max() * 1.25)
ax.legend(title='', loc='upper center', ncol=2)
plt.show()
""")

md(note("CREATIVE"))

md("""
### 4.3. Forecast vs. actual
**Figure 7 – First 14 days of the Test set:** actual values, the baseline and the best model of each city (chosen on Validation)
(the time axis is the **forecast** time, i.e. $t + 24h$).
""")

code("""
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for ax, (city, f) in zip(axes, forecasters.items()):
    p = f.test_predictions
    w = p.loc[: p.index.min() + pd.Timedelta(days=14)]
    target_time = w.index + pd.Timedelta(hours=HORIZON)
    ax.plot(target_time, w['actual'], color='k', lw=1.3, label='Actual')
    ax.plot(target_time, w['Naive blend (baseline)'], color='gray', ls=':', label='Naive blend (baseline)')
    ax.plot(target_time, w[best_model[city]], color=CITY_COLORS[city], label=best_model[city])
    ax.set(title=f'{city} – 24h-ahead forecast', ylabel='PM2.5 (µg/m³)')
    ax.legend()
plt.tight_layout()
plt.show()
""")

md("""
### 4.4. Error analysis
A regression problem has no direct confusion matrix, so we (1) analyze the error by **hour of day** and **month**, and
(2) convert PM2.5 into **air quality levels** (US EPA 2024 breakpoints) to build a **level confusion matrix** —
showing which level the model tends to confuse with which.
""")

code("""
MIN_SAMPLES = 20   # hours/months with fewer Test samples are not shown (too noisy)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for city, f in forecasters.items():
    p = f.test_predictions
    err = (p[best_model[city]] - p['actual']).abs()
    t = p.index + pd.Timedelta(hours=HORIZON)
    hour_stats = err.groupby(t.hour).agg(['mean', 'size'])
    by_hour = hour_stats.loc[hour_stats['size'] >= MIN_SAMPLES, 'mean']
    by_month = err.groupby(t.strftime('%Y-%m')).agg(['mean', 'size']).query('size >= @MIN_SAMPLES')['mean']
    by_hour.plot(ax=axes[0], marker='o', ls='', color=CITY_COLORS[city], label=city)
    by_month.plot(ax=axes[1], marker='o', color=CITY_COLORS[city], label=city)
    sparse = hour_stats.index[hour_stats['size'] < MIN_SAMPLES].tolist()
    print(f'{city}: MAE by hour min {by_hour.min():.1f} (h{by_hour.idxmin()}) – max {by_hour.max():.1f} '
          f'(h{by_hour.idxmax()}) | hours with < {MIN_SAMPLES} Test samples (not shown): {sparse} '
          f'| by month: {by_month.round(1).to_dict()}')
axes[0].set(title='MAE by hour of day (best model)', xlabel='Hour', ylabel='MAE (µg/m³)', xticks=range(0, 24, 3))
axes[1].set(title='MAE by month in the Test set (best model)', xlabel='Month', ylabel='MAE (µg/m³)')
for ax in axes:
    ax.legend()
plt.tight_layout()
plt.show()
""")

code("""
AQI_BINS = [-np.inf, 9.0, 35.4, 55.4, 125.4, np.inf]
AQI_LABELS = ['Good', 'Moderate', 'USG', 'Unhealthy', 'Very unhealthy+']


def to_aqi_level(values):
    return pd.cut(pd.Series(np.asarray(values, dtype=float)), bins=AQI_BINS, labels=AQI_LABELS)


fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, (city, f) in zip(axes, forecasters.items()):
    p = f.test_predictions
    actual_lvl, pred_lvl = to_aqi_level(p['actual']), to_aqi_level(p[best_model[city]])
    cm = pd.crosstab(actual_lvl, pred_lvl, rownames=['Actual'], colnames=['Predicted'], dropna=False)
    cm = cm.reindex(index=AQI_LABELS, columns=AQI_LABELS, fill_value=0)
    cm_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    sns.heatmap(cm_norm, annot=cm.values, fmt='d', cmap='Blues', cbar=False, ax=ax)
    acc = (actual_lvl == pred_lvl).mean()
    ax.set_title(f'{city} – {best_model[city]} | correct level: {acc:.1%}')
    print(f'{city}: correct AQI level {acc:.1%} | within ±1 level: '
          f'{(abs(actual_lvl.cat.codes - pred_lvl.cat.codes) <= 1).mean():.1%}')
    print(cm, end='\\n\\n')
plt.tight_layout()
plt.show()
""")

md("*USG = Unhealthy for Sensitive Groups (35.5–55.4 µg/m³).*\n\n" + note("ERROR"))

md("""
### 4.5. Feature importance (permutation importance)
Measures how much the Test MAE increases when each original feature of the best model is randomly shuffled.
*Caveat:* the PM2.5 lag/rolling features are strongly correlated, so shuffling one of them hurts less than its real
contribution (the others carry the same information) — the importance of the "PM2.5 history" group is spread over several features.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, (city, f) in zip(axes, forecasters.items()):
    model = f.models[best_model[city]]
    imp = permutation_importance(model, f.test[FEATURES], f.test[TARGET], n_repeats=5,
                                 scoring='neg_mean_absolute_error', random_state=RANDOM_STATE)
    s = pd.Series(imp.importances_mean, index=FEATURES).nlargest(10).sort_values()
    s.plot.barh(ax=ax, color=CITY_COLORS[city])
    print(f'{city} – top 5:', s[::-1].head(5).round(2).to_dict())
    ax.set(title=f'{city} – Top 10 features ({best_model[city]})', xlabel='MAE increase (µg/m³)')
plt.tight_layout()
plt.show()
""")

md("""
Because single-feature importance is diluted by correlated features, we also shuffle **whole groups** of features together
(grouped permutation importance): PM2.5 history, weather (including wind sector) and calendar (hour, season, day of week).
""")

code("""
FEATURE_GROUPS = {'PM2.5 history': PM_FEATURES,
                  'Weather': WEATHER_FEATURES + ['wind_sector'],
                  'Calendar': CYCLIC_FEATURES + ['season', 'day_of_week']}


def group_importance(model, X, y, groups, n_repeats=10, seed=RANDOM_STATE):
    \"\"\"Mean MAE increase when all columns of a group are shuffled together (same row permutation).\"\"\"
    rng = np.random.default_rng(seed)
    base = mean_absolute_error(y, model.predict(X))
    result = {}
    for name, cols in groups.items():
        deltas = []
        for _ in range(n_repeats):
            Xs = X.copy()
            Xs[cols] = X[cols].values[rng.permutation(len(X))]
            deltas.append(mean_absolute_error(y, model.predict(Xs)) - base)
        result[name] = np.mean(deltas)
    return result


group_imp = pd.DataFrame({city: group_importance(f.models[best_model[city]], f.test[FEATURES], f.test[TARGET], FEATURE_GROUPS)
                          for city, f in forecasters.items()})
print('Grouped importance (MAE increase, µg/m³):', group_imp.round(2).to_dict())
display(group_imp.round(2))
""")

md(note("IMPORTANCE"))

md("""
### 4.6. Inference demo: `predict_next`
Takes the hourly observation history up to time $t$ and returns the PM2.5 forecast at $t + 24h$ together with its air quality level.
Demo: we use a history ending 24 hours before the last available observation, then compare with the known actual value.
""")

code("""
def predict_next(city, history):
    \"\"\"history: hourly DataFrame (same columns as the cleaned data), ending at time t.\"\"\"
    model = forecasters[city].models[best_model[city]]
    x_last = build_features(history, city).iloc[[-1]]
    value = float(model.predict(x_last[FEATURES])[0])
    return {'city': city, 'model': best_model[city],
            'forecast_time': x_last.index[0] + pd.Timedelta(hours=HORIZON),
            'pm25_forecast': round(value, 1), 'aqi_level': str(to_aqi_level([value])[0])}


demo_rows = []
for city, df in clean.items():
    t_end = df['pm25_raw'].last_valid_index() - pd.Timedelta(hours=HORIZON)
    history = df.loc[t_end - pd.Timedelta(days=7): t_end]
    res = predict_next(city, history)
    actual = df.loc[res['forecast_time'], 'pm25_raw']
    res.update({'pm25_actual': round(float(actual), 1), 'aqi_actual': str(to_aqi_level([actual])[0])})
    demo_rows.append(res)
display(pd.DataFrame(demo_rows))
""")

md("### 4.7. Conclusions, limitations and future work\n\n" + note("CONCLUSION"))
