# Section 4: summary and extensions
md("""
## 4. SUMMARY AND EXTENSIONS

### 4.1. Results on the independent Test set
Each model is retrained on Train + Validation (with its hyperparameters fixed) and evaluated **once** on Test.
""")

code("""
test_results = pd.concat([f.evaluate_on_test() for f in forecasters.values()], ignore_index=True)
base_mae = test_results[test_results.model == 'Persistence (baseline)'].set_index('city')['MAE']
test_results['MAE reduction vs baseline'] = test_results.apply(lambda r: 1 - r['MAE'] / base_mae[r['city']], axis=1)
display(test_results.set_index(['city', 'model']).style.format(
    {'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'R2': '{:.3f}', 'MAE reduction vs baseline': '{:+.1%}'}))
print('Test:', {c: f.test.index.min().strftime('%Y-%m-%d') + ' → ' + f.test.index.max().strftime('%Y-%m-%d')
                for c, f in forecasters.items()})
""")

md("""
### 4.2. Question: *How much better than the baseline is machine learning in each city?*
**Figure 6 – Test MAE** of each model, per city (bar labels: % MAE reduction vs. the Persistence baseline).
""")

code("""
MODEL_ORDER = list(make_models())
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=test_results, x='model', y='MAE', hue='city', order=MODEL_ORDER,
            hue_order=list(CITY_FILES), palette=CITY_COLORS, ax=ax)
for container, city in zip(ax.containers, CITY_FILES):
    sub = test_results[test_results.city == city].set_index('model')
    ax.bar_label(container, labels=[f"{sub.loc[m, 'MAE reduction vs baseline']:+.0%}" for m in MODEL_ORDER],
                 fontsize=9, padding=2)
ax.set(title='Test MAE (lower is better)', xlabel='', ylabel='MAE (µg/m³)')
ax.set_ylim(0, test_results['MAE'].max() * 1.25)
ax.legend(title='', loc='upper center', ncol=2)
plt.show()
""")

md(note("CREATIVE"))

md("""
### 4.3. Forecast vs. actual
**Figure 7 – First 14 days of the Test set:** actual values, the baseline and the best model for each city
(the time axis is the **forecast** time, i.e. $t + 24h$).
""")

code("""
best_model = test_results.loc[test_results.groupby('city')['MAE'].idxmin()].set_index('city')['model']

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for ax, (city, f) in zip(axes, forecasters.items()):
    p = f.test_predictions
    w = p.loc[: p.index.min() + pd.Timedelta(days=14)]
    target_time = w.index + pd.Timedelta(hours=HORIZON)
    ax.plot(target_time, w['actual'], color='k', lw=1.3, label='Actual')
    ax.plot(target_time, w['Persistence (baseline)'], color='gray', ls=':', label='Persistence (baseline)')
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
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for city, f in forecasters.items():
    p = f.test_predictions
    err = (p[best_model[city]] - p['actual']).abs()
    t = p.index + pd.Timedelta(hours=HORIZON)
    err.groupby(t.hour).mean().plot(ax=axes[0], marker='o', color=CITY_COLORS[city], label=city)
    by_hour, by_month = err.groupby(t.hour).mean(), err.groupby(t.strftime('%Y-%m')).mean()
    by_month.plot(ax=axes[1], marker='o', color=CITY_COLORS[city], label=city)
    print(f'{city}: MAE by hour min {by_hour.min():.1f} (h{by_hour.idxmin()}) – max {by_hour.max():.1f} '
          f'(h{by_hour.idxmax()}) | by month: {by_month.round(1).to_dict()}')
axes[0].set(title='MAE by hour of day (best model)', xlabel='Hour', ylabel='MAE (µg/m³)')
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
    t_end = df['pm25'].last_valid_index() - pd.Timedelta(hours=HORIZON)
    history = df.loc[t_end - pd.Timedelta(days=7): t_end]
    res = predict_next(city, history)
    actual = df.loc[res['forecast_time'], 'pm25']
    res.update({'pm25_actual': round(float(actual), 1), 'aqi_actual': str(to_aqi_level([actual])[0])})
    demo_rows.append(res)
display(pd.DataFrame(demo_rows))
""")

md("### 4.7. Conclusions, limitations and future work\n\n" + note("CONCLUSION"))
