# Section 4: summary and extensions
md("""
## 4. SUMMARY AND EXTENSIONS

### 4.1. Results on the independent Test set
Each model is refitted on the whole Development set (with its hyperparameters fixed) and evaluated **once** on Test
(01/09/2026 to the end of the data). The **best model of each city was chosen by walk-forward validation** (Section 3.2), not on Test,
so the Test set is used only for reporting and its scores are not biased by model selection.
*Note:* the Test period is the **start** of the polluted season (September–November), not its peak (December–February);
the winter is covered by the walk-forward months.
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
print('Best model (chosen by walk-forward validation, Section 3.2):', best_model.to_dict())
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
### 4.3. Are the differences real? Block-bootstrap confidence intervals
Thousands of hourly forecasts sound like a lot, but forecast errors are **strongly autocorrelated** (an error at 10h is very
likely to be followed by a similar error at 11h), so the number of *independent* observations is much smaller, and an
ordinary i.i.d. bootstrap over hours would give intervals that are far too narrow.

**Method (non-overlapping block bootstrap, Carlstein 1986; the block idea for dependent data is also in Künsch 1989).**
For a comparison of model $A$ (the best model) with model $B$ on one evaluation set (the pooled walk-forward forecasts, or Test):
1. Split the samples into $k$ consecutive **3-day calendar blocks** of forecast time $t+24h$. Blocks do not overlap; because of
   missing hours they do not all contain the same number of samples. Let $S_j^A = \\sum_{i \\in j} |y_i - \\hat y_i^A|$ be the sum of
   absolute errors of $A$ in block $j$ (same for $B$, on the **same** hours — the comparison is *paired*).
2. Draw $k$ blocks **with replacement**, i.e. counts $(c_1, \\dots, c_k) \\sim \\mathrm{Multinomial}(k; 1/k, \\dots, 1/k)$, and compute
   $g^* = 1 - \\sum_j c_j S_j^A \\, / \\, \\sum_j c_j S_j^B$ (the hour counts cancel because both models are scored on the same hours).
3. Repeat 2,000 times; the 2.5th and 97.5th percentiles of $g^*$ form the 95% **percentile** confidence interval of the gain
   $g = 1 - \\mathrm{MAE}_A / \\mathrm{MAE}_B$.

An interval that excludes 0 corresponds (approximately) to rejecting "no difference" in a two-sided test at the 5% level.
**Choice of block length:** blocks must be long enough that the error differences of different blocks are close to independent.
The cell below prints, on the walk-forward forecasts, the autocorrelation of the *daily mean* error difference between the best model and the
Naive blend at lags of 1, 2 and 3 days; the 3-day block length is checked against it, and 1-day and 7-day blocks are reported as a sensitivity check.

**Limitations of the method, stated explicitly:** (1) the percentile interval is an approximation, especially on Test, which has few blocks
(the number of blocks is the `blocks` column of the table); (2) we report 16 intervals without a multiple-comparison correction, so they are
**descriptive** — the conclusions are based on the four comparisons fixed in advance: best model vs Naive blend, on walk-forward and on Test,
for each city; (3) blocks assume the error process is roughly stationary within the evaluation period, which is only approximately true over
11 months with a winter in the middle; (4) on walk-forward, the best model was selected on these same forecasts and its hyperparameters
were tuned on the Development set, so its walk-forward gain is slightly optimistic — the Test interval is the unbiased one.
""")

code("""
BLOCK_DAYS = 3
N_BOOT = 2000

# Block-length check on the walk-forward forecasts: autocorrelation of the daily mean |error| difference (best model − Naive blend)
for city, f in forecasters.items():
    p = f.wf_predictions
    diff = (p[best_model[city]] - p['actual']).abs() - (p['Naive blend (baseline)'] - p['actual']).abs()
    daily_diff = diff.groupby((diff.index + pd.Timedelta(hours=HORIZON)).floor('D')).mean().asfreq('D')
    print(f'{city} (walk-forward): autocorrelation of the daily error difference at lag 1/2/3 days = '
          + ' / '.join(f'{daily_diff.autocorr(k):.2f}' for k in (1, 2, 3)))


def block_bootstrap_gain(p, model_a, model_b, block_days=BLOCK_DAYS, n_boot=N_BOOT, seed=RANDOM_STATE):
    \"\"\"Paired block bootstrap of the MAE gain of model_a over model_b, i.e. 1 - MAE_a / MAE_b.

    p: DataFrame indexed by issue time t with an 'actual' column and one column per model.
    Samples are grouped into consecutive calendar blocks of `block_days` days (forecast time t + HORIZON);
    whole blocks are resampled with replacement, keeping the autocorrelation of the errors inside each block.
    \"\"\"
    t = p.index + pd.Timedelta(hours=HORIZON)
    codes = pd.factorize(np.asarray((t - t.min()) // pd.Timedelta(days=block_days)))[0]
    abs_err_a = np.bincount(codes, weights=np.abs(p[model_a] - p['actual']))   # error sum per block
    abs_err_b = np.bincount(codes, weights=np.abs(p[model_b] - p['actual']))
    k = len(abs_err_a)
    counts = np.random.default_rng(seed).multinomial(k, np.full(k, 1 / k), size=n_boot)  # how often each block is drawn
    gains = 1 - (counts @ abs_err_a) / (counts @ abs_err_b)
    low, high = np.percentile(gains, [2.5, 97.5])
    return {'gain': 1 - abs_err_a.sum() / abs_err_b.sum(), 'ci_low': low, 'ci_high': high, 'blocks': k}


def competitors(city):
    return ['Persistence (baseline)', 'Naive blend (baseline)'] + [
        m for m in ['Ridge', 'RandomForest', 'HistGradientBoosting'] if m != best_model[city]]


rows = []
for city, f in forecasters.items():
    for set_name, preds in [('Walk-forward', f.wf_predictions), ('Test', f.test_predictions)]:
        for other in competitors(city):
            rows.append({'city': city, 'set': set_name, 'model': best_model[city], 'vs': other,
                         **block_bootstrap_gain(preds, best_model[city], other)})
boot = pd.DataFrame(rows)
boot['CI excludes 0'] = (boot['ci_low'] > 0) | (boot['ci_high'] < 0)
display(boot.style.format({'gain': '{:+.1%}', 'ci_low': '{:+.1%}', 'ci_high': '{:+.1%}'}))
for r in boot.itertuples():
    print(f'BOOT | {r.city} | {r.set} | {r.model} vs {r.vs} | gain {r.gain:+.1%} [{r.ci_low:+.1%}, {r.ci_high:+.1%}]')

print('\\nSensitivity to the block length (best model vs Naive blend):')
for city, f in forecasters.items():
    for set_name, preds in [('Walk-forward', f.wf_predictions), ('Test', f.test_predictions)]:
        cis = {L: block_bootstrap_gain(preds, best_model[city], 'Naive blend (baseline)', block_days=L)
               for L in (1, 3, 7)}
        print(f'{city} {set_name}: ' + ' | '.join(f"{L}-day blocks ({c['blocks']}): [{c['ci_low']:+.1%}, {c['ci_high']:+.1%}]"
                                                  for L, c in cis.items()))
""")

md("""
**Figure 7 – Gain of the best model of each city over every competitor**, with 95% block-bootstrap confidence intervals,
on the walk-forward forecasts (hollow) and Test (filled). A bar that crosses the dashed zero line means the difference could be luck.
""")

code("""
# Forest plot of the bootstrap confidence intervals
SET_STYLE = {'Walk-forward': {'color': '#eb6834', 'mfc': 'white', 'offset': -0.17},   # y axis is inverted: walk-forward on top
             'Test': {'color': '#2a78d6', 'mfc': '#2a78d6', 'offset': 0.17}}
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharex=True)
for ax, city in zip(axes, CITY_FILES):
    labels = competitors(city)
    for set_name, st in SET_STYLE.items():
        sub = boot[(boot.city == city) & (boot.set == set_name)].set_index('vs').loc[labels]
        y = np.arange(len(labels)) + st['offset']
        ax.hlines(y, sub['ci_low'] * 100, sub['ci_high'] * 100, color=st['color'], lw=2)
        ax.plot(sub['gain'] * 100, y, 'o', ms=8, color=st['color'], mfc=st['mfc'], mew=2, label=set_name)
        if set_name == 'Test':
            for yi, g in zip(y, sub['gain']):
                ax.annotate(f'{g:+.1%}', (g * 100, yi), xytext=(0, -13), textcoords='offset points',
                            ha='center', fontsize=9, color='#333333')
    ax.axvline(0, color='#555555', ls='--', lw=1)
    ax.set_yticks(range(len(labels)), [f'vs {l}' for l in labels])
    ax.invert_yaxis()
    ax.set(title=f'{city} – {best_model[city]}', xlabel='MAE reduction (%), 95% block-bootstrap CI')
    ax.grid(axis='y', visible=False)
axes[0].legend(loc='lower right')
plt.tight_layout()
plt.show()
""")

md(note("BOOTSTRAP"))

md("""
### 4.4. Forecast vs. actual
**Figure 8 – First 14 days of the Test set:** actual values, the Naive blend baseline and the best model of each city (chosen by walk-forward validation)
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
    mae_blend = mean_absolute_error(w['actual'], w['Naive blend (baseline)'])
    mae_best = mean_absolute_error(w['actual'], w[best_model[city]])
    print(f'{city} – first 14 days of Test ({len(w)} hours): MAE Naive blend {mae_blend:.2f} vs {best_model[city]} {mae_best:.2f} | '
          f'std of forecast {w[best_model[city]].std():.1f} vs actual {w["actual"].std():.1f} µg/m³')
    ax.set(title=f'{city} – 24h-ahead forecast (MAE in this window: Naive blend {mae_blend:.2f}, {best_model[city]} {mae_best:.2f})',
           ylabel='PM2.5 (µg/m³)')
    ax.legend()
plt.tight_layout()
plt.show()
""")

md(note("FORECAST"))

md("""
### 4.5. Error analysis
A regression problem has no direct confusion matrix, so we (1) analyze the error by **hour of day** (the error by **month** is the
walk-forward figure of Section 3.2), and
(2) put both the actual and the forecast PM2.5 into **concentration bands** defined by the US EPA 2024 PM2.5 breakpoints
(Good ≤ 9.0, Moderate 9.1–35.4, USG 35.5–55.4, Unhealthy 55.5–125.4, Very unhealthy and above ≥ 125.5 µg/m³) and build a
**band confusion matrix** — showing which band the model tends to confuse with which.

*Precision about the definition:* the official EPA AQI is computed from a **24-hour average** (or the NowCast for hourly reporting).
Here the breakpoints are applied to **hourly** concentrations, so these are "EPA concentration bands", **not** official AQI values.
As in the EPA rule, a concentration is first truncated to one decimal (e.g. 9.05 → 9.0 → Good) and then binned.
""")

code("""
MIN_SAMPLES = 20   # hours of the day with fewer samples are not shown (too noisy)
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for ax, (city, f) in zip(axes, forecasters.items()):
    for set_name, p, style in [('Walk-forward', f.wf_predictions, dict(ls='--', marker='o', mfc='white')),
                               ('Test', f.test_predictions, dict(ls='-', marker='o'))]:
        err = (p[best_model[city]] - p['actual']).abs()
        t = p.index + pd.Timedelta(hours=HORIZON)                  # forecast time
        hour_stats = err.groupby(t.hour).agg(['mean', 'size'])
        by_hour = hour_stats.loc[hour_stats['size'] >= MIN_SAMPLES, 'mean']
        ax.plot(by_hour.index, by_hour, color=CITY_COLORS[city], label=set_name, **style)
        print(f'{city} {set_name}: MAE by hour of the forecast time min {by_hour.min():.1f} (h{by_hour.idxmin()}) – '
              f'max {by_hour.max():.1f} (h{by_hour.idxmax()}) | hours with < {MIN_SAMPLES} samples (not shown): '
              f'{hour_stats.index[hour_stats["size"] < MIN_SAMPLES].tolist()}')
    ax.set(title=f'{city} – MAE by hour of day ({best_model[city]})', xlabel='Hour of the forecast time',
           ylabel='MAE (µg/m³)', xticks=range(0, 24, 3))
    ax.legend()
plt.tight_layout()
plt.show()
""")

code("""
EPA_BINS = [-np.inf, 9.0, 35.4, 55.4, 125.4, np.inf]      # right-closed: (-inf, 9.0], (9.0, 35.4], ...
EPA_LABELS = ['Good', 'Moderate', 'USG', 'Unhealthy', 'Very unhealthy+']


def to_epa_band(values):
    \"\"\"Truncate to 0.1 µg/m³ (EPA rule), then map each hourly PM2.5 value to its EPA concentration band.\"\"\"
    truncated = np.floor(np.asarray(values, dtype=float) * 10) / 10
    return pd.cut(pd.Series(truncated), bins=EPA_BINS, labels=EPA_LABELS)


fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, (city, f) in zip(axes, forecasters.items()):
    wf = f.wf_predictions
    wf_acc = (to_epa_band(wf['actual']) == to_epa_band(wf[best_model[city]])).mean()
    print(f'{city}: correct EPA band on the walk-forward forecasts {wf_acc:.1%} (for comparison with Test below)')
    p = f.test_predictions
    actual_lvl, pred_lvl = to_epa_band(p['actual']), to_epa_band(p[best_model[city]])
    cm = pd.crosstab(actual_lvl, pred_lvl, rownames=['Actual'], colnames=['Predicted'], dropna=False)
    cm = cm.reindex(index=EPA_LABELS, columns=EPA_LABELS, fill_value=0)
    cm_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    sns.heatmap(cm_norm, annot=cm.values, fmt='d', cmap='Blues', cbar=False, ax=ax)
    acc = (actual_lvl == pred_lvl).mean()
    ax.set_title(f'{city} – {best_model[city]} (Test) | correct EPA band: {acc:.1%}')
    print(f'{city} Test: correct EPA band {acc:.1%} | within ±1 band: '
          f'{(abs(actual_lvl.cat.codes - pred_lvl.cat.codes) <= 1).mean():.1%}')
    print(cm, end='\\n\\n')
plt.tight_layout()
plt.show()
""")

md("*USG = Unhealthy for Sensitive Groups (35.5–55.4 µg/m³).*\n\n" + note("ERROR"))

md("""
### 4.6. Feature importance (permutation importance)
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
### 4.7. Inference demo: `predict_next`
Takes the hourly observation history up to time $t$ and returns the PM2.5 forecast at $t + 24h$ together with its EPA concentration band.
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
            'pm25_forecast': round(value, 1), 'epa_band': str(to_epa_band([value])[0])}


demo_rows = []
for city, df in clean.items():
    t_end = df['pm25_raw'].last_valid_index() - pd.Timedelta(hours=HORIZON)
    history = df.loc[t_end - pd.Timedelta(days=7): t_end]
    res = predict_next(city, history)
    actual = df.loc[res['forecast_time'], 'pm25_raw']
    res.update({'pm25_actual': round(float(actual), 1), 'epa_band_actual': str(to_epa_band([actual])[0])})
    demo_rows.append(res)
display(pd.DataFrame(demo_rows))
""")

md("### 4.8. Conclusions, limitations and future work\n\n" + note("CONCLUSION"))
