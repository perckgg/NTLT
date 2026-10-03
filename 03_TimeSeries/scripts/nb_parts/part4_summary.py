# Phần 4: Tổng kết và mở rộng
md("""
## 4. TỔNG KẾT VÀ MỞ RỘNG

### 4.1. Kết quả trên tập Test độc lập
Mỗi mô hình được huấn luyện lại trên Train + Validation (siêu tham số đã cố định), rồi đánh giá **một lần** trên Test.
""")

code("""
test_results = pd.concat([f.evaluate_on_test() for f in forecasters.values()], ignore_index=True)
base_mae = test_results[test_results.model == 'Persistence (baseline)'].set_index('city')['MAE']
test_results['MAE giảm so với baseline'] = test_results.apply(lambda r: 1 - r['MAE'] / base_mae[r['city']], axis=1)
display(test_results.set_index(['city', 'model']).style.format(
    {'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'R2': '{:.3f}', 'MAE giảm so với baseline': '{:+.1%}'}))
print('Test:', {c: f.test.index.min().strftime('%Y-%m-%d') + ' → ' + f.test.index.max().strftime('%Y-%m-%d')
                for c, f in forecasters.items()})
""")

md("""
### 4.2. Câu hỏi: *Mô hình học máy tốt hơn baseline bao nhiêu ở mỗi thành phố?*
**Hình 6 – MAE trên tập Test** của từng mô hình, tách theo thành phố (nhãn trên cột: % MAE giảm so với baseline Persistence).
""")

code("""
MODEL_ORDER = list(make_models())
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=test_results, x='model', y='MAE', hue='city', order=MODEL_ORDER,
            hue_order=list(CITY_FILES), palette=CITY_COLORS, ax=ax)
for container, city in zip(ax.containers, CITY_FILES):
    sub = test_results[test_results.city == city].set_index('model')
    ax.bar_label(container, labels=[f"{sub.loc[m, 'MAE giảm so với baseline']:+.0%}" for m in MODEL_ORDER],
                 fontsize=9, padding=2)
ax.set(title='MAE trên tập Test (thấp hơn là tốt hơn)', xlabel='', ylabel='MAE (µg/m³)')
plt.show()
""")

md(note("CREATIVE"))

md("""
### 4.3. Dự báo so với thực tế
**Hình 7 – 14 ngày đầu của tập Test:** giá trị thực, baseline và mô hình tốt nhất của từng thành phố
(trục thời gian là thời điểm **được dự báo**, tức $t + 24h$).
""")

code("""
best_model = test_results.loc[test_results.groupby('city')['MAE'].idxmin()].set_index('city')['model']

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for ax, (city, f) in zip(axes, forecasters.items()):
    p = f.test_predictions
    w = p.loc[: p.index.min() + pd.Timedelta(days=14)]
    target_time = w.index + pd.Timedelta(hours=HORIZON)
    ax.plot(target_time, w['actual'], color='k', lw=1.3, label='Thực tế')
    ax.plot(target_time, w['Persistence (baseline)'], color='gray', ls=':', label='Persistence (baseline)')
    ax.plot(target_time, w[best_model[city]], color=CITY_COLORS[city], label=best_model[city])
    ax.set(title=f'{city} – dự báo trước 24h', ylabel='PM2.5 (µg/m³)')
    ax.legend()
plt.tight_layout()
plt.show()
""")

md("""
### 4.4. Phân tích lỗi (Error Analysis)
Bài toán hồi quy không có confusion matrix trực tiếp, nên ta (1) phân tích sai số theo **giờ trong ngày** và **tháng**,
(2) quy đổi PM2.5 về **mức chất lượng không khí** (ngưỡng US EPA 2024) để lập **confusion matrix theo mức** —
cho biết mô hình hay đoán nhầm mức nào sang mức nào.
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
    print(f'{city}: MAE theo giờ min {by_hour.min():.1f} (h{by_hour.idxmin()}) – max {by_hour.max():.1f} '
          f'(h{by_hour.idxmax()}) | theo tháng: {by_month.round(1).to_dict()}')
axes[0].set(title='MAE theo giờ trong ngày (mô hình tốt nhất)', xlabel='Giờ', ylabel='MAE (µg/m³)')
axes[1].set(title='MAE theo tháng của tập Test (mô hình tốt nhất)', xlabel='Tháng', ylabel='MAE (µg/m³)')
for ax in axes:
    ax.legend()
plt.tight_layout()
plt.show()
""")

code("""
AQI_BINS = [-np.inf, 9.0, 35.4, 55.4, 125.4, np.inf]
AQI_LABELS = ['Tốt', 'Trung bình', 'Kém (nhạy cảm)', 'Xấu', 'Rất xấu+']


def to_aqi_level(values):
    return pd.cut(pd.Series(np.asarray(values, dtype=float)), bins=AQI_BINS, labels=AQI_LABELS)


fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, (city, f) in zip(axes, forecasters.items()):
    p = f.test_predictions
    actual_lvl, pred_lvl = to_aqi_level(p['actual']), to_aqi_level(p[best_model[city]])
    cm = pd.crosstab(actual_lvl, pred_lvl, rownames=['Thực tế'], colnames=['Dự báo'], dropna=False)
    cm = cm.reindex(index=AQI_LABELS, columns=AQI_LABELS, fill_value=0)
    cm_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    sns.heatmap(cm_norm, annot=cm.values, fmt='d', cmap='Blues', cbar=False, ax=ax)
    acc = (actual_lvl == pred_lvl).mean()
    ax.set_title(f'{city} – {best_model[city]} | đúng mức: {acc:.1%}')
    print(f'{city}: dự báo đúng mức AQI {acc:.1%} | đúng ±1 mức: '
          f'{(abs(actual_lvl.cat.codes - pred_lvl.cat.codes) <= 1).mean():.1%}')
    print(cm, end='\\n\\n')
plt.tight_layout()
plt.show()
""")

md(note("ERROR"))

md("""
### 4.5. Mức độ quan trọng của đặc trưng (Permutation Importance)
Đo mức tăng MAE trên tập Test khi xáo trộn ngẫu nhiên từng feature gốc của mô hình tốt nhất.
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
    ax.set(title=f'{city} – Top 10 feature ({best_model[city]})', xlabel='Mức tăng MAE (µg/m³)')
plt.tight_layout()
plt.show()
""")

md(note("IMPORTANCE"))

md("""
### 4.6. Demo hàm suy luận `predict_next`
Nhận lịch sử quan trắc theo giờ đến thời điểm $t$, trả về dự báo PM2.5 lúc $t + 24h$ và mức chất lượng không khí.
Demo: dùng lịch sử kết thúc 24 giờ trước thời điểm có dữ liệu cuối cùng, rồi so với giá trị thực đã biết.
""")

code("""
def predict_next(city, history):
    \"\"\"history: DataFrame theo giờ (các cột như dữ liệu đã làm sạch), kết thúc tại thời điểm t.\"\"\"
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

md("### 4.7. Kết luận, hạn chế và hướng phát triển\n\n" + note("CONCLUSION"))
