# Phần 3: Phân tích và mô hình hóa
md("""
## 3. PHÂN TÍCH VÀ MÔ HÌNH HÓA

### 3.1. Thước đo đánh giá
Vì dữ liệu mất cân bằng (~20% rời bỏ), Accuracy dễ gây hiểu nhầm. Nhóm dùng:
- **F1 (lớp rời bỏ)** — thước đo chính, cân bằng giữa Precision và Recall.
- **Recall** — tỉ lệ khách hàng rời bỏ bắt được (bỏ sót một khách hàng sắp rời đi thường tốn kém hơn gọi nhầm một khách hàng ở lại).
- **ROC-AUC, PR-AUC** — đánh giá khả năng xếp hạng, không phụ thuộc ngưỡng.

### 3.2. Baseline và so sánh các mô hình (trên Validation)
- **Baseline 1 — Dummy:** luôn đoán lớp đa số.
- **Baseline 2 — Logistic Regression** không xử lý mất cân bằng.
- **Mô hình mở rộng:** Logistic Regression, Random Forest, HistGradientBoosting — mỗi mô hình chạy **hai phiên bản**:
  không trọng số và `class_weight='balanced'` (phạt lỗi trên lớp thiểu số nặng hơn).
""")

code("""
def evaluate(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {'Accuracy': accuracy_score(y_true, pred), 'Precision': precision_score(y_true, pred, zero_division=0),
            'Recall': recall_score(y_true, pred), 'F1': f1_score(y_true, pred),
            'ROC-AUC': roc_auc_score(y_true, prob), 'PR-AUC': average_precision_score(y_true, prob)}


def make_pipeline(clf):
    return Pipeline([('prep', build_preprocessor()), ('clf', clf)])


ALGORITHMS = {
    'LogisticRegression': lambda cw: LogisticRegression(max_iter=2000, class_weight=cw),
    'RandomForest': lambda cw: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight=cw,
                                                      n_jobs=-1, random_state=RANDOM_STATE),
    'HistGradientBoosting': lambda cw: HistGradientBoostingClassifier(class_weight=cw, random_state=RANDOM_STATE),
}

models = {'Dummy (lớp đa số)': make_pipeline(DummyClassifier(strategy='most_frequent'))}
for name, factory in ALGORITHMS.items():
    models[name] = make_pipeline(factory(None))
    models[f'{name} + balanced'] = make_pipeline(factory('balanced'))

val_rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    val_rows.append({'Mô hình': name, **evaluate(y_val, model.predict_proba(X_val)[:, 1])})
val_results = pd.DataFrame(val_rows).set_index('Mô hình')
display(val_results.style.format('{:.3f}').highlight_max(color='#c6efce', axis=0))
""")

md("""
**Câu hỏi: xử lý mất cân bằng (`class_weight='balanced'`) thay đổi Recall và F1 của từng thuật toán thế nào?**
""")

code("""
plot_rows = []
for name in ALGORITHMS:
    for suffix, label in [('', 'Không trọng số'), (' + balanced', 'class_weight=balanced')]:
        r = val_results.loc[name + suffix]
        plot_rows += [{'Thuật toán': name, 'Phiên bản': label, 'Metric': m, 'Giá trị': r[m]} for m in ['Recall', 'F1']]
plot_df = pd.DataFrame(plot_rows)

g = sns.catplot(data=plot_df, x='Thuật toán', y='Giá trị', hue='Phiên bản', col='Metric', kind='bar',
                palette=['#BBBBBB', '#DD5145'], height=4, aspect=1.4)
for ax in g.axes.flat:
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', fontsize=9)
    ax.set_ylim(0, 1)
g.figure.suptitle('Validation: ảnh hưởng của class_weight lên Recall và F1 (lớp rời bỏ)', y=1.05)
plt.show()
""")

md(note("VAL"))

md("""
### 3.3. Tinh chỉnh siêu tham số (GridSearchCV, 5-fold phân tầng trên Train)
Tinh chỉnh hai mô hình cây tốt nhất, coi `class_weight` cũng là một siêu tham số; tiêu chí chọn là **F1**.
""")

code("""
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
GRIDS = {
    'RandomForest': (ALGORITHMS['RandomForest'](None),
                     {'clf__max_depth': [8, 12, None], 'clf__min_samples_leaf': [2, 5, 10],
                      'clf__class_weight': [None, 'balanced']}),
    'HistGradientBoosting': (ALGORITHMS['HistGradientBoosting'](None),
                             {'clf__learning_rate': [0.05, 0.1], 'clf__max_leaf_nodes': [15, 31],
                              'clf__max_iter': [100, 300], 'clf__class_weight': [None, 'balanced']}),
}

tuned, tune_rows = {}, []
for name, (clf, grid) in GRIDS.items():
    search = GridSearchCV(make_pipeline(clf), grid, scoring='f1', cv=cv, n_jobs=-1).fit(X_train, y_train)
    tuned[name] = search.best_estimator_
    prob = tuned[name].predict_proba(X_val)[:, 1]
    tune_rows.append({'Mô hình': f'{name} (tuned)', 'CV F1': search.best_score_, **evaluate(y_val, prob),
                      'Tham số tốt nhất': {k.replace('clf__', ''): v for k, v in search.best_params_.items()}})
tune_results = pd.DataFrame(tune_rows).set_index('Mô hình')
display(tune_results.drop(columns='Tham số tốt nhất'))
for name, params in tune_results['Tham số tốt nhất'].items():
    print(f'{name}: {params}')
""")

md("""
### 3.4. Chọn ngưỡng quyết định trên Validation
Mặc định mô hình dự đoán "rời bỏ" khi xác suất ≥ 0.5. Với dữ liệu mất cân bằng, ngưỡng này thường không tối ưu.
Nhóm quét ngưỡng 0.10–0.90 **trên Validation** và chọn ngưỡng cho F1 cao nhất. Tập Test không tham gia bước này.
""")

code("""
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.01), 2)
best_name = tune_results['F1'].idxmax().replace(' (tuned)', '')
final_model = tuned[best_name]
val_prob = final_model.predict_proba(X_val)[:, 1]

curve = pd.DataFrame([{'threshold': t, **evaluate(y_val, val_prob, t)} for t in THRESHOLDS]).set_index('threshold')
best_threshold = float(curve['F1'].idxmax())
print(f'Mô hình cuối: {best_name} | ngưỡng tối ưu trên Validation = {best_threshold:.2f}')
display(curve.loc[[0.5, best_threshold], ['Precision', 'Recall', 'F1']].rename(index=lambda t: f'ngưỡng {t:.2f}'))

ax = curve[['Precision', 'Recall', 'F1']].plot(figsize=(10, 4), color=['#4C72B0', '#DD5145', '#55A868'])
ax.axvline(0.5, ls=':', c='gray', label='0.5 (mặc định)')
ax.axvline(best_threshold, ls='--', c='k', label=f'{best_threshold:.2f} (tối ưu F1)')
ax.set(title=f'{best_name}: Precision / Recall / F1 theo ngưỡng (Validation)', xlabel='Ngưỡng', ylabel='Giá trị')
ax.legend(); plt.show()
""")

md(note("TUNE"))
