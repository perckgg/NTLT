# Phần 4: Tổng kết và mở rộng
md("""
## 4. TỔNG KẾT VÀ MỞ RỘNG

### 4.1. Đánh giá trên tập Test độc lập
Mô hình, siêu tham số và ngưỡng đều đã được chốt ở Phần 3; tập Test chỉ được dùng **một lần** ở đây.
""")

code("""
test_prob = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)
print(f'=== {best_name} (tuned), ngưỡng {best_threshold:.2f} — TẬP TEST ===')
print(classification_report(y_test, test_pred, target_names=list(CLASS_NAMES.values()), digits=3))

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, test_pred), display_labels=list(CLASS_NAMES.values())).plot(
    ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title(f'Confusion matrix (ngưỡng {best_threshold:.2f})')
for name, model in {'LogisticRegression (baseline)': models['LogisticRegression'], **tuned}.items():
    p = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p)
    prec, rec, _ = precision_recall_curve(y_test, p)
    axes[1].plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, p):.3f})')
    axes[2].plot(rec, prec, label=f'{name} (AP={average_precision_score(y_test, p):.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set(title='ROC curve (Test)', xlabel='False Positive Rate', ylabel='True Positive Rate')
axes[2].axhline(y_test.mean(), ls='--', c='gray', label=f'Ngẫu nhiên ({y_test.mean():.2f})')
axes[2].set(title='Precision–Recall curve (Test)', xlabel='Recall', ylabel='Precision')
for ax in axes[1:]:
    ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md("""
### 4.2. Câu hỏi: *Từng bước cải tiến giúp bắt thêm được bao nhiêu khách hàng rời bỏ trên tập Test?*
So sánh 4 cấu hình trên **cùng tập Test**:
1. **Baseline** — Logistic Regression, không xử lý mất cân bằng, ngưỡng 0.5.
2. **Mô hình tốt nhất, không trọng số** — cùng siêu tham số với mô hình cuối nhưng `class_weight=None`, ngưỡng 0.5.
3. **+ class_weight='balanced'**, ngưỡng 0.5.
4. **+ ngưỡng tối ưu** chọn trên Validation (mô hình cuối).
""")

code("""
unweighted = clone(final_model).set_params(clf__class_weight=None).fit(X_train, y_train)
balanced = clone(final_model).set_params(clf__class_weight='balanced').fit(X_train, y_train)
balanced_val_prob = balanced.predict_proba(X_val)[:, 1]
balanced_threshold = max(THRESHOLDS, key=lambda t: f1_score(y_val, (balanced_val_prob >= t).astype(int)))

steps = [('1. Baseline LogReg', models['LogisticRegression'], 0.5),
         (f'2. {best_name}', unweighted, 0.5),
         ('3. + class_weight', balanced, 0.5),
         (f'4. + class_weight + ngưỡng {balanced_threshold:.2f}', balanced, balanced_threshold)]
step_rows = []
for label, model, thr in steps:
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= thr).astype(int)
    step_rows.append({'Cấu hình': label, **evaluate(y_test, prob, thr),
                      'Bắt được (TP)': int(((pred == 1) & (y_test == 1)).sum()),
                      'Báo nhầm (FP)': int(((pred == 1) & (y_test == 0)).sum())})
steps_df = pd.DataFrame(step_rows).set_index('Cấu hình')
display(steps_df.style.format({c: '{:.3f}' for c in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']}))

fig, ax = plt.subplots(figsize=(13, 5))
long = steps_df[['Precision', 'Recall', 'F1']].reset_index().melt('Cấu hình', var_name='Metric', value_name='Giá trị')
sns.barplot(data=long, x='Cấu hình', y='Giá trị', hue='Metric', palette=['#4C72B0', '#DD5145', '#55A868'], ax=ax)
for c in ax.containers:
    ax.bar_label(c, fmt='%.2f', fontsize=9)
n_pos = int(y_test.sum())
ax.set_xticks(range(len(steps_df)))
ax.set_xticklabels([f'{i}\\nbắt {int(r["Bắt được (TP)"])}/{n_pos} | nhầm {int(r["Báo nhầm (FP)"])}'
                    for i, r in steps_df.iterrows()], fontsize=9)
ax.set(title='Test: Precision / Recall / F1 của lớp rời bỏ qua từng bước cải tiến', xlabel='', ylim=(0, 1))
plt.tight_layout(); plt.show()
""")

md(note("CREATIVE"))

md("""
### 4.3. Mức độ quan trọng của đặc trưng (Permutation Importance trên Test)
Xáo trộn ngẫu nhiên từng **cột gốc** và đo mức giảm PR-AUC. Cách này áp dụng cho mọi mô hình
và cho kết quả theo tên cột ban đầu (dễ giải thích hơn feature importance của cây sau one-hot).
""")

code("""
imp = permutation_importance(final_model, X_test, y_test, scoring='average_precision', n_repeats=10,
                             random_state=RANDOM_STATE, n_jobs=-1)
imp_df = pd.DataFrame({'Mức giảm PR-AUC': imp.importances_mean, 'std': imp.importances_std},
                      index=X_test.columns).sort_values('Mức giảm PR-AUC', ascending=False)
display(imp_df)

plt.figure(figsize=(9, 5))
plt.barh(imp_df.index[::-1], imp_df['Mức giảm PR-AUC'][::-1], xerr=imp_df['std'][::-1], color='#8172B3')
plt.title(f'Permutation importance — {best_name} (Test)'); plt.xlabel('Mức giảm PR-AUC khi xáo trộn cột')
plt.tight_layout(); plt.show()
""")

md(note("IMPORTANCE"))

md("""
### 4.4. Phân tích lỗi (Error Analysis)
Mô hình bỏ sót (**False Negative**) và báo nhầm (**False Positive**) ở những nhóm khách hàng nào?
""")

code("""
err = X_test.copy()
err['actual'], err['prob'], err['pred'] = y_test, test_prob, test_pred
err['AgeGroup'] = pd.cut(err['Age'], [0, 30, 40, 50, 60, 100], labels=['<30', '30-39', '40-49', '50-59', '60+'])

def segment_table(col):
    g = err.groupby(col, observed=True)
    return pd.DataFrame({'Số KH': g.size(), '% rời bỏ thực tế': g['actual'].mean(),
                         'Recall': g.apply(lambda d: recall_score(d.actual, d.pred, zero_division=0)),
                         'Precision': g.apply(lambda d: precision_score(d.actual, d.pred, zero_division=0))})

fig, axes = plt.subplots(1, 4, figsize=(22, 4))
for ax, col in zip(axes, ['Geography', 'NumOfProducts', 'IsActiveMember', 'AgeGroup']):
    t = segment_table(col)
    t[['Recall', 'Precision']].plot.bar(ax=ax, rot=0, color=['#DD5145', '#4C72B0'], ylim=(0, 1.1))
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', fontsize=8)
    ax.set(title=f'Theo {col}', xlabel='')
plt.suptitle('Recall / Precision của lớp rời bỏ theo nhóm khách hàng (Test)', y=1.03)
plt.tight_layout(); plt.show()
display(segment_table('AgeGroup'))

fn, fp = err[(err.actual == 1) & (err.pred == 0)], err[(err.actual == 0) & (err.pred == 1)]
tp = err[(err.actual == 1) & (err.pred == 1)]
print(f'False Negative: {len(fn)} | False Positive: {len(fp)}')
cols = ['Age', 'Geography', 'Balance', 'NumOfProducts', 'IsActiveMember']
num_cmp = ['Age', 'Balance', 'CreditScore', 'NumOfProducts', 'IsActiveMember']
display(pd.DataFrame({'Bỏ sót (FN)': fn[num_cmp].mean(), 'Bắt đúng (TP)': tp[num_cmp].mean(),
                      'Báo nhầm (FP)': fp[num_cmp].mean()}).rename_axis('Trung bình'))
print('Ví dụ khách hàng bị BỎ SÓT với xác suất thấp nhất:')
display(fn.sort_values('prob')[cols + ['prob']].head(5))
""")

md(note("ERROR"))

md("""
### 4.5. Demo hàm suy luận
Hàm nhận thông tin **thô** của một khách hàng (có thể thiếu giá trị), tự chạy toàn bộ pipeline (tạo đặc trưng → điền thiếu →
scale → encode → mô hình) và áp dụng ngưỡng đã chọn.
""")

code("""
def predict_customer_churn(customer, model=final_model, threshold=best_threshold):
    x = pd.DataFrame([customer]).reindex(columns=X_train.columns)
    prob = float(model.predict_proba(x)[0, 1])
    return {'Xác suất rời bỏ': f'{prob:.1%}', 'Ngưỡng': threshold,
            'Kết luận': 'NGUY CƠ RỜI BỎ — nên liên hệ giữ chân' if prob >= threshold else 'Ít nguy cơ rời bỏ'}


demo_customers = {
    'KH A — 52 tuổi, Đức, 1 sản phẩm, không hoạt động': dict(CreditScore=580, Geography='Germany', Gender='Female', Age=52,
        Tenure=2, Balance=125000.0, NumOfProducts=1, HasCrCard=1, IsActiveMember=0, EstimatedSalary=85000.0),
    'KH B — 30 tuổi, Pháp, 2 sản phẩm, hoạt động': dict(CreditScore=720, Geography='France', Gender='Male', Age=30,
        Tenure=6, Balance=0.0, NumOfProducts=2, HasCrCard=1, IsActiveMember=1, EstimatedSalary=60000.0),
    'KH C — thiếu Age và Balance': dict(CreditScore=650, Geography='Spain', Gender='Female', Age=None,
        Tenure=4, Balance=None, NumOfProducts=3, HasCrCard=0, IsActiveMember=0, EstimatedSalary=None),
}
display(pd.DataFrame({k: predict_customer_churn(v) for k, v in demo_customers.items()}).T)
""")

md("### 4.6. Kết luận, hạn chế và hướng phát triển\n\n" + note("CONCLUSION"))
