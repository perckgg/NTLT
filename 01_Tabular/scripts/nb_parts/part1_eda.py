# Phần 0 + 1: Giới thiệu, môi trường, mô tả dữ liệu & EDA (theo khung báo cáo mẫu của GV)
md("""
# BÀI TẬP LỚN — NỀN TẢNG LẬP TRÌNH CHO PHÂN TÍCH VÀ TRỰC QUAN DỮ LIỆU
**Học kỳ 261 – Năm học 2026–2027 · Giảng viên: Lê Thành Sách**

## BÀI 1 — DỮ LIỆU BẢNG (TABULAR): DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ NGÂN HÀNG
- **Nhóm:** `<Tên nhóm>` — **Thành viên:** `<Họ tên – MSSV – vai trò>`

**Bài toán:** phân loại nhị phân — dự đoán khách hàng có rời bỏ ngân hàng (`Exited = 1`) hay không,
để bộ phận chăm sóc khách hàng ưu tiên giữ chân đúng người.

**Câu hỏi chính của bài:**
1. Những đặc điểm nào phân biệt khách hàng rời bỏ với khách hàng ở lại?
2. Dữ liệu mất cân bằng (~20% rời bỏ) ảnh hưởng thế nào và **xử lý mất cân bằng giúp bắt thêm được bao nhiêu khách hàng rời bỏ**?

**Quy trình:** EDA → làm sạch & tiền xử lý trong Pipeline (chống rò rỉ dữ liệu) → so sánh baseline với 3 mô hình →
tinh chỉnh siêu tham số và ngưỡng quyết định → đánh giá một lần trên tập Test → phân tích lỗi và suy luận.
""")

md("## 0. CÀI ĐẶT MÔI TRƯỜNG")

code("""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', lambda v: f'{v:,.3f}')
sns.set_theme(style='whitegrid')
RANDOM_STATE = 42
CLASS_COLORS = {0: '#4C72B0', 1: '#DD5145'}
CLASS_NAMES = {0: 'Ở lại (0)', 1: 'Rời bỏ (1)'}
print('Môi trường đã sẵn sàng!')
""")

md("""
## 1. MÔ TẢ TẬP DỮ LIỆU VÀ KHÁM PHÁ (EDA)

**Nguồn:** *Bank Customer Churn* (Churn_Modelling) — 10,000 khách hàng của một ngân hàng châu Âu (Pháp, Đức, Tây Ban Nha), 14 cột.

> ⚠️ **Lưu ý minh bạch:** dữ liệu gốc **không có missing values**. Để thực hành thống kê và xử lý thiếu dữ liệu theo yêu cầu đề bài,
> nhóm tạo missing **ngẫu nhiên hoàn toàn (MCAR)** ở 5 cột với tỉ lệ 2–4% bằng hàm `inject_missing` (seed = 42, script
> `scripts/prepare_churn_dataset.py`). Dữ liệu đọc từ file local hay tải từ URL gốc đều cho kết quả giống hệt nhau.

### 1.1. Tải dữ liệu
""")

code("""
GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/perckgg/NTLT/main/01_Tabular/data/tabular'
ORIGINAL_URL = 'https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv'
DATA_FILE = 'customer_churn_dataset.csv'
LOCAL_DIRS = ['data/tabular', '../data/tabular', '/content/data/tabular', '/content']
MISSING_RATES = {'CreditScore': 0.025, 'Geography': 0.020, 'Age': 0.035, 'Balance': 0.030, 'EstimatedSalary': 0.040}


def inject_missing(df, rates=MISSING_RATES, seed=RANDOM_STATE):
    \"\"\"Tạo missing MCAR — giống hệt scripts/prepare_churn_dataset.py.\"\"\"
    df = df.copy()
    rng = np.random.default_rng(seed)
    for col, rate in rates.items():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float)
        df.loc[rng.random(len(df)) < rate, col] = np.nan
    return df


def load_data():
    for d in LOCAL_DIRS:
        if (Path(d) / DATA_FILE).exists():
            return pd.read_csv(Path(d) / DATA_FILE), f'local: {Path(d) / DATA_FILE}'
    try:
        return pd.read_csv(f'{GITHUB_RAW_BASE}/{DATA_FILE}'), 'GitHub của nhóm'
    except Exception:
        return inject_missing(pd.read_csv(ORIGINAL_URL)), 'URL gốc + inject_missing(seed=42)'


df, source = load_data()
print(f'Nguồn: {source} | {df.shape[0]:,} dòng × {df.shape[1]} cột')
df.head()
""")

md("""
### 1.2. Tổng quan và mô tả thuộc tính

| Cột | Kiểu | Mô tả |
|---|---|---|
| RowNumber, CustomerId, Surname | Định danh | Chỉ để quản lý — **loại bỏ** khi mô hình hóa |
| CreditScore | Numerical | Điểm tín dụng (350–850) |
| Geography | Categorical | Quốc gia: France / Germany / Spain |
| Gender | Categorical | Giới tính |
| Age | Numerical | Tuổi |
| Tenure | Numerical | Số năm là khách hàng (0–10) |
| Balance | Numerical | Số dư tài khoản |
| NumOfProducts | Numerical (rời rạc) | Số sản phẩm đang dùng (1–4) |
| HasCrCard | Binary | Có thẻ tín dụng |
| IsActiveMember | Binary | Có hoạt động thường xuyên |
| EstimatedSalary | Numerical | Lương ước tính |
| **Exited** | **Target** | 1 = rời bỏ, 0 = ở lại |
""")

code("""
ID_COLS = ['RowNumber', 'CustomerId', 'Surname']
TARGET = 'Exited'
NUM_COLS = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
BIN_COLS = ['HasCrCard', 'IsActiveMember']
CAT_COLS = ['Geography', 'Gender']

overview = pd.Series({'Số mẫu': len(df), 'Số feature dùng được': len(NUM_COLS + BIN_COLS + CAT_COLS),
                      'Numerical': len(NUM_COLS), 'Binary': len(BIN_COLS), 'Categorical': len(CAT_COLS),
                      'Dòng trùng lặp': df.duplicated().sum(), 'CustomerId trùng': df['CustomerId'].duplicated().sum()})
display(overview.to_frame('Giá trị'))
df.info()

for label in [0, 1]:
    print(f'\\nMẫu dữ liệu — {CLASS_NAMES[label]}:')
    display(df[df[TARGET] == label].drop(columns=ID_COLS).head(3))
""")

md("""
### 1.3. Phân tích Missing Values
- **What:** cột nào thiếu, thiếu bao nhiêu. **Why:** missing làm lệch thống kê và nhiều mô hình không chấp nhận NaN.
- **How (quy tắc trong báo cáo mẫu):** < 5% → điền đơn giản (median/mode); 5–30% → phương pháp nâng cao; > 50% → cân nhắc bỏ cột.
""")

code("""
miss = df.isna().sum()
miss = pd.DataFrame({'Missing': miss, 'Tỉ lệ (%)': miss / len(df) * 100})
miss = miss[miss['Missing'] > 0].sort_values('Missing', ascending=False)
miss['Mức độ'] = pd.cut(miss['Tỉ lệ (%)'], [0, 5, 30, 100], labels=['Thấp', 'Trung bình', 'Cao'])
print(f'Tổng ô thiếu: {miss.Missing.sum():,} | số cột thiếu: {len(miss)} | dòng có ít nhất 1 ô thiếu: {df.isna().any(axis=1).mean():.1%}')
display(miss)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.barplot(x=miss['Tỉ lệ (%)'], y=miss.index, ax=axes[0], color='#DD5145')
axes[0].axvline(5, ls='--', c='gray'); axes[0].set_title('Tỉ lệ missing theo cột (ngưỡng 5%)')
rate_by_class = df[miss.index].isna().groupby(df[TARGET]).mean().T * 100
rate_by_class.columns = [CLASS_NAMES[c] for c in rate_by_class.columns]
rate_by_class.plot.bar(ax=axes[1], color=list(CLASS_COLORS.values()), rot=0)
axes[1].set(title='Tỉ lệ missing theo lớp — kiểm tra missing có phụ thuộc nhãn không', ylabel='%')
plt.tight_layout(); plt.show()
""")

md("""
### 1.4. Phân phối các biến Numerical
- **Why:** hình dạng phân phối quyết định cách scaling/biến đổi; boxplot cho thấy outlier.
- **How:** so sánh mean với median, xem độ lệch (skewness); |skew| > 1 là lệch mạnh.
""")

code("""
stats = df[NUM_COLS].describe().T[['mean', '50%', 'std', 'min', 'max']].rename(columns={'50%': 'median'})
stats['skew'] = df[NUM_COLS].skew()
stats['% bằng 0'] = (df[NUM_COLS] == 0).mean() * 100
display(stats)

fig, axes = plt.subplots(2, len(NUM_COLS), figsize=(22, 7), gridspec_kw={'height_ratios': [3, 1]})
for i, col in enumerate(NUM_COLS):
    sns.histplot(df[col].dropna(), bins=30, kde=col not in ['NumOfProducts', 'Tenure'], ax=axes[0, i], color='#4C72B0')
    axes[0, i].set_title(col)
    sns.boxplot(x=df[col], ax=axes[1, i], color='#8DA0CB')
plt.suptitle('Phân phối (trên) và boxplot (dưới) của các biến số', y=1.02)
plt.tight_layout(); plt.show()
""")

md("""
### 1.5. Phân phối các biến Categorical / Binary
- **How (báo cáo mẫu):** cardinality < 10 → One-Hot Encoding; > 50 → Target/Frequency Encoding; nhóm hiếm (< 1%) nên gộp.
""")

code("""
cat_info = pd.DataFrame({c: {'Số giá trị': df[c].nunique(), 'Mode': df[c].mode()[0],
                             'Tỉ lệ mode': df[c].value_counts(normalize=True).iloc[0],
                             'Nhóm nhỏ nhất': df[c].value_counts(normalize=True).iloc[-1]}
                         for c in CAT_COLS + BIN_COLS}).T
display(cat_info)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, col in zip(axes, CAT_COLS + BIN_COLS):
    sns.countplot(data=df, x=col, ax=ax, color='#55A868', order=df[col].value_counts().index)
    ax.bar_label(ax.containers[0], fmt='%d')
    ax.set_title(col)
plt.tight_layout(); plt.show()
""")

md("""
### 1.6. Phân phối biến mục tiêu
- **How:** cân bằng ~ 40:60–50:50; mất cân bằng > 2:1; rất mất cân bằng > 10:1. Mất cân bằng → dùng stratified split,
  class weight và các metric F1 / Recall / ROC-AUC / PR-AUC thay cho Accuracy.
""")

code("""
counts = df[TARGET].value_counts().sort_index()
print(f'Ở lại: {counts[0]:,} ({counts[0] / len(df):.1%}) | Rời bỏ: {counts[1]:,} ({counts[1] / len(df):.1%}) '
      f'| tỉ lệ {counts[0] / counts[1]:.1f}:1')
print(f'→ Một mô hình luôn đoán "Ở lại" đã đạt Accuracy {counts[0] / len(df):.1%} nhưng không bắt được khách hàng rời bỏ nào.')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].pie(counts, labels=[CLASS_NAMES[i] for i in counts.index], autopct='%1.1f%%',
            colors=list(CLASS_COLORS.values()), startangle=90)
axes[0].set_title('Tỉ lệ các lớp')
bars = axes[1].bar([CLASS_NAMES[i] for i in counts.index], counts, color=list(CLASS_COLORS.values()))
axes[1].bar_label(bars); axes[1].set_title('Số lượng mỗi lớp')
plt.tight_layout(); plt.show()
""")

md("""
### 1.7. Phân tích tương quan
- **How:** |r| > 0.7 → đa cộng tuyến, cân nhắc bỏ bớt một biến; |r| < 0.3 → tương quan tuyến tính yếu.
""")

code("""
corr = df[NUM_COLS + BIN_COLS + [TARGET]].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, mask=np.triu(np.ones_like(corr, dtype=bool), 1))
plt.title('Ma trận tương quan Pearson'); plt.show()

pairs = corr.drop(index=TARGET, columns=TARGET).where(np.triu(np.ones((len(corr) - 1,) * 2, dtype=bool), 1)).stack()
print(f'Cặp feature tương quan mạnh nhất: {pairs.abs().idxmax()} r = {pairs[pairs.abs().idxmax()]:.2f}')
print('Tương quan với Exited:', corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(2).to_dict())
""")

md("""
### 1.8. Phát hiện Outlier (phương pháp IQR)
Giá trị ngoài khoảng $[Q_1 - 1.5\\,IQR,\\; Q_3 + 1.5\\,IQR]$ được coi là outlier. Mức độ: < 1% thấp, 1–5% trung bình, > 5% cao.
""")

code("""
def iqr_table(data, cols, k=1.5):
    rows = []
    for c in cols:
        s = data[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        n = ((s < lo) | (s > hi)).sum()
        rows.append({'Feature': c, 'Outliers': n, 'Tỉ lệ (%)': n / len(s) * 100,
                     'Khoảng hợp lệ': f'[{lo:,.1f}, {hi:,.1f}]'})
    t = pd.DataFrame(rows).set_index('Feature')
    t['Mức độ'] = pd.cut(t['Tỉ lệ (%)'], [-0.01, 1, 5, 100], labels=['Thấp', 'Trung bình', 'Cao'])
    return t


outliers = iqr_table(df, NUM_COLS)
display(outliers)
age_out = df['Age'] > df['Age'].quantile(0.75) + 1.5 * (df['Age'].quantile(0.75) - df['Age'].quantile(0.25))
print(f'Tỉ lệ rời bỏ ở nhóm outlier Age (> 62 tuổi, n={age_out.sum()}): {df.loc[age_out, TARGET].mean():.1%} '
      f'(toàn bộ: {df[TARGET].mean():.1%})')
display(df.groupby(pd.cut(df['Age'], [17, 30, 40, 50, 62, 70, 100]), observed=True)[TARGET]
          .agg(['size', 'mean']).rename(columns={'size': 'Số KH', 'mean': '% rời bỏ'}).rename_axis('Nhóm tuổi'))
""")

md("""
### 1.9. Quan hệ giữa biến mục tiêu và các feature
- **How:** với categorical so sánh **tỉ lệ rời bỏ** theo từng nhóm; với numerical so sánh phân phối theo lớp.
  Chênh lệch lớn → feature có khả năng dự báo tốt.
""")

code("""
overall = df[TARGET].mean()
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, col in zip(axes, ['Geography', 'Gender', 'NumOfProducts', 'IsActiveMember']):
    rate = df.groupby(col)[TARGET].agg(['mean', 'size'])
    sns.barplot(x=rate.index.astype(str), y=rate['mean'] * 100, ax=ax, color='#DD5145')
    ax.bar_label(ax.containers[0], labels=[f'{m:.0%}\\n(n={n:,})' for m, n in rate.values], fontsize=9)
    ax.axhline(overall * 100, ls='--', c='gray', label=f'Trung bình {overall:.0%}')
    ax.set(title=f'Tỉ lệ rời bỏ theo {col}', ylabel='% rời bỏ', xlabel='', ylim=(0, rate['mean'].max() * 125))
    ax.legend(loc='upper left', fontsize=8)
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, col in zip(axes, ['Age', 'Balance', 'CreditScore']):
    sns.kdeplot(data=df.assign(Lớp=df[TARGET].map(CLASS_NAMES)), x=col, hue='Lớp', common_norm=False, fill=True,
                palette={CLASS_NAMES[k]: v for k, v in CLASS_COLORS.items()}, ax=ax)
    ax.set_title(f'{col} theo lớp')
plt.tight_layout(); plt.show()
display(df.groupby(TARGET)[NUM_COLS].median().rename(index=CLASS_NAMES).T.rename_axis('Median'))
""")

md("### 1.10. Key Insights từ EDA\n\n" + note("EDA"))
