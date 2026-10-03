# Phần 2: Chuẩn bị dữ liệu
md("""
## 2. CHUẨN BỊ DỮ LIỆU

Nguyên tắc: mọi bước **có học tham số từ dữ liệu** (median để điền thiếu, IQR để scale, danh mục để one-hot) đều nằm trong
`Pipeline` và chỉ được `fit` trên tập **Train** → tránh rò rỉ dữ liệu (data leakage) sang Validation/Test.

### 2.1. Loại cột định danh và kiểm tra trùng lặp
""")

code("""
data = df.drop(columns=ID_COLS)
print(f'Sau khi bỏ {ID_COLS}: {data.shape[1]} cột | dòng trùng lặp: {data.duplicated().sum()}')
""")

md("""
### 2.2. Tạo đặc trưng (Feature Engineering) — dựa trên EDA
| Đặc trưng | Lý do (từ EDA) |
|---|---|
| `IsZeroBalance` = 1 nếu `Balance == 0` | Balance có phân phối **hai đỉnh**: ~36% khách hàng số dư bằng 0, nhóm này có tỉ lệ rời bỏ khác hẳn |
| `ProductsGroup` = `NumOfProducts` dạng **categorical**, gộp 3 và 4 thành `3+` | Tỉ lệ rời bỏ theo số sản phẩm **không tuyến tính** (2 sản phẩm thấp nhất, ≥ 3 rất cao); nhóm 4 hiếm (< 1%) nên gộp theo quy tắc nhóm hiếm |

Các phép biến đổi này chỉ tính trên **từng dòng** (không học tham số), được đóng gói bằng `FunctionTransformer`
để hàm suy luận nhận trực tiếp dữ liệu thô.
""")

code("""
def add_features(X):
    X = X.copy()
    X['IsZeroBalance'] = np.where(X['Balance'].isna(), np.nan, (X['Balance'] == 0).astype(float))
    X['ProductsGroup'] = X['NumOfProducts'].clip(upper=3).astype(int).astype(str).replace({'3': '3+'})
    return X.drop(columns=['NumOfProducts'])


feature_maker = FunctionTransformer(add_features)
print(add_features(data.drop(columns=TARGET)).head())
print('\\nTỉ lệ rời bỏ theo IsZeroBalance:',
      data.groupby(add_features(data)['IsZeroBalance'])[TARGET].mean().round(3).to_dict())
""")

md("""
### 2.3. Chia tập Train / Validation / Test (70 / 15 / 15, phân tầng theo nhãn)
- **Train:** huấn luyện. **Validation:** so sánh mô hình, chọn ngưỡng. **Test:** chỉ dùng **một lần** ở Phần 4.
- `stratify=y` giữ tỉ lệ rời bỏ ~20% ở cả ba tập. Chia **trước** khi điền missing/scale để tránh rò rỉ.
""")

code("""
X, y = data.drop(columns=TARGET), data[TARGET]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)

display(pd.DataFrame({name: {'Số mẫu': len(t), 'Tỉ lệ': len(t) / len(X), 'Số rời bỏ': int(t.sum()), '% rời bỏ': t.mean()}
                      for name, t in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]}).T)
""")

md("""
### 2.4. Pipeline tiền xử lý
| Nhóm cột | Missing | Outlier / Scaling | Encoding |
|---|---|---|---|
| Numerical (`CreditScore, Age, Tenure, Balance, EstimatedSalary`) | median | **RobustScaler** (dựa trên median/IQR, ít bị outlier kéo lệch) | — |
| Binary (`HasCrCard, IsActiveMember, IsZeroBalance`) | mode | giữ 0/1 | — |
| Categorical (`Geography, Gender, ProductsGroup`) | mode | — | **One-Hot** (cardinality ≤ 3) |

**Xử lý outlier:** outlier của `Age` (63–92 tuổi) là **giá trị thật**, không phải lỗi nhập liệu. Tỉ lệ rời bỏ tăng theo tuổi đến
khoảng 60 rồi **giảm trở lại** ở nhóm này (xem 1.8 và 1.9); xóa đi sẽ làm mất quy luật phi tuyến đó. Vì vậy nhóm **giữ lại**
outlier và dùng RobustScaler để giảm ảnh hưởng tới mô hình tuyến tính; mô hình cây vốn không nhạy với outlier.
""")

code("""
NUM_FEATURES = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
BIN_FEATURES = ['HasCrCard', 'IsActiveMember', 'IsZeroBalance']
CAT_FEATURES = ['Geography', 'Gender', 'ProductsGroup']


def build_preprocessor():
    columns = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), NUM_FEATURES),
        ('bin', SimpleImputer(strategy='most_frequent'), BIN_FEATURES),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CAT_FEATURES),
    ], verbose_feature_names_out=False)
    return Pipeline([('features', FunctionTransformer(add_features)), ('columns', columns)])


prep = build_preprocessor().fit(X_train)
Xt_train = pd.DataFrame(prep.transform(X_train), columns=prep.named_steps['columns'].get_feature_names_out(),
                        index=X_train.index)
print(f'Sau tiền xử lý: {Xt_train.shape[1]} cột | missing còn lại: {int(Xt_train.isna().sum().sum())}')
print('Median dùng để điền (học từ Train):',
      {c: float(v) for c, v in zip(NUM_FEATURES, prep.named_steps['columns'].named_transformers_['num']['imputer'].statistics_.round(1))})
Xt_train.head()
""")

md("""
### 2.5. Kiểm tra trực quan: trước và sau khi điền thiếu và scale
So sánh phân phối gốc với hai cách scale cho hai biến có outlier/lệch: `Age` và `Balance`.
""")

code("""
fig, axes = plt.subplots(2, 3, figsize=(18, 7))
for r, col in enumerate(['Age', 'Balance']):
    raw = X_train[col]
    filled = raw.fillna(raw.median())
    std_scaled = StandardScaler().fit_transform(filled.to_frame()).ravel()
    sns.histplot(raw.dropna(), bins=40, ax=axes[r, 0], color='#4C72B0')
    axes[r, 0].set_title(f'{col} — gốc (thiếu {raw.isna().sum()} giá trị)')
    sns.histplot(std_scaled, bins=40, ax=axes[r, 1], color='#8DA0CB')
    axes[r, 1].set_title(f'StandardScaler — khoảng [{std_scaled.min():.1f}, {std_scaled.max():.1f}]')
    sns.histplot(Xt_train[col], bins=40, ax=axes[r, 2], color='#55A868')
    axes[r, 2].set_title(f'RobustScaler (dùng trong pipeline) — IQR = 1, median = 0')
plt.tight_layout(); plt.show()
""")

md(note("PREP"))
