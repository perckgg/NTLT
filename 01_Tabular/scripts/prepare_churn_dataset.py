"""
Chuẩn bị tập dữ liệu Bank Customer Churn cho bài Tabular.

Dữ liệu gốc (Churn_Modelling, 10,000 dòng × 14 cột) KHÔNG có missing values.
Đề BTL yêu cầu dữ liệu có missing để thực hành thống kê và xử lý, nên script này
tạo missing ngẫu nhiên kiểu MCAR (Missing Completely At Random) ở 5 cột với tỉ lệ 2–4%.
Việc này được ghi rõ trong notebook và README.

Notebook dùng lại đúng hàm inject_missing (cùng seed), nên dù đọc file local hay tải
file gốc từ URL thì dữ liệu đều giống hệt nhau.

Chạy từ thư mục 01_Tabular:  py scripts/prepare_churn_dataset.py
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/tabular/Churn_Modelling_raw.csv"
OUTPUT_PATH = "data/tabular/customer_churn_dataset.csv"
SEED = 42
MISSING_RATES = {"CreditScore": 0.025, "Geography": 0.020, "Age": 0.035,
                 "Balance": 0.030, "EstimatedSalary": 0.040}


def inject_missing(df, rates=MISSING_RATES, seed=SEED):
    df = df.copy()
    rng = np.random.default_rng(seed)
    for col, rate in rates.items():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float)
        df.loc[rng.random(len(df)) < rate, col] = np.nan
    return df


def main():
    raw = pd.read_csv(RAW_PATH)
    df = inject_missing(raw)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Đã lưu {OUTPUT_PATH}: {df.shape[0]} dòng × {df.shape[1]} cột")
    for col in MISSING_RATES:
        n = df[col].isna().sum()
        print(f"  {col:16s}: {n:4d} missing ({n / len(df):.2%})")


if __name__ == "__main__":
    main()
