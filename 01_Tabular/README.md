# Part 1 — Tabular: Bank Customer Churn Prediction

Course project for **Programming Foundations for Data Analysis and Visualization** (Semester 261, Instructor: Lê Thành Sách) — tabular data part.

| | |
|---|---|
| **Task** | Binary classification: will a customer leave the bank (`Exited = 1`) or not |
| **Data** | Bank Customer Churn (Churn_Modelling): 10,000 rows × 14 columns, 20.4% churn (imbalance 3.9 : 1) |
| **Notebook** | [`notebooks/01_Tabular_Customer_Churn.ipynb`](notebooks/01_Tabular_Customer_Churn.ipynb) — runs end-to-end on Colab (Run all) |
| **Executed copy (with outputs)** | [`result/01_Tabular_Customer_Churn_executed.ipynb`](result/01_Tabular_Customer_Churn_executed.ipynb) |
| **Result (Test)** | RandomForest + `class_weight='balanced'` + threshold 0.61: **F1 = 0.639**, Recall = 0.60, Precision = 0.68, ROC-AUC = 0.863 |

## 1. Approach

The notebook answers two questions:
1. **Which characteristics distinguish churners from customers who stay?** → EDA and permutation importance.
2. **How does class imbalance affect the model, and how many more churners can we catch by handling it?** → a step-by-step experiment on the same Test set.

Every preprocessing and modeling decision is **backed by EDA numbers**. The EDA follows the structure of the instructor's sample report
([Adult Income EDA Report](https://ltsach.github.io/AILearningHub/01_Data_Analysis/01_EDA/adult_income_tabular/eda_report_tabular.html)):
Missing → Numerical → Categorical → Target → Correlation → Outliers (IQR) → Target vs Features → Key Insights.

## 2. Method (following the 4 required sections)

| Section | Main content |
|---|---|
| **1. Data description & EDA** | Feature description table; missing values with severity and per-class comparison; distributions + skewness + boxplots; cardinality → encoding choice; class ratio → metric choice; correlation heatmap; IQR outlier table; churn rate per group; Key Insights with numbers |
| **2. Data preparation** | Drop ID columns → engineer `IsZeroBalance`, `ProductsGroup` → **stratified 70/15/15** split → `Pipeline` + `ColumnTransformer`: median/mode imputation, **RobustScaler**, **One-Hot** (fitted on Train only) → before/after scaling figure |
| **3. Modeling** | Baselines (Dummy, Logistic Regression) → 3 algorithms (LogReg, RandomForest, HistGradientBoosting) × {unweighted, `balanced`} → GridSearchCV 5-fold (with `class_weight` in the grid) → **decision threshold chosen on Validation** |
| **4. Summary** | **Single** evaluation on Test (classification report, confusion matrix, ROC, PR curve) → step-by-step improvement experiment → permutation importance → error analysis by customer group → inference function `predict_customer_churn` → conclusions, limitations, future work |

### Key decisions and evidence

| Decision | Evidence from the data |
|---|---|
| Impute (median/mode) instead of dropping rows | Each column is only 2–4% missing, but 14.1% of rows have at least one missing cell |
| Treat `NumOfProducts` as **categorical**, merge 3 and 4 | Churn rate is non-linear: 1 → 27.7%, 2 → 7.6%, 3 → 82.7%, 4 → 100%; group 4 is only 0.6% of customers |
| Add `IsZeroBalance` | 35.2% of customers have a zero balance (bimodal distribution); churn 13.8% vs 24.0% |
| Keep `Age` outliers, use RobustScaler | Outliers are real values (63–92 years); churn rises up to ~60 and then falls — removing them would lose this pattern |
| F1 / Recall / PR-AUC instead of Accuracy | A model that always predicts "stay" already reaches 79.6% Accuracy |
| `class_weight='balanced'` + threshold tuning | Validation Recall 0.36 → 0.79 (LogReg); GridSearch picks `balanced` for both tree models |

## 3. What is special (creativity)

1. **Step-by-step improvement experiment on the same Test set.** The figure includes comparison numbers and answers question 2:

   | Configuration | Caught / 305 | False alarms | F1 |
   |---|---|---|---|
   | Baseline Logistic Regression | 113 | 40 | 0.493 |
   | RandomForest (imbalance not handled) | 132 | 33 | 0.562 |
   | + `class_weight='balanced'` | 218 | 182 | 0.618 |
   | + threshold 0.61 (chosen on Validation) | **183** | **85** | **0.639** |

   → The final model catches **70 more churners (+62%)** than the baseline. Class weighting raises Recall; threshold tuning rebalances Precision.
2. **With / without `class_weight` experiment** for all 3 algorithms on Validation. Recall rises sharply while ROC-AUC barely changes,
   i.e. weighting only shifts the decision boundary; it does not make the model separate classes better.
3. **Error analysis by customer group.** Recall is only 0.13 for customers under 30, versus 0.77–0.78 for ages 40–59.
   Missed churners tend to be young and active, i.e. they "look like" loyal customers.
4. **Permutation importance on the original columns** (NumOfProducts and Age lead), consistent with the EDA.
5. **Fully leakage-free pipeline.** Feature engineering, imputation, scaling and encoding all live inside a `Pipeline`, so the inference function accepts raw data, including missing values.
6. **Reproducible:** the notebook is generated by a script; every observation is written from real run results (`random_state = 42`).

## 4. Transparency note about the data

The original dataset **has no missing values**. To meet the assignment requirement (data must contain missing values to practice handling them), we injected
**completely-at-random (MCAR)** missing values into 5 columns: CreditScore 2.4%, Geography 2.1%, Age 3.7%, Balance 3.0%, EstimatedSalary 3.9%,
using the `inject_missing` function (seed 42). The notebook uses **the same function and the same seed**, so reading the local file or downloading the original from the URL gives identical data.
This is a limitation: real-world missingness usually depends on the data (MAR/MNAR).

## 5. Folder structure

```
01_Tabular/
├── README.md
├── data/tabular/
│   ├── Churn_Modelling_raw.csv          # original data (no missing values)
│   └── customer_churn_dataset.csv       # data used by the notebook (with MCAR missing values)
├── notebooks/
│   └── 01_Tabular_Customer_Churn.ipynb  # submission notebook (no outputs)
├── result/
│   └── 01_Tabular_Customer_Churn_executed.ipynb   # executed copy with all outputs
└── scripts/
    ├── prepare_churn_dataset.py         # builds customer_churn_dataset.csv from the original data
    ├── generate_tabular_notebook.py     # generates the notebook from the parts below
    └── nb_parts/
        ├── part1_eda.py                 # Sections 0–1: introduction, EDA
        ├── part2_preparation.py         # Section 2: data preparation
        ├── part3_modeling.py            # Section 3: modeling
        ├── part4_summary.py             # Section 4: evaluation, creativity, error analysis, inference
        └── notes.py                     # observations (written from real results)
```

## 6. How to run

**On Google Colab:** open the notebook and choose *Runtime → Run all*. The notebook looks for data in this order:
local folders → `GITHUB_RAW_BASE` (set it to the group's repo) → original URL + `inject_missing`. So it always runs, even before the data is uploaded.

**Locally (Windows), from the `01_Tabular` folder:**
```powershell
py scripts\prepare_churn_dataset.py        # (optional) regenerate the dataset with missing values
py scripts\generate_tabular_notebook.py    # regenerate the notebook after editing nb_parts/
```
Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (≥ 1.2). A full run takes about 1–2 minutes.

## 7. Before submitting

- [ ] Fill in the group name and members in the first notebook cell.
- [x] Set `GITHUB_RAW_BASE` to the group's repo (`perckgg/NTLT`, branch `main`).
- [ ] Run all on Colab and save the copy with outputs.
- [ ] Prepare the PDF report and video (5–10 minutes). Numbers in the report must match the notebook.
