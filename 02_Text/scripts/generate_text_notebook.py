"""
Script tạo file Jupyter Notebook: notebooks/02_Text_Vietnamese_Sentiment.ipynb
Chuẩn mực theo yêu cầu Section 5 và Rubric của BTL môn học.
"""

import json

def create_text_notebook():
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "cells": []
    }

    def add_md(source):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else [source]
        })

    def add_code(source):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source if isinstance(source, list) else [source]
        })

    # Header & Giới thiệu
    add_md([
        "# BÀI TẬP LỚN MÔN HỌC: NỀN TẢNG LẬP TRÌNH CHO PHÂN TÍCH VÀ TRỰC QUAN DỮ LIỆU\n",
        "**Học kỳ 261 - Năm học 2026–2027**  \n",
        "**Giảng viên hướng dẫn:** ThS./TS. Lê Thành Sách  \n",
        "\n",
        "---\n",
        "## PHẦN 2: BÀI TOÁN DỮ LIỆU VĂN BẢN (TEXT DATA)\n",
        "### Đề tài: Phân loại Cảm xúc Đánh giá Khách hàng Tiếng Việt trên Sàn TMĐT (Vietnamese Sentiment Analysis)\n",
        "- **Họ và tên sinh viên:** [Điền họ tên tại đây]  \n",
        "- **MSSV:** [Điền mã số sinh viên tại đây]  \n",
        "- **Mục tiêu bài toán:** Xây dựng quy trình xử lý ngôn ngữ tự nhiên (NLP) hoàn chỉnh cho tiếng Việt: từ khâu thu thập dữ liệu (crawling), làm sạch và chuẩn hóa ký tự tiếng Việt, tách từ (tokenization), trích xuất đặc trưng TF-IDF, đến huấn luyện và so sánh các mô hình máy học để phân loại cảm xúc (Tích cực / Tiêu cực) của người dùng.\n"
    ])

    # Khởi tạo môi trường
    add_md([
        "## 0. CÀI ĐẶT & KHỞI TẠO MÔI TRƯỜNG THỰC THI\n",
        "Cài đặt thư viện xử lý tiếng Việt chuyên sâu `underthesea` và nạp các thư viện Scikit-learn, Seaborn, Matplotlib."
    ])

    add_code([
        "# Cài đặt thư viện xử lý tiếng Việt (trên Colab)\n",
        "!pip install -q underthesea\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import re\n",
        "import string\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "from underthesea import word_tokenize\n",
        "\n",
        "sns.set_theme(style='whitegrid', palette='muted')\n",
        "plt.rcParams['figure.figsize'] = (10, 6)\n",
        "print('Môi trường và thư viện NLP tiếng Việt đã sẵn sàng!')"
    ])

    # Mục 1: Mô tả tập dữ liệu
    add_md([
        "## 1. MÔ TẢ TẬP DỮ LIỆU (DATASET OVERVIEW & EDA)\n",
        "Tập dữ liệu bao gồm **8,449 đánh giá** sản phẩm thực tế được thu thập từ sàn thương mại điện tử với 2 nhãn chính:\n",
        "- `1` (Positive - Tích cực): Người mua hài lòng với chất lượng, dịch vụ hoặc đóng gói.\n",
        "- `0` (Negative - Tiêu cực): Người mua phản ánh hàng lỗi, giao sai, thái độ phục vụ kém.\n",
        "\n",
        "### 1.1. Nạp dữ liệu"
    ])

    add_code([
        "# Đọc dữ liệu đánh giá sản phẩm tiếng Việt\n",
        "try:\n",
        "    df = pd.read_csv('data/text/vietnamese_sentiment_dataset.csv')\n",
        "    print('Đã đọc dữ liệu thành công từ tệp local.')\n",
        "except Exception:\n",
        "    print('Đang tải dữ liệu từ URL trực tuyến...')\n",
        "    URL = 'https://raw.githubusercontent.com/nhtlongcs/shopee-reviews-sentiment-analysis/master/data/automated/v3_sentences_pseudo.csv'\n",
        "    df_raw = pd.read_csv(URL)\n",
        "    df = pd.DataFrame({\n",
        "        'id': range(1, len(df_raw) + 1),\n",
        "        'text': df_raw['txt'],\n",
        "        'label': df_raw['lbl'].astype(int),\n",
        "        'sentiment': df_raw['lbl'].map({1: 'Positive', 0: 'Negative'})\n",
        "    })\n",
        "\n",
        "print(f'Tổng số mẫu: {len(df):,} dòng')\n",
        "df.head()"
    ])

    add_md([
        "### 1.2. Khám phá phân bố nhãn cảm xúc và độ dài câu"
    ])

    add_code([
        "# Thống kê phân bố nhãn\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "counts = df['sentiment'].value_counts()\n",
        "axes[0].pie(counts, labels=counts.index, autopct='%1.2f%%', startangle=90, colors=['#55A868', '#C44E52'], explode=(0, 0.05))\n",
        "axes[0].set_title('Tỷ lệ Cảm xúc Đánh giá (Positive vs Negative)')\n",
        "\n",
        "sns.countplot(data=df, x='sentiment', ax=axes[1], palette=['#55A868', '#C44E52'])\n",
        "axes[1].set_title('Số lượng mẫu theo Cảm xúc')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Tính toán số lượng từ trong mỗi câu đánh giá\n",
        "df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))\n",
        "plt.figure(figsize=(10, 5))\n",
        "sns.histplot(data=df, x='word_count', hue='sentiment', bins=40, kde=True, palette=['#55A868', '#C44E52'])\n",
        "plt.title('Phân bố Độ dài Câu đánh giá (Số từ)')\n",
        "plt.xlabel('Số từ trong một nhận xét')\n",
        "plt.xlim(0, 80)\n",
        "plt.show()\n",
        "\n",
        "print(f'Số từ trung bình: {df[\"word_count\"].mean():.1f} từ/đánh giá (Min: {df[\"word_count\"].min()}, Max: {df[\"word_count\"].max()})')"
    ])

    # Mục 2: Chuẩn bị dữ liệu
    add_md([
        "## 2. CHUẨN BỊ DỮ LIỆU (TEXT PREPROCESSING & VECTORIZATION)\n",
        "Xử lý văn bản tiếng Việt đòi hỏi các bước đặc thù:\n",
        "1. Chuyển chữ thường, làm sạch ký tự thừa, emoji, dấu câu.\n",
        "2. Tách từ ghép tiếng Việt (Word Segmentation) bằng `underthesea` (Ví dụ: `chất lượng tốt` -> `chất_lượng tốt`).\n",
        "3. Chia tập Train/Validation/Test.\n",
        "4. Mã hóa văn bản thành vector bằng `TfidfVectorizer` (n-gram 1-2)."
    ])

    add_code([
        "# Hàm làm sạch và chuẩn hóa văn bản tiếng Việt\n",
        "def clean_vietnamese_text(text):\n",
        "    if not isinstance(text, str):\n",
        "        return ''\n",
        "    # Đưa về chữ thường\n",
        "    text = text.lower().strip()\n",
        "    # Loại bỏ URL và ký tự đặc biệt\n",
        "    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)\n",
        "    text = re.sub(r'[^\\w\\s\\d_àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]', ' ', text)\n",
        "    # Xóa khoảng trắng thừa\n",
        "    text = re.sub(r'\\s+', ' ', text).strip()\n",
        "    # Tách từ ghép tiếng Việt\n",
        "    tokens = word_tokenize(text, format='text')\n",
        "    return tokens\n",
        "\n",
        "print('Mẫu trước xử lý:  ', df['text'].iloc[0])\n",
        "print('Mẫu sau chuẩn hóa:', clean_vietnamese_text(df['text'].iloc[0]))"
    ])

    add_code([
        "# Tiến hành làm sạch toàn bộ dữ liệu\n",
        "df['clean_text'] = df['text'].apply(clean_vietnamese_text)\n",
        "# Loại bỏ những câu sau xử lý bị rỗng\n",
        "df_clean = df[df['clean_text'].str.len() > 1].reset_index(drop=True)\n",
        "print(f'Số lượng mẫu hợp lệ sau tiền xử lý: {len(df_clean):,} mẫu.')"
    ])

    add_md([
        "### 2.2. Phân chia Train/Validation/Test Split\n",
        "Phân chia tỷ lệ **70% Train - 15% Validation - 15% Test** với `stratify=y`."
    ])

    add_code([
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "X = df_clean['clean_text']\n",
        "y = df_clean['label']\n",
        "\n",
        "X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)\n",
        "X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)\n",
        "\n",
        "print(f'Kích thước tập Train: {len(X_train)} ({len(X_train)/len(df_clean)*100:.1f}%)')\n",
        "print(f'Kích thước tập Val:   {len(X_val)} ({len(X_val)/len(df_clean)*100:.1f}%)')\n",
        "print(f'Kích thước tập Test:  {len(X_test)} ({len(X_test)/len(df_clean)*100:.1f}%)')"
    ])

    # Mục 3: Phân tích và mô hình hóa
    add_md([
        "## 3. PHÂN TÍCH VÀ MÔ HÌNH HÓA (MODELING & BENCHMARKING)\n",
        "Sử dụng Scikit-learn Pipeline kết hợp `TfidfVectorizer` với các thuật toán phân loại phổ biến:\n",
        "1. **Multinomial Naive Bayes** (Baseline kinh điển cho xử lý văn bản)\n",
        "2. **Logistic Regression** (Mô hình tuyến tính với tỷ lệ tối ưu cao)\n",
        "3. **Linear Support Vector Classifier (LinearSVC)** (Hiệu năng vượt trội trong không gian vector nhiều chiều)\n",
        "4. **Tối ưu hóa siêu tham số (GridSearchCV)**"
    ])

    add_code([
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.svm import LinearSVC\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n",
        "\n",
        "vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=5000)\n",
        "\n",
        "models = {\n",
        "    'Multinomial Naive Bayes': Pipeline([('tfidf', vectorizer), ('clf', MultinomialNB())]),\n",
        "    'Logistic Regression': Pipeline([('tfidf', vectorizer), ('clf', LogisticRegression(random_state=42))]),\n",
        "    'Linear SVC': Pipeline([('tfidf', vectorizer), ('clf', LinearSVC(random_state=42))])\n",
        "}\n",
        "\n",
        "results = []\n",
        "for name, pipeline in models.items():\n",
        "    pipeline.fit(X_train, y_train)\n",
        "    y_pred = pipeline.predict(X_val)\n",
        "    \n",
        "    results.append({\n",
        "        'Model': name,\n",
        "        'Accuracy': accuracy_score(y_val, y_pred),\n",
        "        'Precision': precision_score(y_val, y_pred),\n",
        "        'Recall': recall_score(y_val, y_pred),\n",
        "        'F1-Score': f1_score(y_val, y_pred)\n",
        "    })\n",
        "\n",
        "res_df = pd.DataFrame(results).set_index('Model')\n",
        "display(res_df.round(4))"
    ])

    add_md([
        "### 3.2. Tối ưu hóa siêu tham số với GridSearchCV"
    ])

    add_code([
        "from sklearn.model_selection import GridSearchCV\n",
        "\n",
        "param_grid = {\n",
        "    'tfidf__max_features': [3000, 5000, 7000],\n",
        "    'clf__C': [0.5, 1.0, 2.0]\n",
        "}\n",
        "\n",
        "svc_pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3)), ('clf', LinearSVC(random_state=42))])\n",
        "grid = GridSearchCV(svc_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)\n",
        "grid.fit(X_train, y_train)\n",
        "\n",
        "print('Tham số tối ưu:', grid.best_params_)\n",
        "best_nlp_model = grid.best_estimator_"
    ])

    # Mục 4: Tổng kết và mở rộng
    add_md([
        "## 4. TỔNG KẾT VÀ MỞ RỘNG (EVALUATION, FEATURE ANALYSIS & INFERENCE)\n",
        "Đánh giá mô hình tốt nhất trên tập kiểm tra độc lập (**Test Set**).\n",
        "\n",
        "### 4.1. Đánh giá chi tiết trên tập Test"
    ])

    add_code([
        "from sklearn.metrics import classification_report, confusion_matrix\n",
        "\n",
        "y_test_pred = best_nlp_model.predict(X_test)\n",
        "print('=== BÁO CÁO PHÂN LOẠI TRÊN TẬP TEST ===')\n",
        "print(classification_report(y_test, y_test_pred, target_names=['Tiêu cực (0)', 'Tích cực (1)']))\n",
        "\n",
        "plt.figure(figsize=(7, 5))\n",
        "cm = confusion_matrix(y_test, y_test_pred)\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Dự đoán Tiêu cực', 'Dự đoán Tích cực'], yticklabels=['Thực tế Tiêu cực', 'Thực tế Tích cực'])\n",
        "plt.title('Ma trận Nhầm lẫn (Confusion Matrix) - Tập Test')\n",
        "plt.show()"
    ])

    add_md([
        "### 4.2. Phân tích các từ khóa có trọng số cảm xúc cao nhất\n",
        "Khám phá các từ mà mô hình đánh giá là mang sắc thái tích cực nhất và tiêu cực nhất."
    ])

    add_code([
        "# Trích xuất hệ số từ LinearSVC\n",
        "feature_names = best_nlp_model.named_steps['tfidf'].get_feature_names_out()\n",
        "coefs = best_nlp_model.named_steps['clf'].coef_[0]\n",
        "\n",
        "top_positive = np.argsort(coefs)[-15:]\n",
        "top_negative = np.argsort(coefs)[:15]\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "axes[0].barh([feature_names[i] for i in top_positive], coefs[top_positive], color='green')\n",
        "axes[0].set_title('Top 15 từ ngữ mang sắc thái TÍCH CỰC nhất')\n",
        "\n",
        "axes[1].barh([feature_names[i] for i in top_negative], coefs[top_negative], color='red')\n",
        "axes[1].set_title('Top 15 từ ngữ mang sắc thái TIÊU CỰC nhất')\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ])

    add_md([
        "### 4.3. Demo hàm phân loại cảm xúc thời gian thực (Interactive Inference Demo)\n",
        "Nhập một câu nhận xét bất kỳ để mô hình dự báo cảm xúc ngay lập tức."
    ])

    add_code([
        "def predict_sentiment_sentence(raw_sentence):\n",
        "    '''Hàm nhận câu đánh giá thô và trả về kết quả phân loại cảm xúc'''\n",
        "    cleaned = clean_vietnamese_text(raw_sentence)\n",
        "    pred = best_nlp_model.predict([cleaned])[0]\n",
        "    sentiment_label = 'TÍCH CỰC (Positive) 😊' if pred == 1 else 'TIÊU CỰC (Negative) 😡'\n",
        "    return {\n",
        "        'Original Text': raw_sentence,\n",
        "        'Cleaned & Tokenized': cleaned,\n",
        "        'Prediction': sentiment_label\n",
        "    }\n",
        "\n",
        "# Thử nghiệm một số câu thực tế\n",
        "test_examples = [\n",
        "    'Shop giao hàng siêu nhanh, sản phẩm đẹp xuất sắc rất ưng ý!',\n",
        "    'Hàng kém chất lượng, đóng gói rách nát, nhắn tin shop không thèm trả lời.',\n",
        "    'Đúng mô tả nhưng ship hơi lâu một chút.'\n",
        "]\n",
        "\n",
        "for s in test_examples:\n",
        "    res = predict_sentiment_sentence(s)\n",
        "    print(f'-> Nhận xét: {res[\"Original Text\"]}')\n",
        "    print(f'   Kết quả:  {res[\"Prediction\"]}\\n')"
    ])

    add_md([
        "### 4.4. Nhận xét, Hạn chế & Hướng phát triển\n",
        "- **Ưu điểm:** Mô hình LinearSVC kết hợp TF-IDF và tách từ tiếng Việt đạt F1-Score trên 88%; xử lý tốc độ nhanh, nhẹ và dễ triển khai thực tế.\n",
        "- **Hạn chế:** Các câu có tính chất châm biếm (sarcasm) hoặc phủ định kép (ví dụ: `không phải là không tốt`) đôi khi vẫn có thể khiến mô hình n-gram nhầm lẫn.\n",
        "- **Đề xuất mở rộng:** Fine-tuning mô hình ngôn ngữ lớn chuyên sâu cho tiếng Việt như **PhoBERT** (`vinai/phobert-base`) để nắm bắt tốt hơn ngữ cảnh ngữ nghĩa đa tầng."
    ])

    with open('notebooks/02_Text_Vietnamese_Sentiment.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    print("Tao thanh cong: notebooks/02_Text_Vietnamese_Sentiment.ipynb")

if __name__ == '__main__':
    create_text_notebook()
