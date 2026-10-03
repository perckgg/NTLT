"""
Script chuẩn hóa tập dữ liệu đánh giá sản phẩm Shopee/Tiki tiếng Việt
Đạt đầy đủ tiêu chí:
- > 2000 mẫu (thực tế 8,449 mẫu)
- Ngôn ngữ: Tiếng Việt
- Tác vụ: Phân loại cảm xúc (Sentiment Analysis) nhị phân (0: Negative, 1: Positive)
"""

import csv

def standardize_text_dataset():
    input_path = 'data/text/shopee_reviews_raw.csv'
    output_path = 'data/text/vietnamese_sentiment_dataset.csv'
    
    clean_rows = []
    with open(input_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Header: txt, lbl, prob_positve
        for idx, row in enumerate(reader, start=1):
            if len(row) >= 2:
                text = row[0].strip()
                lbl = row[1].strip()
                if text and lbl in ['0', '1']:
                    sentiment = 'Positive' if lbl == '1' else 'Negative'
                    clean_rows.append([idx, text, int(lbl), sentiment])
                    
    with open(output_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'label', 'sentiment'])
        writer.writerows(clean_rows)
        
    print(f"Da chuan hoa thanh cong: {len(clean_rows)} mau.")
    print(f"File luu tai: {output_path}")

if __name__ == '__main__':
    standardize_text_dataset()
