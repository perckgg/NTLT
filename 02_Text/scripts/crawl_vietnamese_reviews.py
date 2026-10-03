"""
Script Crawler: Thu thập đánh giá sản phẩm tiếng Việt từ sàn thương mại điện tử Tiki
Phục vụ mục tiêu BTL: Tự thu thập dữ liệu (crawling) để đạt điểm tối đa tính sáng tạo.
"""

import json
import time
import csv
import urllib.request
import urllib.error

def crawl_tiki_reviews(product_ids, limit_per_product=50, max_pages=5):
    """
    Crawl danh sách reviews từ Tiki public API cho danh sách các sản phẩm.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*'
    }
    
    collected_reviews = []
    
    for pid in product_ids:
        print(f"[*] Dang thu thap danh gia cho san pham ID: {pid}...")
        for page in range(1, max_pages + 1):
            url = f"https://tiki.vn/api/v2/reviews?product_id={pid}&sort=score|desc,id|desc,stars|all&page={page}&limit={limit_per_product}"
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode('utf-8'))
                        reviews = data.get('data', [])
                        if not reviews:
                            break
                        for r in reviews:
                            content = r.get('content', '').strip()
                            rating = r.get('rating', 0)
                            if content:
                                # Gán nhãn: rating >= 4 là Positive (1), rating <= 2 là Negative (0)
                                if rating >= 4:
                                    lbl = 1
                                    sentiment = 'Positive'
                                elif rating <= 2:
                                    lbl = 0
                                    sentiment = 'Negative'
                                else:
                                    continue # Bỏ qua trung tính để bài toán phân loại rõ ràng
                                    
                                collected_reviews.append({
                                    'product_id': pid,
                                    'text': content,
                                    'rating': rating,
                                    'label': lbl,
                                    'sentiment': sentiment
                                })
                time.sleep(0.5) # Nghi thức tôn trọng server (rate-limiting)
            except Exception as e:
                print(f"    [!] Loi tai page {page}: {e}")
                break
                
    print(f"[+] Hoan tat crawl: Tong so danh gia thu thap duoc: {len(collected_reviews)}")
    return collected_reviews

def save_to_csv(reviews, output_filepath):
    if not reviews:
        print("[!] Khong co review nao de luu.")
        return
    with open(output_filepath, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'product_id', 'text', 'rating', 'label', 'sentiment'])
        for idx, r in enumerate(reviews, start=1):
            writer.writerow([idx, r['product_id'], r['text'], r['rating'], r['label'], r['sentiment']])
    print(f"[+] Da luu vao file: {output_filepath}")

if __name__ == '__main__':
    # Danh sách product_id mẫu từ các ngành hàng: Sách, Công nghệ, Gia dụng
    sample_product_ids = [
        188313854, 274987084, 197825227, 245999882, 194165561
    ]
    print("Khoi chay demo crawler...")
    reviews = crawl_tiki_reviews(sample_product_ids, limit_per_product=20, max_pages=2)
    save_to_csv(reviews, 'data/text/crawled_reviews_demo.csv')
