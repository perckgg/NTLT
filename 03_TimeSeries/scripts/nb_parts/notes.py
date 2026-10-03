# Nhận xét viết dựa trên kết quả chạy thật (dữ liệu 2024-12-01 → 2026-09-30, random_state=42).
NOTES = {
"EDA": """
- **Mức ô nhiễm trung bình tương đương nhưng có hình dạng khác nhau:** PM2.5 trung bình Hà Nội 31.7 µg/m³, TP.HCM 32.1 µg/m³.
  Hà Nội biến động mạnh hơn (std 20.6 so với 16.3; max 332 so với 171). Cả hai đều vượt ngưỡng WHO 24h (15 µg/m³)
  trong phần lớn số giờ: Hà Nội 56%, TP.HCM 82%.
- **Theo mùa:** Hà Nội có mùa ô nhiễm rõ rệt vào mùa đông (tháng 12: 53, tháng 1: 55 µg/m³) và sạch nhất vào tháng 8 (19 µg/m³),
  tức chênh lệch gần 3 lần. TP.HCM cũng ô nhiễm cao nhất vào cuối mùa khô / đầu năm (tháng 1: 46 µg/m³) và thấp nhất vào giữa mùa mưa (tháng 7: 24 µg/m³), nhưng biên độ hẹp hơn.
- **Theo giờ:** TP.HCM có đỉnh buổi sáng lúc 7h (≈41 µg/m³), trùng giờ cao điểm giao thông; Hà Nội đỉnh vào buổi tối (≈34 µg/m³ lúc 22h),
  phù hợp với việc lớp nghịch nhiệt ban đêm giữ bụi lại gần mặt đất.
- **Tương quan với thời tiết:** Hà Nội tương quan âm với nhiệt độ (-0.31), mây (-0.25) và dương với áp suất (+0.31), cho thấy
  không khí lạnh, áp cao của mùa đông làm tích tụ bụi. TP.HCM phụ thuộc nhiều nhất vào tốc độ gió (-0.36): gió mạnh thì bụi được phát tán.
- **Tự tương quan:** tự tương quan lag 1h rất cao (0.92 / 0.88), còn lag 24h giảm xuống 0.56 / 0.45. Vì vậy dự báo trước 24h
  khó hơn đáng kể so với dự báo giờ kế tiếp, và cần thêm đặc trưng thời tiết, mùa, chu kỳ ngày.
- **Chất lượng dữ liệu:** Hà Nội thiếu 32.8% giờ PM2.5, tập trung từ cuối 2025 (tháng 12/2025 thiếu 97%); TP.HCM thiếu 8.4%.
  Đây là vấn đề phải xử lý kỹ ở Phần 2.
""",

"VAL": """
**Nhận xét Validation:**
- Ở cả hai thành phố, mọi mô hình học máy đều có MAE thấp hơn baseline Persistence. Mức cải thiện lớn ở Hà Nội
  (14.97 → 12.38 với RandomForest, giảm ≈17%) và nhỏ ở TP.HCM (7.71 → 7.07 với Ridge, giảm ≈8%).
- Baseline có R² âm (-0.06 / -0.15): lấy nguyên giá trị hiện tại làm dự báo cho 24h sau còn tệ hơn dự báo bằng giá trị trung bình.
- Ở Hà Nội, ba mô hình học máy chênh nhau rất ít (MAE 12.4–12.6). Ở TP.HCM, Ridge tốt nhất: khi tín hiệu yếu (R² chỉ 0.08–0.14),
  mô hình tuyến tính có regularization mạnh (alpha=100) ít bị overfit hơn mô hình cây.
- GridSearch chọn learning_rate nhỏ (0.03) và cây ít lá (15) cho HGB, tức là ưu tiên mô hình đơn giản. Điều này nhất quán với
  việc dữ liệu nhiễu và chỉ dài 22 tháng.
""",

"CREATIVE": """
**Chú thích Hình 6:** Trên tập Test độc lập, mô hình tốt nhất giảm MAE so với baseline Persistence
**13.9% ở Hà Nội** (RandomForest: 8.32 → 7.16 µg/m³) và **10.6% ở TP.HCM** (Ridge: 9.53 → 8.52 µg/m³).
Mô hình tốt nhất khác nhau giữa hai thành phố: Hà Nội hưởng lợi từ mô hình phi tuyến (RandomForest, HGB) nhờ tương tác
giữa mùa, mưa và mức bụi gần đây, còn ở TP.HCM mô hình tuyến tính Ridge lại ổn định nhất.
Như vậy **không có một mô hình chung tốt nhất cho cả hai thành phố**, và đó là lý do nên huấn luyện riêng từng nơi thay vì gộp dữ liệu.
Lưu ý: hai tập Test không cùng giai đoạn (Hà Nội 19/03–29/09/2026, TP.HCM 29/06–29/09/2026, do Hà Nội thiếu nhiều dữ liệu hơn)
và đều rơi vào giai đoạn ít ô nhiễm. Vì vậy không nên so sánh trực tiếp giá trị MAE tuyệt đối giữa hai thành phố, mà nên so sánh mức cải thiện tương đối so với baseline.
(MAE Test của Hà Nội thấp hơn Validation (7.2 so với 12.4) vì giai đoạn Test không có mùa đông; ở TP.HCM thì ngược lại (8.5 so với 7.1) do mức bụi tăng trở lại vào tháng 9.)
""",

"ERROR": """
**Nhận xét phân tích lỗi:**
- **Theo giờ:** sai số Hà Nội khá đều trong ngày (6.2–8.5 µg/m³). TP.HCM sai nhiều nhất vào ban ngày (đỉnh 9.9 lúc 14h),
  là lúc giao thông và đối lưu làm PM2.5 dao động mạnh.
- **Theo tháng:** sai số tăng dần về cuối giai đoạn Test (TP.HCM: 3.2 vào cuối tháng 6 → 10.9 vào tháng 9/2026), khi mức ô nhiễm bắt đầu tăng trở lại.
  Mô hình học chủ yếu trên phân phối cũ nên phản ứng chậm với thay đổi theo mùa.
- **Theo mức AQI:** dự báo đúng mức 82.6% (Hà Nội) và 76.4% (TP.HCM); lệch không quá 1 mức là 99.1% / 98.6%.
  Lỗi chủ yếu nằm ở việc **dự báo thấp các giờ ô nhiễm**: ở TP.HCM, 281/359 giờ "Kém" bị đoán thành "Trung bình", và 29/71 giờ "Xấu"
  bị đoán thành "Trung bình". Mô hình không bao giờ dự báo mức "Tốt", tức là có xu hướng kéo dự báo về giá trị trung bình.
- **Hệ quả thực tế:** mô hình hữu ích để nắm xu hướng chung, nhưng **chưa đủ tin cậy để cảnh báo sớm các đợt ô nhiễm nặng**.
  Đây là trường hợp mà một lỗi bỏ sót gây hậu quả lớn hơn một lỗi báo nhầm.
""",

"IMPORTANCE": """
**Nhận xét:** Ở cả hai thành phố, **mức PM2.5 hiện tại (`pm25_lag_0`) là đặc trưng quan trọng nhất**, phù hợp với tự tương quan cao.
Điểm khác biệt nằm ở các đặc trưng tiếp theo:
- **Hà Nội:** trung bình 6h gần nhất, **lượng mưa 24h** (`rain_24h`) và **mùa** (`season`). Bụi Hà Nội phụ thuộc chế độ mùa và việc mưa rửa trôi.
- **TP.HCM:** **nhiệt độ**, **chu kỳ giờ trong ngày** (`hour_cos`) và **tốc độ gió**. Bụi TP.HCM gắn với nhịp sinh hoạt/giao thông hằng ngày và khả năng phát tán của gió.

Điều này khớp với EDA (Hà Nội có mùa rõ rệt, TP.HCM có đỉnh giờ cao điểm và tương quan mạnh với gió).
Đây cũng là bằng chứng nữa cho thấy nên xây dựng mô hình riêng cho từng thành phố.
""",

"CONCLUSION": """
**Kết luận:**
- Đã xây dựng pipeline dự báo PM2.5 trước 24h, áp dụng giống nhau nhưng **huấn luyện độc lập** cho Hà Nội và TP.HCM, dùng dữ liệu
  quan trắc thật từ OpenAQ kết hợp thời tiết Open-Meteo (12/2024 – 09/2026).
- Trên tập Test, mô hình tốt nhất giảm MAE **13.9% (Hà Nội, RandomForest)** và **10.6% (TP.HCM, Ridge)** so với baseline Persistence,
  dự báo đúng mức AQI 76–83% số giờ.
- Hai thành phố có cơ chế ô nhiễm khác nhau: Hà Nội chi phối bởi mùa và mưa, TP.HCM bởi chu kỳ ngày và gió.

**Hạn chế:**
- Mỗi thành phố chỉ có **một trạm**. Trạm TP.HCM là cảm biến giá rẻ (AirGradient), có thể kém chính xác khi độ ẩm cao.
- Hà Nội thiếu nhiều dữ liệu (32.8%, phần lớn từ cuối 2025), nên tập Test không chứa mùa đông, là mùa ô nhiễm nhất.
- Chỉ có 22 tháng dữ liệu, tức chưa đến 2 chu kỳ mùa, nên mô hình khó học được quy luật theo mùa.
- Mô hình thường dự báo thấp các đợt ô nhiễm cao. Đặc trưng thời tiết dùng giá trị quan trắc tại thời điểm $t$, chưa dùng thời tiết dự báo cho $t+24h$.

**Hướng phát triển:**
- Thêm **dự báo thời tiết** cho thời điểm mục tiêu (Open-Meteo forecast API) và dữ liệu nhiều trạm trong cùng thành phố.
- Dùng hàm mất mát / trọng số mẫu ưu tiên các giờ ô nhiễm cao, hoặc thêm mô hình phân loại "có vượt ngưỡng hay không" để cảnh báo sớm.
- Thử các mô hình chuỗi thời gian chuyên biệt (SARIMAX, LightGBM với nhiều horizon, LSTM) khi có dữ liệu dài hơn.
""",
}
