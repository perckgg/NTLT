# Nhận xét viết dựa trên kết quả chạy thật (random_state = 42).
NOTES = {
"EDA": """
**Dữ liệu và chất lượng**
- 10,000 khách hàng × 14 cột, không có dòng trùng lặp hay CustomerId trùng. Loại 3 cột định danh, còn 10 feature: 6 numerical, 2 binary, 2 categorical.
- **Missing:** 1,521 ô thiếu ở 5 cột, mỗi cột 2.1–3.9% (đều ở mức *Thấp* < 5%), nhưng **14.1% số dòng** có ít nhất 1 ô thiếu.
  Xóa dòng sẽ mất 14% dữ liệu, nên chọn **điền median (numerical) / mode (categorical)**.
  Tỉ lệ thiếu ở hai lớp gần như bằng nhau (ví dụ Age: 3.8% so với 3.5%), đúng với cơ chế MCAR đã tạo.

**Phân phối**
- **Age** lệch phải (skew = 1.01; median 37, max 92). **Balance** có hai đỉnh: **35.2% khách hàng có số dư bằng 0**,
  nên mean (76k) nhỏ hơn hẳn median (97k). Đây là lý do tạo thêm feature `IsZeroBalance`.
- CreditScore và EstimatedSalary gần đối xứng (|skew| < 0.1); EstimatedSalary gần như phân phối đều.
- Các biến categorical có cardinality ≤ 3, nên dùng **One-Hot Encoding** theo quy tắc của báo cáo mẫu.

**Mục tiêu và tương quan**
- **Mất cân bằng 3.9 : 1** (79.6% ở lại, 20.4% rời bỏ): mô hình luôn đoán "ở lại" đã đạt Accuracy 79.6%.
  Vì vậy cần chia tập phân tầng, dùng class weight, và đánh giá bằng F1 / Recall / PR-AUC.
- Không có đa cộng tuyến: cặp tương quan mạnh nhất chỉ là Balance–NumOfProducts (r = −0.30).
  Tương quan tuyến tính với Exited đều yếu (Age 0.29, IsActiveMember −0.16, Balance 0.12), nên cần mô hình phi tuyến.
- **Outlier:** chủ yếu ở Age (348 KH > 62 tuổi, 3.6%); NumOfProducts có 60 KH dùng 4 sản phẩm (0.6%).

**Quan hệ target – feature (yếu tố phân biệt rõ nhất)**
- **NumOfProducts quan hệ phi tuyến:** 1 sản phẩm 27.7%, **2 sản phẩm chỉ 7.6%**, 3 sản phẩm 82.7%, 4 sản phẩm **100%** rời bỏ.
- **Age:** median 45 tuổi (rời bỏ) so với 36 (ở lại). Tỉ lệ rời bỏ tăng từ 7.5% (≤ 30 tuổi) lên **54.7% (50–62 tuổi)** rồi giảm còn 8.7% (> 70 tuổi).
- **Germany 32.4%** so với France 16.1% / Spain 16.6%; **nữ 25.1%** so với nam 16.5%; **không hoạt động 26.9%** so với hoạt động 14.3%.
""",

"PREP": """
**Nhận xét Phần 2:**
- `IsZeroBalance` tách được hai nhóm có tỉ lệ rời bỏ khác nhau: số dư bằng 0 rời bỏ **13.8%**, có số dư rời bỏ **24.0%**.
- Sau khi chia phân tầng, tỉ lệ rời bỏ ở Train / Validation / Test lần lượt là 20.4% / 20.4% / 20.3%, tức ba tập đồng nhất.
- Pipeline biến 10 cột gốc thành 16 cột số, không còn missing. Median để điền thiếu chỉ học từ Train (ví dụ Age = 37).
- Hình trên cho thấy StandardScaler bị outlier Age kéo dài đuôi phải, còn RobustScaler giữ phần thân phân phối quanh 0 với IQR = 1.
""",

"VAL": """
**Nhận xét Validation:**
- **Accuracy gây hiểu nhầm:** Dummy đạt Accuracy 0.796 nhưng F1 = 0. Logistic Regression thường có Accuracy 0.844 nhưng **chỉ bắt được 36% khách hàng rời bỏ**.
- **`class_weight='balanced'` tăng mạnh Recall ở cả 3 thuật toán:** LogReg 0.36 → **0.79**, RandomForest 0.49 → 0.65, HistGB 0.48 → 0.72.
  F1 cũng tăng (0.487 → 0.583; 0.613 → 0.642; 0.590 → 0.630). Đổi lại Precision giảm (ví dụ LogReg 0.74 → 0.46).
- **ROC-AUC gần như không đổi** khi thêm trọng số (ví dụ RF 0.881 so với 0.879): class weight không làm mô hình phân biệt tốt hơn,
  mà chủ yếu **dịch ranh giới quyết định** về phía lớp thiểu số.
- Mô hình cây vượt Logistic Regression (ROC-AUC ≈ 0.88 so với 0.85; PR-AUC 0.72 so với 0.66), khớp với EDA:
  quan hệ giữa Age, NumOfProducts và Exited là phi tuyến.
""",

"TUNE": """
**Nhận xét tinh chỉnh:**
- Khi `class_weight` được đưa vào lưới tìm kiếm, **cả hai mô hình đều chọn `balanced`**. Đây là bằng chứng độc lập (qua 5-fold CV)
  rằng xử lý mất cân bằng có lợi cho F1.
- RandomForest và HistGradientBoosting sau tinh chỉnh gần như ngang nhau (F1 Validation 0.634 so với 0.630, PR-AUC đều 0.718).
  Nhóm chọn **RandomForest** vì F1 cao hơn một chút.
- **Chọn ngưỡng:** ở ngưỡng 0.5 mô hình đạt Recall 0.716 nhưng Precision chỉ 0.569. Ngưỡng **0.61** cho F1 cao nhất (0.646),
  với Precision 0.693 và Recall 0.605, tức giảm đáng kể số lần báo nhầm và chỉ mất một phần Recall.
""",

"CREATIVE": """
**Chú thích hình:** Trên **cùng tập Test (305 khách hàng rời bỏ)**:
- Baseline Logistic Regression chỉ bắt được **113 khách hàng (Recall 0.37, F1 0.493)**.
- Chỉ đổi sang RandomForest (chưa xử lý mất cân bằng): 132 khách hàng (F1 0.562).
- Thêm `class_weight` giúp bắt nhiều nhất, **218 khách hàng (Recall 0.715)**, nhưng số báo nhầm tăng từ 33 lên **182**.
- Dời ngưỡng lên 0.61 (chọn trên Validation): mô hình cuối bắt **183 khách hàng (Recall 0.60)** và chỉ báo nhầm **85**, đạt **F1 cao nhất 0.639**.

**Kết luận:** so với baseline, mô hình cuối **bắt thêm 70 khách hàng rời bỏ (+62%)** và F1 tăng **+0.146**, đổi lại thêm 45 lần báo nhầm.
Hai bước cải tiến có vai trò khác nhau: **class weight tăng Recall, chọn ngưỡng cân lại Precision**.
Ngưỡng cuối cùng nên do bài toán kinh doanh quyết định. Nếu chi phí giữ chân rẻ, có thể dùng ngưỡng 0.5 để bắt 218/305 khách hàng.
""",

"IMPORTANCE": """
**Nhận xét:** Hai feature quan trọng nhất vượt trội hẳn: **NumOfProducts** (xáo trộn làm PR-AUC giảm 0.284) và **Age** (0.243).
Tiếp theo là IsActiveMember (0.062), Geography (0.047) và Balance (0.043).
Gender, CreditScore, HasCrCard, Tenure, EstimatedSalary gần như không đóng góp (≤ 0.004).
Kết quả **khớp với EDA**: đây chính là các feature có chênh lệch tỉ lệ rời bỏ lớn nhất ở mục 1.9.
""",

"ERROR": """
**Nhận xét phân tích lỗi:** trên Test có 122 khách hàng rời bỏ bị bỏ sót (FN) và 85 lần báo nhầm (FP).
- **Mô hình yếu nhất ở khách hàng trẻ:** Recall chỉ **0.13 ở nhóm < 30 tuổi** và 0.34 ở nhóm 30–39,
  so với **0.77–0.78 ở nhóm 40–59**. Khách hàng trẻ rời bỏ ít (5.6% ở nhóm < 30), nên mô hình gần như luôn đoán họ ở lại.
- **Chân dung khách hàng bị bỏ sót:** trẻ hơn (tuổi trung bình 41 so với 49 ở nhóm bắt đúng) và **hoạt động nhiều hơn**
  (47% là thành viên tích cực so với 26%), tức là "trông giống" khách hàng trung thành.
  Các ca bị bỏ sót với xác suất thấp nhất đều là khách hàng Pháp, 2 sản phẩm, đúng nhóm có tỉ lệ rời bỏ thấp nhất trong EDA.
- **Hệ quả:** với các feature hiện có, những khách hàng này gần như không thể phân biệt. Cần thêm dữ liệu hành vi
  (giao dịch gần đây, khiếu nại, tương tác với ngân hàng) mới bắt được.
""",

"CONCLUSION": """
**Kết luận**
- Mô hình cuối là **RandomForest + `class_weight='balanced'` + ngưỡng 0.61**. Trên Test đạt **F1 = 0.639, Recall = 0.60, Precision = 0.68,
  ROC-AUC = 0.863**, Accuracy 0.862.
- So với baseline Logistic Regression (F1 0.493), mô hình bắt thêm **70/305 khách hàng rời bỏ**.
- F1 trên Validation (0.646) và Test (0.639) gần nhau, cho thấy mô hình **không overfit** vào Validation khi chọn ngưỡng.
- Yếu tố quyết định nhất là **số sản phẩm** và **tuổi**, sau đó là mức độ hoạt động và quốc gia.
  Gợi ý cho ngân hàng: ưu tiên chăm sóc khách hàng 40–60 tuổi, ở Đức, không hoạt động, đang dùng 1 hoặc ≥ 3 sản phẩm.

**Hạn chế**
- **Missing values do nhóm tự tạo (MCAR)**, không phản ánh cơ chế thiếu dữ liệu thực tế (thường là MAR/MNAR).
- Bộ dữ liệu phổ biến, chỉ có ảnh chụp tĩnh tại một thời điểm, không có hành vi theo thời gian. Mô hình bỏ sót nhiều khách hàng trẻ.
- Ngưỡng tối ưu theo F1 chưa tính chi phí thật của việc bỏ sót so với báo nhầm.

**Hướng phát triển**
- Chọn ngưỡng theo **chi phí kinh doanh** (ví dụ chi phí ưu đãi giữ chân so với doanh thu mất đi) thay vì F1.
- Thử SMOTE, hoặc cân chỉnh xác suất (calibration) để dùng xác suất như điểm rủi ro.
- Bổ sung dữ liệu hành vi/giao dịch theo thời gian; dùng SHAP để giải thích từng dự đoán cho nhân viên chăm sóc khách hàng.
""",
}
