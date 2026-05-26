# Báo Cáo Đánh Giá Dự Án Intro to Data Science

Chào nhóm của bạn. Tôi đã chạy thử nghiệm toàn bộ hệ thống code của các bạn (từ cào dữ liệu, làm sạch, tiền xử lý, trực quan hóa đến mô hình hóa giám sát và phân cụm) và đối chiếu với đề bài cũng như bản thiết kế pipeline của nhóm.

Dưới đây là phần đánh giá chi tiết và các khuyến nghị tối ưu giúp dự án đạt điểm tối đa (mức "Excellent").

---

## 1. Trả Lời Câu Hỏi Của Nhóm Về Cột `direction`

> **Câu hỏi của nhóm:** Dữ liệu thô của nhóm hiện tại đã mã hóa sẵn cột `direction` thành `0` và `1` chưa, hay vẫn đang ở dạng chuỗi văn bản (String) cần dùng thư viện Pandas để chuyển đổi?

**Trả lời:**
- Dữ liệu thô trong file `VINH_TUY.csv` của các bạn **vẫn đang ở dạng chuỗi văn bản (String)** với các giá trị cụ thể là `'Inbound'` và `'Outbound'`.
- Cách các bạn xử lý trong file `Data_Processing.py` (dòng 26):
  ```python
  df['direction_inbound'] = df['direction'].apply(lambda x: 1 if 'inbound' in str(x).lower() else 0)
  ```
  **là hoàn toàn chính xác**. Nó sử dụng hàm lambda kết hợp xử lý chuỗi (chuyển thường) để tạo cột nhị phân `direction_inbound` (`1` cho Inbound và `0` cho Outbound). Nhóm không cần thay đổi phần logic này.

---

## 2. Các Lỗi Nghiêm Trọng Cần Khắc Phục Ngay (Critical Bugs)

Tôi phát hiện **02 lỗi nghiêm trọng** trong code gốc và đã sửa trực tiếp giúp nhóm:

### 🔴 Lỗi 1: Sai lệch đường dẫn gốc (`BASE_DIR`) khiến `Data_Cleaning.py` bị lỗi Runtime
- **Vị trí lỗi:** File [Data_Cleaning.py](file:///d:/Datumi/FinalDataScience-1/Code/DataCleaning/Data_Cleaning.py#L6)
- **Chi tiết:** Các bạn định nghĩa:
  ```python
  BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  ```
  Do file này nằm ở thư mục `Code/DataCleaning/Data_Cleaning.py`, nên khi gọi `dirname` hai lần, `BASE_DIR` sẽ trỏ về thư mục `Code/`. Kết quả là khi load dữ liệu thô:
  ```python
  DATA_PATH = os.path.join(BASE_DIR, "Raw Datasets", "VINH_TUY.csv")
  ```
  Python sẽ tìm ở `Code/Raw Datasets/VINH_TUY.csv` và báo lỗi **`FileNotFoundError`** vì thư mục `Raw Datasets` thực tế nằm ở thư mục gốc của project (ngoài thư mục `Code`).
- **Cách khắc phục (Đã cập nhật):** Đi lên 3 cấp thư mục để về đúng thư mục gốc của project:
  ```python
  BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  ```

### 🔴 Lỗi 2: Rò rỉ dữ liệu (Data Leakage) khi chuẩn hóa đặc trưng
- **Vị trí lỗi:**
  - File [log_reg_model.py](file:///d:/Datumi/FinalDataScience-1/Code/SupervisedLearning/log_reg_model.py#L43-L50)
  - File [svm_model.py](file:///d:/Datumi/FinalDataScience-1/Code/SupervisedLearning/svm_model.py#L38-L45)
- **Chi tiết:** Trong cả hai mô hình, trước đó các bạn đang thực hiện fit và transform `StandardScaler` trên **toàn bộ dữ liệu `X` trước khi chia tách Train/Test** (`fit_transform(X)`). Việc tính toán giá trị trung bình (`mean`) và độ lệch chuẩn (`std`) của tập Test rồi áp dụng cho tập Train làm rò rỉ thông tin từ tập Test vào quá trình huấn luyện, khiến kết quả đánh giá không còn khách quan.
- **Cách khắc phục (Đã cập nhật):** Chia dữ liệu thành Train và Test **trước**, sau đó chỉ dùng tập Train để học tham số chuẩn hóa (`fit_transform`) và dùng các tham số đó để biến đổi (`transform`) cho tập Test:
  ```python
  X_train_raw, X_test_raw, y_train, y_test = train_test_split(
      X, y, test_size=0.5, random_state=42, stratify=y
  )

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train_raw)
  X_test = scaler.transform(X_test_raw)
  ```

---

## 3. Đóng Góp Chuyên Môn Quan Trọng: Cải Tiến K-Means Clustering

Trong file [k_mean.py](file:///d:/Datumi/FinalDataScience-1/Code/UnsupervisedLearning/k_mean.py), lúc đầu nhóm thực hiện giảm chiều bằng PCA xuống 2 thành phần chính rồi mới gom cụm K-Means trên không gian PCA đó.

> [!WARNING]
> **PCA giải thích quá ít phương sai (Variance Explained quá thấp):**
> Tổng lượng phương sai được giải thích bởi 2 PC đầu chỉ đạt **45.46%**.
> Điều này có nghĩa là **54.54%** thông tin trong 6 đặc trưng gốc đã bị loại bỏ hoàn toàn trước khi K-Means phân cụm. 

### Minh chứng thực nghiệm (Sự khác biệt)
Tôi đã điều chỉnh thuật toán chạy trực tiếp K-Means trên 6 đặc trưng ban đầu đã chuẩn hóa (retaining 100% variance), chỉ dùng PCA để vẽ biểu đồ trực quan hóa 2D. Hãy xem sự vượt trội qua bảng đặc trưng hướng đi (`direction_inbound`):

*   **Kết quả cách cũ (K-Means trên không gian PCA):** Tất cả các cụm cao điểm (Rush Hour) đều có `direction_inbound` xấp xỉ **`0.50`** (Không phân tách được hướng Inbound và Outbound).
*   **Kết quả cách cải tiến (K-Means trực tiếp trên đặc trưng gốc):** Tách bạch hoàn hảo bối cảnh thực tế:
    *   **Cluster 1:** Giờ cao điểm, **chỉ đi hướng vào trung tâm (Inbound = 1.0)**, thời tiết đẹp. Tỷ lệ tắc đường cao nhất (**57.32%**).
    *   **Cluster 5:** Giờ cao điểm, **chỉ đi hướng ra ngoài (Inbound = 0.0)**, thời tiết đẹp. Tỷ lệ tắc đường (**49.57%**).
    *   **Cluster 2:** Giờ cao điểm kèm thời tiết xấu, xảy ra trên cả 2 hướng (Inbound = 0.5), tỷ lệ tắc đường (**52.67%**).

### Tại sao biểu đồ PCA lại hiển thị các cụm bị chồng lấn (trộn lẫn)?
- **Bản chất của PCA là tuyến tính:** PCA chiếu không gian 6 chiều về 2 chiều. Vì 2 PC chỉ giải thích được 45.46% phương sai, nên 54% thông tin ở các chiều khác bị ép phẳng về 0, làm cho các cụm thực chất phân tách rõ ở chiều thứ 3, 4, 5, 6 trông giống như chồng lên nhau trên đồ thị PCA 2 chiều.
- **Chứng minh thực tế:** Nhìn vào bảng Profiling ở trên, các cụm có thông số đặc trưng rất khác biệt (ví dụ cụm 1 có `direction_inbound` = 1.0 và cụm 5 có `direction_inbound` = 0.0). Điều này khẳng định trong không gian 6 chiều thực tế, các cụm **hoàn toàn tách biệt** và không hề trộn lẫn!

---

## 4. Kết Quả Chạy Mô Hình Sau Khi Khắc Phục Lỗi Data Leakage

Tỷ lệ chia Train/Test 50/50 như Rubric yêu cầu:

### Logistic Regression
- **Testing Accuracy:** 80.25%
- **Testing F1-Score:** 54.56%
- **Testing Precision:** 59.29%
- **Testing Recall:** 50.52%

### Support Vector Machine (Linear SVM)
- **Testing Accuracy:** 80.96%
- **Testing F1-Score:** 60.82%
- **Testing Precision:** 58.80%
- **Testing Recall:** 62.98%

> [!NOTE]
> **Nhận xét cho phần phân tích báo cáo:**
> 1. Cả hai mô hình đều không bị Overfitting (độ chính xác tập Test tương đồng tập Train).
> 2. Mô hình **SVM cho hiệu năng vượt trội hơn** về mặt F1-Score (60.82% so với 54.56% của Logistic Regression) nhờ độ bao phủ (Recall) tốt hơn đáng kể (62.98% so với 50.52%), giúp nhận diện được nhiều trường hợp ùn tắc thực tế hơn.

### 📊 Các biểu đồ trực quan quan trọng đã được tạo:
Tôi đã tạo và chạy một tập lệnh hỗ trợ [visualize_supervised.py](file:///d:/Datumi/FinalDataScience-1/Code/SupervisedLearning/visualize_supervised.py) để tự động xuất ra **03 biểu đồ trực quan hóa chuyên nghiệp** trong thư mục `Data Visualization/Task 2`:
*   **`Task 2/8_confusion_matrices.png` (Ma trận nhầm lẫn của cả hai mô hình đặt cạnh nhau):** Giúp người đọc dễ dàng thấy được chính xác số lượng ca dự đoán đúng (True Positive, True Negative) và dự đoán sai (False Positive, False Negative) trên tập Test.
*   **`Task 2/9_roc_curves.png` (So sánh đường cong ROC và chỉ số AUC):** Biểu diễn trực quan tỷ lệ True Positive Rate so với False Positive Rate ở các ngưỡng khác nhau. Chỉ số AUC của SVM cao hơn cho thấy mô hình này có khả năng phân biệt tắc đường tốt hơn.
*   **`Task 2/10_feature_coefficients.png` (So sánh trọng số hệ số đặc trưng - Feature Importance):** Đây là biểu đồ **quan trọng nhất** dùng để trả lời câu hỏi *"Tại sao mô hình dự đoán như vậy và yếu tố nào ảnh hưởng nhiều nhất?"*. Nó biểu diễn trọng số của 6 đặc trưng. Ví dụ, đặc trưng *Rush Hour Period* và *Rush x Weather Interaction* có trọng số dương lớn nhất, chứng tỏ đây là hai tác nhân mạnh nhất thúc đẩy ùn tắc giao thông.

---

## 5. Trạng Thế Code Hiện Tại (Đã Cập Nhật)

Tôi đã cập nhật trực tiếp toàn bộ các bản sửa lỗi và tối ưu vào file gốc của các bạn:
1. Sửa lỗi `BASE_DIR` đường dẫn trong [Data_Cleaning.py](file:///d:/Datumi/FinalDataScience-1/Code/DataCleaning/Data_Cleaning.py).
2. Sửa lỗi chuẩn hóa rò rỉ dữ liệu trong [log_reg_model.py](file:///d:/Datumi/FinalDataScience-1/Code/SupervisedLearning/log_reg_model.py) và [svm_model.py](file:///d:/Datumi/FinalDataScience-1/Code/SupervisedLearning/svm_model.py).
3. Sửa cấu trúc gom cụm phân tích đặc trưng trực tiếp trên 6 chiều và xuất biểu đồ PCA (`6_pca_clusters_direct.png`), biểu đồ Elbow (`5_elbow_silhouette.png`) cùng biểu đồ nhiệt thuộc tính các cụm (`7_cluster_profiles.png`) trong thư mục `Data Visualization/Task 3` thông qua file [k_mean.py](file:///d:/Datumi/FinalDataScience-1/Code/UnsupervisedLearning/k_mean.py).
