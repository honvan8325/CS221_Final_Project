# GLiNER NER Evaluation & Demo

Đồ án môn **CS212 - XỬ LÝ NGÔN NGỮ TỰ NHIÊN**

## Thông tin

**Giảng viên hướng dẫn:** TS. Nguyễn Thị Quý

**Sinh viên thực hiện:**

| Họ và tên         | MSSV     | Email                  |
| ----------------- | -------- | ---------------------- |
| Nguyễn Thanh Tùng | 23521745 | 23521745@gm.uit.edu.vn |
| Mai Lê Bá Vương   | 23521821 | 23521821@gm.uit.edu.vn |
| Chướng Hồng Văn   | 23521769 | 23521769@gm.uit.edu.vn |

## Giới thiệu

Repository này chứa code thực nghiệm đánh giá mô hình GLiNER (Generalist and Lightweight Named Entity Recognition) trên nhiều benchmark NER khác nhau và ứng dụng demo web tương tác.

**Yêu cầu hệ thống:** Python 3.8+

## Clone Repository

```bash
git clone https://github.com/honvan8325/CS221_Final_Project
cd CS221_Final_Project
```

---

## 1. Đánh giá trên CrossNER Dataset

Đánh giá mô hình GLiNER trên benchmark CrossNER gồm 5 domains: AI, Literature, Music, Politics, Science.

### 1.1. Cài đặt thư viện

```bash
pip install gliner torch pandas tqdm numpy
```

### 1.2. Chuẩn bị dữ liệu

Dữ liệu CrossNER cần đặt tại: `data/crossner`

Hoặc cập nhật biến `CROSSNER_ROOT` trong file `scripts/test_crossner.py` nếu đặt ở vị trí khác.

### 1.3. Chạy thực nghiệm của nhóm

```bash
cd scripts
python test_crossner.py --model l --subset all
```

Lệnh này sẽ chạy đánh giá với:

-   Model: Large (GLiNER-L)
-   Domains: Tất cả (ai, literature, music, politics, science)

### 1.4. Các biến thể khác

**Chạy cơ bản (mô hình large, tất cả subsets):**

```bash
cd scripts
python test_crossner.py
```

**Chọn kích thước mô hình:**

```bash
# Mô hình small
python test_crossner.py --model s

# Mô hình medium
python test_crossner.py --model m

# Mô hình large (mặc định)
python test_crossner.py --model l
```

**Chọn domains cụ thể:**

```bash
# Một domain
python test_crossner.py --subset ai

# Nhiều domains
python test_crossner.py --subset ai music politics

# Tất cả domains
python test_crossner.py --subset all
```

**Lưu kết quả ra CSV:**

```bash
python test_crossner.py --save-csv
```

**Kết hợp các tùy chọn:**

```bash
python test_crossner.py --model m --subset ai literature --save-csv
```

### 1.5. Domains có sẵn

-   `ai` - AI domain
-   `literature` - Literature domain
-   `music` - Music domain
-   `politics` - Politics domain
-   `science` - Science domain

### 1.6. Kết quả

Kết quả sẽ hiển thị dạng:

```
======================================================================
RESULTS
======================================================================
   Model   Ai  Literature  Music  Politics  Science  Average
GLiNER-L 85.2        82.7   79.5      88.1     84.3     84.0
======================================================================
```

File CSV (nếu dùng `--save-csv`): `crossner_results_{model_size}.csv`

---

## 2. Đánh giá trên 7 NER Benchmarks

Đánh giá mô hình GLiNER trên 7 datasets NER khác nhau bao gồm WikiANN, biomedical datasets, và social media datasets.

### 2.1. Cài đặt thư viện

```bash
pip install gliner torch pandas tqdm numpy datasets requests
```

### 2.2. Chuẩn bị dữ liệu

Datasets sẽ **tự động tải** từ Hugging Face Hub khi chạy lần đầu.  
**Lưu ý:** Cần kết nối internet cho lần chạy đầu tiên.

### 2.3. Chạy thực nghiệm của nhóm

```bash
cd scripts
python test_20nerbenchmark.py --model l --dataset all
```

Lệnh này sẽ chạy đánh giá với:

-   Model: Large (GLiNER-L)
-   Datasets: Tất cả benchmarks

### 2.4. Các biến thể khác

**Chạy cơ bản (mô hình large, tất cả datasets):**

```bash
cd scripts
python test_20nerbenchmark.py
```

**Chọn kích thước mô hình:**

```bash
# Mô hình small
python test_20nerbenchmark.py --model s

# Mô hình medium
python test_20nerbenchmark.py --model m

# Mô hình large (mặc định)
python test_20nerbenchmark.py --model l
```

**Chọn datasets cụ thể:**

```bash
# Một dataset
python test_20nerbenchmark.py --dataset WikiANN

# Nhiều datasets
python test_20nerbenchmark.py --dataset WikiANN BC5CDR NCBI

# Tất cả datasets
python test_20nerbenchmark.py --dataset all
```

**Lưu kết quả ra CSV:**

```bash
python test_20nerbenchmark.py --save-csv
```

**Kết hợp các tùy chọn:**

```bash
python test_20nerbenchmark.py --model s --dataset WikiANN BC5CDR GENIA --save-csv
```

### 2.5. Datasets có sẵn

-   `WikiANN` - Wikipedia-based NER
-   `BC5CDR` - BioCreative V Chemical-Disease Relations
-   `NCBI` - NCBI Disease corpus
-   `GENIA` - GENIA corpus
-   `BC2GM` - BioCreative II Gene Mention
-   `TweetNER7` - Twitter NER (7 entity types)
-   `HarveyNER` - Hurricane Harvey Twitter NER

### 2.6. Kết quả

Kết quả sẽ hiển thị dạng:

```
============================================================
EVALUATION RESULTS
============================================================

Dataset              F1 Score     Labels
------------------------------------------------------------
WikiANN               85.30%       3 types
BC5CDR                87.20%       2 types
NCBI                  82.10%       1 types
GENIA                 79.40%       5 types
BC2GM                 84.60%       1 types
TweetNER7             76.80%       7 types
HarveyNER             71.50%       4 types
```

File CSV (nếu dùng `--save-csv`): `7ner_results_{model_size}.csv`

---

## 3. Web Demo

Giao diện web tương tác để test mô hình GLiNER trên văn bản tùy chỉnh.

### 3.1. Cài đặt thư viện

```bash
cd demo
pip install -r requirements.txt
```

Hoặc:

```bash
pip install gradio gliner torch
```

### 3.2. Chạy demo

**Chạy cơ bản:**

```bash
cd demo
python app.py
```

Demo sẽ:

-   Khởi động server tại `http://localhost:7860`
-   Tự động mở trong trình duyệt mặc định

**Chỉ định cổng:**

```bash
python app.py --port 8080
```

**Chia sẻ công khai (qua Gradio):**

```bash
python app.py --share
```

**Kết hợp các tùy chọn:**

```bash
python app.py --port 8080 --share
```

### 3.3. Sử dụng demo

1. Mở web interface trong trình duyệt
2. Chọn kích thước model (Small, Medium, hoặc Large)
3. Nhập hoặc dán văn bản cần phân tích
4. Chỉ định các loại entity cần trích xuất (vd: "person, organization, location")
5. Click "Extract Entities" để xem kết quả
6. Kết quả sẽ hiển thị với entities được highlight
