# Bài gốc: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0353166

# Đánh giá các mô hình ngôn ngữ tiền huấn luyện gần đây cho nhận diện thực thể có tên trong văn bản hành chính tiếng Việt

## 1. Bối cảnh nghiên cứu

Nhận diện thực thể có tên (NER) là một nhiệm vụ quan trọng trong Xử lý ngôn ngữ tự nhiên (NLP) nhằm nhận diện và phân loại các thực thể có tên trong văn bản.

Các tài liệu hành chính tiếng Việt bao gồm những thực thể đặc thù như cơ quan nhà nước, văn bản pháp luật, tổ chức, ngày tháng và số lượng. Những đặc trưng này khiến NER hành chính tiếng Việt khác với NER trong miền dữ liệu phổ thông.

Gần đây, La và cộng sự đã giới thiệu PAP_NER, một bộ corpora NER hành chính tiếng Việt quy mô lớn, đồng thời đề xuất kiến trúc PhoBERT-CRF lai. Cách tiếp cận này đạt Micro-F1 97.95% trên chuẩn đánh giá PAP_NER.

Tuy nhiên, các mô hình ngôn ngữ tiền huấn luyện liên tục phát triển mạnh mẽ. Do đó, cần thiết đánh giá xem những mô hình này có thể cải thiện hiệu quả so với PhoBERT-CRF trên văn bản hành chính tiếng Việt hay không.

---

# 2. Vấn đề nghiên cứu

Nghiên cứu gốc về PAP_NER đã đưa ra PhoBERT-CRF như một chuẩn mạnh cho NER hành chính tiếng Việt.

Tuy nhiên, hiệu quả của các mô hình ngôn ngữ tiền huấn luyện gần đây trên tập dữ liệu PAP_NER vẫn chưa được nghiên cứu đầy đủ.

Vì vậy, nghiên cứu này tập trung đánh giá, so sánh các mô hình ngôn ngữ tiền huấn luyện mới trong cùng điều kiện thực nghiệm với nghiên cứu gốc PAP_NER.

Vấn đề nghiên cứu chính là:

> Liệu các mô hình ngôn ngữ tiền huấn luyện gần đây có thể cải thiện hiệu quả NER hành chính tiếng Việt so với chuẩn PhoBERT-CRF trên tập dữ liệu PAP_NER hay không?

---

# 3. Mục tiêu nghiên cứu

## 3.1. Mục tiêu chính

Đánh giá hiệu quả của các mô hình ngôn ngữ tiền huấn luyện gần đây cho nhiệm vụ nhận diện thực thể có tên trong văn bản hành chính tiếng Việt trên chuẩn PAP_NER.

## 3.2. Mục tiêu cụ thể

1. Tái hiện kết quả chuẩn PhoBERT-CRF được báo cáo trong nghiên cứu gốc PAP_NER.
2. Lựa chọn một số mô hình ngôn ngữ tiền huấn luyện gần đây phù hợp với tiếng Việt.
3. Tinh chỉnh các mô hình đã chọn trên tập PAP_NER.
4. Tích hợp các mô hình ngôn ngữ tiền huấn luyện này với lớp CRF.
5. So sánh hiệu năng của các mô hình với PhoBERT-CRF gốc.
6. Phân tích hiệu năng trên từng loại thực thể hành chính.
7. Phân tích lỗi để nhận diện điểm mạnh, điểm yếu của từng mô hình.
8. So sánh hiệu quả và chi phí tính toán giữa các mô hình.

---

# 4. Câu hỏi nghiên cứu

## RQ1

> Liệu các mô hình ngôn ngữ tiền huấn luyện gần đây có thể vượt qua chuẩn PhoBERT-CRF trên benchmark PAP_NER?

## RQ2

> Việc lựa chọn mô hình ngôn ngữ tiền huấn luyện ảnh hưởng như thế nào tới hiệu quả NER hành chính tiếng Việt trên từng loại thực thể?

## RQ3

> Những loại lỗi nào giảm hoặc mới xuất hiện khi thay PhoBERT bằng mô hình ngôn ngữ tiền huấn luyện gần đây?

---

# 5. Giả thuyết nghiên cứu

## H0

> Các mô hình ngôn ngữ tiền huấn luyện gần đây không cải thiện đáng kể hiệu quả NER so với PhoBERT-CRF trên PAP_NER.

## H1

> Ít nhất một mô hình ngôn ngữ tiền huấn luyện gần đây cải thiện rõ rệt hiệu quả NER so với PhoBERT-CRF trên PAP_NER.

---

# 6. Bộ dữ liệu

## PAP_NER

Sử dụng bộ dữ liệu PAP_NER như trong nghiên cứu gốc.

Đặc trưng dữ liệu:

- Văn bản hành chính tiếng Việt
- 162.801 câu
- 205.807 thực thể được gán nhãn
- 5 loại thực thể

Các loại thực thể:

| Nhãn | Mô tả |
|---|---|
| CQ | Cơ quan hành chính / Nhà nước |
| VBPL | Văn bản pháp luật |
| ĐT | Đối tượng |
| NG | Ngày/Thời gian |
| SL | Số lượng |

Quy trình chia tập và xử lý dữ liệu phải tuân theo nghiên cứu gốc PAP_NER càng sát càng tốt.

---

# 7. Chuẩn so sánh

## Chuẩn gốc

Chuẩn so sánh chính:

> PhoBERT + CRF

Kết quả báo cáo trong nghiên cứu gốc:

> Micro-F1 = 97.95%

Thí nghiệm đầu phải nỗ lực tái lặp kết quả này sát nhất có thể.

### Thí nghiệm chuẩn

```text
PAP_NER
   |
   v
PhoBERT
   |
   v
CRF
   |
   v
Dự đoán nhãn BIO/BIOES
   |
   v
Đánh giá Precision / Recall / Micro-F1
```

# 8. Đề xuất đánh giá

Nghiên cứu này không xây dựng kiến trúc NER mới.

Thay vào đó, nghiên cứu thay đổi mô hình ngôn ngữ tiền huấn luyện nhưng giữ nguyên kiến trúc và quy trình đánh giá phía sau.

```text
                    PAP_NER
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
    PhoBERT          Model A         Model B
        |              |              |
        v              v              v
       CRF            CRF            CRF
        |              |              |
        +--------------+--------------+
                       |
                    Evaluation
```

Biến độc lập chính là:

> **Mô hình ngôn ngữ tiền huấn luyện**

Lớp CRF, bộ dữ liệu, nhiệm vụ và chỉ số đánh giá cần được giữ nguyên tối đa.

# 9. Lựa chọn mô hình

Chọn 2–3 mô hình ngôn ngữ tiền huấn luyện gần đây.

## Tiêu chí lựa chọn

1. Hỗ trợ tiếng Việt hoặc đa ngôn ngữ có tiếng Việt.
2. Được phát hành sau PhoBERT.
3. Có trọng số tiền huấn luyện công khai.
4. Có tài liệu hoặc nguồn mở đầy đủ.
5. Có thể tinh chỉnh trong điều kiện GPU hiện có.
6. Thể hiện tốt trên các nhiệm vụ NLP tiếng Việt.
7. Chưa được đánh giá rộng rãi trên PAP_NER.

Việc chọn mô hình sẽ quyết định sau khi khảo cứu các công trình liên quan từ năm 2024–2026.

# 10. Thiết lập thực nghiệm

Giữ điều kiện thực nghiệm nhất quán giữa các mô hình.

## 10.1. Phần cứng

Ghi lại:

- GPU
- VRAM
- CPU
- RAM
- Ổ lưu trữ

## 10.2. Phần mềm

Ghi lại:

- Phiên bản Python
- Phiên bản PyTorch
- Phiên bản Transformers
- Phiên bản CUDA
- Hệ điều hành

## 10.3. Cấu hình huấn luyện

Cố gắng giữ cố định các tham số:

- Chia train / validation / test
- Batch size
- Số epoch
- Learning rate
- Độ dài chuỗi tối đa
- Optimizer
- Weight decay
- Random seed

Nếu phải thay đổi, cần ghi chú rõ ràng.

# 11. Chỉ số đánh giá

Chỉ số chính:

> **Micro-F1**

Báo cáo thêm:

- Precision
- Recall
- F1-score

Đánh giá cả hiệu suất tổng thể và theo từng loại thực thể.

## 11.1. Tổng thể

```text
Precision
Recall
Micro-F1
```

## 11.2. Theo loại thực thể

```text
CQ
VBPL
ĐT
NG
SL
```

Nhờ vậy, đánh giá được mô hình cải thiện tổng thể hay chỉ một số loại thực thể nhất định.

# 12. Thí nghiệm chính

Chạy các thí nghiệm sau:

| Thí nghiệm | Mô hình | CRF |
|---|---|---|
| Chuẩn | PhoBERT | Có |
| Thí nghiệm 1 | Model A | Có |
| Thí nghiệm 2 | Model B | Có |
| Thí nghiệm 3 | Model C | Có |

Mục tiêu chính là xác định việc thay PhoBERT bằng mô hình ngôn ngữ tiền huấn luyện mới có cải thiện benchmark không.

Kết quả baseline PAP_NER:

> **PhoBERT + CRF: 97.95 Micro-F1**

# 13. Nghiên cứu cắt lớp (ablation)

Chỉ thực hiện ở quy mô nhỏ nếu đủ tài nguyên.

Ví dụ:

| Mô hình | CRF | F1 |
|---|---:|---:|
| PhoBERT | Không | ... |
| PhoBERT | Có | 97.95 |
| Model A | Không | ... |
| Model A | Có | ... |

Mục đích là xác định cải thiện đến từ chính mô hình tiền huấn luyện hay lớp CRF.

Nghiên cứu cắt lớp này chỉ nên làm nhỏ do giới hạn thời gian.

# 14. Phân tích theo loại thực thể

So sánh các mô hình theo 5 loại thực thể.

| Loại thực thể | PhoBERT-CRF | Model A-CRF | Model B-CRF |
|---|---:|---:|---:|
| CQ | ... | ... | ... |
| VBPL | ... | ... | ... |
| ĐT | ... | ... | ... |
| NG | ... | ... | ... |
| SL | ... | ... | ... |

Phân tích:

- Loại nào dễ nhất?
- Loại nào khó nhất?
- Mô hình nào tốt nhất với mỗi loại?
- Cải thiện Micro-F1 tổng thể chủ yếu nhờ loại thực thể nào?

Đây là phân tích quan trọng vì sự tăng tổng thể đôi khi do một loại thực thể cụ thể.

# 15. Phân tích lỗi

Phân tích lỗi định tính trên các ví dụ đại diện.

Tập trung vào:

## 15.1. Lỗi biên thực thể

Ví dụ:

```text
Mong muốn:
[Ủy ban nhân dân xã Đồng Văn]

Dự đoán:
[Ủy ban nhân dân]
```

Đánh giá liệu các mô hình mới nhận diện biên thực thể tốt hơn không.

## 15.2. Lỗi loại thực thể

Ví dụ:

```text
Mong muốn:
[Công văn số 123] → VBPL

Dự đoán:
[Công văn số 123] → CQ
```

Mô hình nhận diện đúng vùng nhưng gán nhãn sai loại thực thể.

## 15.3. Lỗi thực thể dài

Xem xét liệu mô hình gặp khó khăn với những thực thể hành chính có độ dài lớn.

## 15.4. Thực thể mơ hồ

Một số từ hoặc cụm từ có thể mang nghĩa khác nhau trong từng ngữ cảnh.

Ví dụ, một thuật ngữ hành chính có thể là tổ chức ở câu này nhưng lại thuộc loại khác ở câu khác.

## 15.5. Thực thể hiếm

Phân tích lỗi với các thực thể có tần suất xuất hiện thấp trong tập huấn luyện.

Mục tiêu không chỉ là xác định mô hình nào có F1-score cao nhất mà còn là lý giải tại sao chúng khác biệt.

# 16. Hiệu quả tính toán

Chỉ số chính không chỉ là độ chính xác.

Theo dõi:

- Số lượng tham số mô hình
- Kích thước mô hình
- Thời gian huấn luyện
- Thời gian suy diễn (inference)
- Đỉnh bộ nhớ GPU sử dụng

Ví dụ:

| Mô hình | Micro-F1 | Số tham số | GPU Memory | Thời gian inference |
|---|---:|---:|---:|---:|
| PhoBERT-CRF | 97.95 | ... | ... | ... |
| Model A-CRF | ... | ... | ... | ... |
| Model B-CRF | ... | ... | ... | ... |

Kết hợp chỉ số hiệu quả – độ chính xác này sẽ cho cái nhìn toàn diện.

Ví dụ, nếu Model A chỉ tăng F1 0.1% nhưng yêu cầu tài nguyên tính toán cao hơn nhiều, ưu thế thực tế có thể thấp.

# 17. Phân tích thống kê

Nếu đủ tài nguyên, chạy mỗi thí nghiệm với nhiều random seed.

Ví dụ:

```text
Seed 42
Seed 123
Seed 2024
```

Báo cáo:

```text
Mean ± Độ lệch chuẩn
```

Ví dụ:

```text
PhoBERT-CRF: 97.95 ± 0.05
Model A-CRF: 98.12 ± 0.08
```

Phân tích này giúp củng cố rằng khác biệt không phải do ngẫu nhiên.

Nếu khác biệt đủ lớn, có thể thực hiện kiểm định ý nghĩa thống kê.

# 18. Đóng góp kỳ vọng của nghiên cứu

Đóng góp chính mang tính thực nghiệm, không mang tính kiến trúc.

## Đóng góp 1: Đánh giá benchmark

Tạo benchmark hệ thống giữa các mô hình tiền huấn luyện gần đây cho NER hành chính tiếng Việt.

## Đóng góp 2: So sánh tái lập

So sánh các mô hình mới với chuẩn PhoBERT-CRF của PAP_NER trong điều kiện thực nghiệm thống nhất.

## Đóng góp 3: Phân tích theo loại thực thể

Phân tích hiệu quả mô hình với từng loại thực thể hành chính:

```text
CQ
VBPL
ĐT
NG
SL
```

## Đóng góp 4: Phân tích lỗi

Nhận diện những kiểu lỗi chủ đạo của các mô hình tiền huấn luyện khác nhau.

## Đóng góp 5: So sánh độ chính xác & chi phí

Bao gồm cả độ chính xác và chi phí tính toán.

Kết luận có thể tóm lại là:

> **Một đánh giá thực nghiệm và phân tích các mô hình ngôn ngữ tiền huấn luyện gần đây cho nhận diện thực thể có tên trong văn bản hành chính tiếng Việt.**

Nghiên cứu không đề xuất kiến trúc NER mới.

# 19. Kết quả kỳ vọng

Hai kịch bản chính có thể xảy ra.

## Trường hợp 1: Mô hình mới vượt PhoBERT

Ví dụ:

```text
PhoBERT-CRF: 97.95
Model A-CRF: 98.20
```

Nghiên cứu có thể kết luận mô hình mới cung cấp biểu diễn tốt hơn cho văn bản hành chính tiếng Việt và cải thiện NER.

Cần làm rõ hơn:

- Loại thực thể nào cải thiện?
- Lỗi nào giảm?
- Có ý nghĩa thống kê không?
- Có xứng đáng với chi phí tính toán tăng lên không?

## Trường hợp 2: Mô hình mới không vượt PhoBERT

Ví dụ:

```text
PhoBERT-CRF: 97.95
Model A-CRF: 97.80
```

Điều này không làm cho nghiên cứu mất giá trị.

Thay vào đó, cần điều tra lý do:

- Chênh lệch miền dữ liệu huấn luyện
- Lợi thế từ tiền huấn luyện chuyên biệt cho tiếng Việt
- Quy mô mô hình
- Khác biệt theo loại thực thể
- Xử lý ngữ cảnh dài
- Ổn định quá trình huấn luyện
- Sự khác biệt khi token hóa
- Giới hạn tính toán

Có thể rút ra kết luận quan trọng:

> Không phải mô hình tiền huấn luyện mới hơn hoặc lớn hơn luôn vượt trội mô hình chuyên biệt tiếng Việt trên tác vụ NER hành chính.

Do đó, không nhất thiết nghiên cứu phải đạt SOTA mới.

Mục tiêu chính là:

> **Đánh giá thực nghiệm liệu các mô hình tiền huấn luyện gần đây mang lại ưu thế đo đếm được so với PhoBERT cho NER hành chính tiếng Việt, cũng như phân tích lý do cơ bản giải thích kết quả.**

# 20. Các câu hỏi nghiên cứu

Nghiên cứu có thể đặt ra các câu hỏi sau.

## RQ1

> Mô hình ngôn ngữ tiền huấn luyện mới có vượt PhoBERT-CRF trên PAP_NER không?

## RQ2

> Việc chọn mô hình tiền huấn luyện ảnh hưởng như thế nào đến hiệu quả NER từng loại thực thể hành chính tiếng Việt?

## RQ3

> Những loại lỗi nào giảm hoặc phát sinh khi thay PhoBERT bằng mô hình mới?

## RQ4

> Đánh đổi giữa hiệu quả (F1) và chi phí tính toán của các mô hình ra sao?


# 21. Giả thuyết nghiên cứu

## Giả thuyết gốc (H0)

> Các mô hình ngôn ngữ tiền huấn luyện gần đây không cải thiện có ý nghĩa thống kê so với PhoBERT-CRF trên NER hành chính tiếng Việt.

## Giả thuyết thay thế (H1)

> Ít nhất một mô hình tiền huấn luyện gần đây cải thiện có ý nghĩa thống kê so với PhoBERT-CRF trên NER hành chính tiếng Việt.

Cũng có thể có giả thuyết cụ thể hơn:

> H1: Mô hình tiền huấn luyện gần đây có biểu diễn ngữ cảnh mạnh hơn sẽ đạt Micro-F1 cao hơn PhoBERT-CRF trên PAP_NER.

# 22. Phạm vi nghiên cứu

Để đảm bảo hoàn thành trong 3–4 tuần, phạm vi phải hạn chế hợp lý.

## Bao gồm

- Tập dữ liệu PAP_NER
- NER hành chính tiếng Việt
- Chuẩn PhoBERT-CRF
- 2–3 mô hình ngôn ngữ tiền huấn luyện mới
- Tinh chỉnh tiêu chuẩn
- Micro-F1, Precision, Recall
- Đánh giá theo thực thể
- Phân tích lỗi
- Phân tích hiệu quả tính toán
- Phân tích thống kê (hạn chế)

## Loại trừ

Nghiên cứu không tập trung vào:

- Tạo tập dữ liệu mới
- Gán nhãn thủ công tập mới
- Xây dựng kiến trúc NER mới hoàn toàn
- Tích hợp mô hình truy hồi (RAG)
- Xây dựng knowledge graph
- Tinh chỉnh lớn kiểu instruction-tuning
- Xây dựng framework NLP mới
- Triển khai hệ thống production

Giới hạn này đảm bảo nghiên cứu tập trung vào so sánh thực nghiệm chặt chẽ chứ không đặt nặng tính mới về kiến trúc.

# 23. Quy trình nghiên cứu đề xuất

Quy trình nghiên cứu tổng thể như sau:

```text
              Tổng quan tài liệu
                     |
                     v
                Nghiên cứu PAP_NER
                     |
                     v
            Chuẩn bị tập dữ liệu
                     |
                     v
          Tái hiện PhoBERT-CRF
                     |
                     v
             Xác thực baseline
                     |
                     v
        Chọn mô hình PLM gần đây
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Model A    Model B    Model C
          |          |          |
          v          v          v
        + CRF      + CRF      + CRF
          |          |          |
          +----------+----------+
                     |
                     v
              Đánh giá mô hình
                     |
          +----------+----------+
          |          |          |
          v          v          v
     Tổng thể   Theo thực thể  Hiệu quả
      metrics      metrics      metrics
          |          |          |
          +----------+----------+
                     |
                     v
               Phân tích lỗi
                     |
                     v
           Phân tích thống kê
                     |
                     v
                 Thảo luận
                     |
                     v
                Kết luận
```

# 24. Kế hoạch 4 tuần

Do cả 4 thành viên đều có việc học hay làm, phạm vi thực nghiệm phải được kiểm soát chặt chẽ.

## Tuần 1 — Baseline và Dữ liệu

### Thành viên 1
- Nghiên cứu PAP_NER
- Chuẩn bị dữ liệu
- Hiểu nhãn BIO
- Xây dựng pipeline tiền xử lý

### Thành viên 2
- Tái hiện PhoBERT-CRF
- Xác thực code
- Lấy kết quả baseline

### Thành viên 3
- Khảo sát các PLM tiếng Việt/đa ngữ mới
- Xác định mô hình ứng viên

### Thành viên 4
- Nghiên cứu quy trình đánh giá
- Chuẩn bị code đánh giá
- Xây dựng khung phân tích lỗi

### Kết quả tuần

```text
Dataset đã sẵn sàng
PhoBERT-CRF chạy được
Có baseline
Đã chọn 2–3 PLM mới
```

## Tuần 2 — Thí nghiệm chính

### Thành viên 1
- Hỗ trợ pipeline xử lý dữ liệu

### Thành viên 2
- Huấn luyện Model A

### Thành viên 3
- Huấn luyện Model B

### Thành viên 4
- Huấn luyện Model C hoặc hỗ trợ đánh giá

### Kết quả tuần

```text
PhoBERT-CRF
Model A-CRF
Model B-CRF
Model C-CRF
```

Cùng với kết quả ban đầu.

## Tuần 3 — Phân tích

Thực hiện:

- Chạy training cuối cùng
- Nếu đủ điều kiện, thử nhiều seed
- Tính Precision / Recall / F1
- F1 theo thực thể
- Phân tích lỗi
- Đo hiệu năng tính toán
- Phân tích thống kê

### Kết quả tuần

Hoàn thiện kết quả thực nghiệm.

## Tuần 4 — Viết báo cáo & trình bày

### Công việc

- Giới thiệu
- Tổng quan tài liệu
- Phương pháp
- Thiết lập thực nghiệm
- Kết quả
- Thảo luận
- Phân tích lỗi
- Kết luận
- Hình ảnh
- Bảng biểu
- Slide thuyết trình

### Kết quả cuối

```text
Báo cáo nghiên cứu
Mã nguồn
Kết quả thực nghiệm
Slide thuyết trình
```

# 25. Phân công công việc 4 thành viên

Chia công việc như sau:

| Thành viên | Trách nhiệm chính |
|---|---|
| Thành viên 1 | Dữ liệu + tiền xử lý + baseline |
| Thành viên 2 | Model A + thực nghiệm |
| Thành viên 3 | Model B/C + thực nghiệm |
| Thành viên 4 | Đánh giá + phân tích lỗi + thống kê |

Tuy nhiên, tất cả đều cùng tham gia:

- Tổng quan tài liệu
- Thảo luận kết quả
- Viết báo cáo
- Làm slide

# 26. Thiết kế nghiên cứu cuối cùng

Tóm tắt thiết kế nghiên cứu:

```text
Chủ đề nghiên cứu:
Đánh giá các mô hình ngôn ngữ tiền huấn luyện gần đây
cho nhận diện thực thể có tên trong văn bản hành chính tiếng Việt

                Dữ liệu PAP_NER
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
     PhoBERT        Model A         Model B
        |              |              |
        v              v              v
       CRF            CRF            CRF
        |              |              |
        +--------------+--------------+
                       |
                       v
                  Đánh giá
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
     Tổng thể      Theo thực thể   Hiệu quả
     Metrics        Metrics        Metrics
        |              |              |
        +--------------+--------------+
                       |
                       v
                Phân tích lỗi
                       |
                       v
             Phân tích thống kê
                       |
                       v
                  Kết luận
```

Câu hỏi trọng tâm của nghiên cứu là:

> **Việc thay thế PhoBERT bằng một mô hình ngôn ngữ tiền huấn luyện gần đây có giúp cải thiện NER hành chính tiếng Việt khi giữ nguyên kiến trúc CRF và điều kiện đánh giá không?**

Nguyên tắc phương pháp luận cốt lõi là:

> **Chỉ thay đổi một yếu tố lớn — mô hình tiền huấn luyện — còn lại giữ kiểm soát mọi điều kiện thực nghiệm tối đa.**

Cách tiếp cận này giúp nghiên cứu vừa khả thi trong khuôn khổ 3–4 tuần, vừa đảm bảo có câu hỏi rõ ràng, giả thuyết đo lường được, thực nghiệm tái lập và phân tích ý nghĩa.