# Báo cáo Day 09 — Nguyen Han Truong Giang, 2A202603112

## 1. Kết luận trước

Pipeline kết luận **HOLD**. Trong tổng số 224 ảnh ban đầu, hệ thống tiếp nhận 193 ảnh và loại 31 ảnh do ảnh hỏng, kích thước không đạt, trùng lặp và rò rỉ tập test. Sau QC và xử lý, còn 0 lỗi bắt buộc nhưng có 34/193 ảnh còn cảnh báo, tương đương 17.62%, vượt ngưỡng Hold 10%.

Dataset được xuất tại `submission/export/clean_v1/` dưới dạng bản nháp Hold, gồm 193 ảnh trong manifest và 193 file nhãn. Đây chưa phải dataset được phép phát hành chính thức.

## 2. Lỗi và phân việc

Các lỗi được phát hiện trong vòng QC đầu tiên:

| Mã lỗi | Số lượng | Ai xử lý | Cách xử lý |
|---|---:|---|---|
| CORRUPT_IMAGE | 2 | Máy | Loại ảnh hỏng |
| BAD_SIZE | 3 | Máy | Loại ảnh sai kích thước |
| DUP_IMAGE_NEAR | 7 | Máy | Loại ảnh gần trùng |
| LEAK_TEST | 11 | Máy | Loại ảnh rò rỉ test |
| DUP_IMAGE_EXACT | 8 | Máy | Loại ảnh trùng hoàn toàn |
| MISSING_LABEL_FILE | 6 | Người | Chuyển người gán nhãn |
| BAD_FORMAT | 3 | Máy | Xóa dòng lỗi |
| BAD_CLASS_ID | 6 | Máy | Xóa dòng lỗi |
| BOX_ZERO | 4 | Máy | Xóa dòng lỗi |
| DUP_BOX | 7 | Máy | Xóa box trùng |
| BOX_OUT_OF_IMAGE | 8 | Máy | Cắt box về phạm vi ảnh |
| BAD_CLASS_NAME | 2 | Máy | Cập nhật tên class trong YAML |
| EMPTY_LABEL | 11 | Người | Kiểm tra file nhãn rỗng |
| BOX_HUGE | 3 | Người | Kiểm tra box quá lớn |
| BOX_TINY | 6 | Người | Kiểm tra box quá nhỏ |
| CLASS_RARE | 4 | Người | Kiểm tra lớp hiếm |
| POSSIBLE_MISSING | 53 | Người | Kiểm tra thiếu nhãn |
| POSSIBLE_LOOSE_BOX | 27 | Người | Kiểm tra box lỏng |
| POSSIBLE_WRONG_CLASS | 17 | Người | Kiểm tra sai class |
| POSSIBLE_FALSE_POS | 28 | Người | Kiểm tra nhãn thừa |

Tổng cộng có 36 lỗi bắt buộc và 149 cảnh báo.

Autofix thực hiện 35 thay đổi: 1 lần cập nhật class, 6 ảnh chuyển người gán nhãn, 20 dòng nhãn bị xóa và 8 box được cắt về trong ảnh.

Sau Autofix còn 6 lỗi bắt buộc, đều liên quan đến ảnh thiếu file nhãn.

Hàng đợi ưu tiên có 121 ảnh, trong khi ngân sách duyệt là 80 ảnh. Ảnh `img_0020` được ưu tiên đầu tiên với priority 11.524 do có lịch sử sửa dòng nhãn và các cảnh báo POSSIBLE_MISSING, POSSIBLE_WRONG_CLASS.

Trong 80 ảnh được đưa qua bước duyệt mô phỏng, có 88 lượt cảnh báo ngữ nghĩa thuộc 74 ảnh:
- POSSIBLE_MISSING: 51 lượt.
- POSSIBLE_WRONG_CLASS: 17 lượt.
- POSSIBLE_LOOSE_BOX: 8 lượt.
- POSSIBLE_FALSE_POS: 12 lượt.

Có 41 ảnh trong hàng đợi chưa được xử lý ở lượt duyệt này.

## 3. Calibration (2 ngưỡng)

| Tham số | Khởi đầu | Đã chọn | Precision / Recall trước → sau | Lý do |
|---|---:|---:|---|---|
| duplicates.phash_dist | 2 | 16 | Near duplicate: 1.00/0.50 → 1.00/0.83 | Tăng khả năng phát hiện ảnh gần trùng, giữ precision cao |
| second_opinion.conf | 0.25 | 0.60 | Missing: 0.50/1.00 → 0.57/1.00 | Cải thiện precision nhưng vẫn giữ recall cao |

Chi tiết thay đổi của second_opinion.conf:

- WRONG_CLASS: Precision 0.56 → 0.62, Recall giữ ở 0.83.
- MISSING: Precision 0.50 → 0.57, Recall giữ ở 1.00.
- LOOSE_BOX: Precision 0.29 → 0.20, Recall 0.40 → 0.20.
- FALSE_POS: Precision 0.31, Recall 1.00, không thay đổi.

Tôi chọn confidence 0.60 thay vì 0.80 vì muốn ưu tiên khả năng phát hiện thiếu nhãn. Ở ngưỡng 0.80, recall của POSSIBLE_MISSING giảm còn 0.75. Các cảnh báo ngữ nghĩa được chuyển cho người kiểm tra nên không nên chỉ tối ưu precision.

Một điểm có thể gây hiểu nhầm khi calibration là bộ calibration chỉ có 39 ảnh, trong đó có 6 ảnh gần trùng. Mỗi trường hợp bị bỏ sót làm recall thay đổi khoảng 16.7 điểm phần trăm. Vì vậy kết quả F1 cao chưa chắc phản ánh chính xác dữ liệu raw. Ngoài ra, calibration không có lỗi LEAK_TEST và các lỗi sai class được chèn không hoàn toàn giống phân bố lỗi trong raw. Cần mở kiểm tra trực quan các cặp ảnh gần trùng trước khi tin hoàn toàn vào kết quả.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được

Tôi chạy kiểm chứng Gate bằng lệnh:

`python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`

Kết quả:
- Không có ngân sách duyệt: 0 ảnh.
- Sau Autofix còn 6 lỗi bắt buộc.
- Có 6 hành động `needs_label` được ghi trong `03b_fixes_after_review.csv`.
- Kết luận: REWORK.
- Exit code: 20.
- `Test-Path .\work\exp_nb\clean_v1` trả về False.

Điều này chứng minh Gate không cho phép phát hành dataset khi còn lỗi bắt buộc và không có người bổ sung nhãn.

Hai điểm mù của pipeline:

1. Phát hiện box lỏng còn yếu. Ở confidence 0.60, POSSIBLE_LOOSE_BOX chỉ đạt Precision 0.20 và Recall 0.20 trên calibration. Sau QC cuối vẫn còn 18 cảnh báo loại này. Vì vậy cần người kiểm tra trực quan bounding box thay vì tin hoàn toàn vào model.

2. Cảnh báo false positive có độ chính xác thấp. Trên calibration, POSSIBLE_FALSE_POS chỉ đạt Precision 0.31, mặc dù Recall là 1.00. QC cuối còn 20 cảnh báo loại này. Điều đó cho thấy dự đoán của model không thể thay thế đánh giá của người gán nhãn.

Ngoài ra, pipeline không tự tạo nhãn cho ảnh thiếu file annotation. Đây là giới hạn an toàn có chủ ý: ảnh cần được người gán nhãn bổ sung thay vì để máy tự tạo nhãn không có kiểm chứng.

## 5. Train: raw vs clean (tùy chọn)

**Chưa chạy train trên Google Colab.**

Tôi đã hoàn thành phần pipeline, unit test, calibration, QC và kiểm chứng Gate. Phần huấn luyện so sánh raw và clean chưa được thực hiện nên chưa có kết quả mAP50, độ lệch chuẩn giữa các seed hoặc cơ sở để kết luận dataset clean tốt hơn raw.

Không đưa ra kết luận về hiệu quả huấn luyện khi chưa có kết quả thực nghiệm.

## 6. Quyết định và ghi chú AI

**Quyết định cuối cùng: HOLD.**

Dataset chưa đủ điều kiện Release vì còn 34/193 ảnh có cảnh báo (17.62%), vượt ngưỡng 10%. Để chuyển sang Release, cần kiểm tra và xử lý cảnh báo còn lại, đặc biệt các ảnh chưa được duyệt, đồng thời bảo đảm không phát sinh lỗi bắt buộc. Với 193 ảnh và ngưỡng 10%, số ảnh còn cảnh báo phải giảm xuống tối đa 19 ảnh.

**Ghi chú sử dụng AI:**

1. Prompt chính: Yêu cầu AI triển khai các module Python theo cấu trúc, docstring và unit test có sẵn trong repo; giữ nguyên các quy tắc và file kiểm thử của đề bài.

2. Cách kiểm code: Chạy unit test riêng từng module (14, 10, 11 và 15 tests), sau đó chạy toàn bộ pytest đạt 79/79. Tiếp tục kiểm tra bằng calibration và chạy pipeline trên dữ liệu thực tế.

3. Một lần AI đưa ra đề xuất chưa phù hợp: Ban đầu AI đề xuất confidence 0.80 vì Precision cao. Sau khi đối chiếu hướng dẫn calibration và đánh đổi Recall/Precision, tôi chọn 0.60 để giữ Recall của POSSIBLE_MISSING ở mức 1.00.
