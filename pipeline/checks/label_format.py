"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""

"""CHECK 2 — Kiểm tra định dạng và chất lượng bounding box YOLO."""

from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line


def run(ds, cfg):
    rules = cfg["label_format"]

    overflow_tol = float(rules["overflow_tol"])
    min_box_px = float(rules["min_box_px"])
    max_box_frac = float(rules["max_box_frac"])
    dup_box_iou = float(rules["dup_box_iou"])

    issues = []

    # Duyệt từng file nhãn
    for label_file in ds.label_files():
        stem = label_file.stem

        lines = ds.read_label_lines(stem) or []
        image_size = ds.image_size(stem)

        # Chỉ so sánh box trong cùng một ảnh
        previous_boxes = []

        # Duyệt từng dòng, giữ nguyên số dòng gốc
        for line_no, raw_line in lines:

            # 1. Kiểm tra định dạng YOLO
            try:
                class_id, cx, cy, w, h = parse_yolo_line(raw_line)

            except ValueError as exc:
                issues.append(
                    Issue(
                        "label_format",
                        ERROR,
                        "BAD_FORMAT",
                        stem,
                        f"Định dạng YOLO không hợp lệ: {exc}",
                        line_no
                    )
                )
                continue

            # 2. Kiểm tra ID class
            if not (0 <= class_id < len(ds.classes)):
                issues.append(
                    Issue(
                        "label_format",
                        ERROR,
                        "BAD_CLASS_ID",
                        stem,
                        f"Class ID {class_id} không hợp lệ",
                        line_no
                    )
                )
                continue

            # 3. Kiểm tra kích thước box
            if w <= 0 or h <= 0:
                issues.append(
                    Issue(
                        "label_format",
                        ERROR,
                        "BOX_ZERO",
                        stem,
                        "Chiều rộng hoặc chiều cao bounding box <= 0",
                        line_no
                    )
                )
                continue

            # 4. Kiểm tra box có vượt khỏi ảnh không
            left = cx - w / 2
            right = cx + w / 2
            top = cy - h / 2
            bottom = cy + h / 2

            overflow = max(
                0.0,
                -left,
                -top,
                right - 1.0,
                bottom - 1.0
            )

            if overflow > overflow_tol:
                issues.append(
                    Issue(
                        "label_format",
                        ERROR,
                        "BOX_OUT_OF_IMAGE",
                        stem,
                        f"Bounding box vượt khỏi ảnh: {overflow:.6f}",
                        line_no,
                        extra={"overflow": overflow}
                    )
                )

            current_box = (cx, cy, w, h)

            # 5. Kiểm tra box trùng cùng class
            duplicated = any(
                old_class == class_id
                and iou_xywh(current_box, old_box) > dup_box_iou
                for old_class, old_box in previous_boxes
            )

            if duplicated:
                issues.append(
                    Issue(
                        "label_format",
                        ERROR,
                        "DUP_BOX",
                        stem,
                        "Bounding box trùng với box trước đó cùng class",
                        line_no
                    )
                )

            previous_boxes.append((class_id, current_box))

            # 6. Kiểm tra box quá nhỏ
            if image_size is not None:
                image_w, image_h = image_size

                box_w_px = w * image_w
                box_h_px = h * image_h

                short_side = min(box_w_px, box_h_px)

                if short_side < min_box_px:
                    issues.append(
                        Issue(
                            "label_format",
                            WARN,
                            "BOX_TINY",
                            stem,
                            f"Cạnh ngắn của box chỉ {short_side:.2f}px",
                            line_no
                        )
                    )

            # 7. Kiểm tra box quá lớn
            if w * h > max_box_frac:
                issues.append(
                    Issue(
                        "label_format",
                        WARN,
                        "BOX_HUGE",
                        stem,
                        f"Diện tích box chiếm {w * h:.4f} ảnh",
                        line_no
                    )
                )

    return issues
