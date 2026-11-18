"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""

"""CHECK 6 — Ý kiến thứ hai: đối chiếu nhãn với YOLO26l."""

import math

from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def run(ds, cfg, preds=None):

    # Nếu không truyền dự đoán, lấy từ model cache
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    rules = cfg["second_opinion"]

    conf_threshold = float(rules["conf"])
    min_area_px = float(rules["min_area_px"])
    fp_conf = float(rules["fp_conf"])

    iou_missing = float(rules["iou_missing"])
    iou_match = float(rules["iou_match"])
    iou_loose = float(rules["iou_loose"])

    issues = []

    # Duyệt ảnh theo thứ tự tên
    for image_path in ds.image_files():

        stem = image_path.stem

        # Bỏ qua ảnh không có dự đoán
        if stem not in preds:
            continue

        # Bỏ qua ảnh không có file nhãn
        raw_labels = ds.read_label_lines(stem)

        if raw_labels is None:
            continue

        # Kích thước ảnh (width, height)
        image_size = ds.image_size(stem)

        if image_size is None:
            continue

        image_w, image_h = image_size

        # ------------------------------------------------
        # 1. Đọc các nhãn hợp lệ, giữ số dòng gốc
        # ------------------------------------------------

        labels = []

        for line_no, raw_line in raw_labels:

            try:
                class_id, cx, cy, w, h = parse_yolo_line(
                    raw_line
                )
            except ValueError:
                continue

            # Chỉ giữ class ID hợp lệ
            if not (0 <= class_id < len(ds.classes)):
                continue

            labels.append({
                "class_id": class_id,
                "box": (cx, cy, w, h),
                "line": line_no
            })

        # ------------------------------------------------
        # 2. Chuẩn hóa danh sách dự đoán của model
        # ------------------------------------------------

        predictions = []

        for row in preds[stem]:

            try:
                if len(row) != 6:
                    continue

                raw_class, raw_conf, cx, cy, w, h = row

                class_id = int(raw_class)

                if float(raw_class) != class_id:
                    continue

                conf = float(raw_conf)
                box = (float(cx), float(cy), float(w), float(h))

                if not (0 <= class_id < len(ds.classes)):
                    continue

                if not math.isfinite(conf) or any(
                    not math.isfinite(v) for v in box
                ):
                    continue

            except (TypeError, ValueError, OverflowError):
                continue

            predictions.append({
                "class_id": class_id,
                "conf": conf,
                "box": box
            })

        # ------------------------------------------------
        # 3. Lọc dự đoán đủ confidence và diện tích
        # ------------------------------------------------

        selected = []

        for p in predictions:

            cx, cy, w, h = p["box"]

            area_px = (w * image_w) * (h * image_h)

            if (
                p["conf"] >= conf_threshold
                and area_px >= min_area_px
            ):
                selected.append(p)

        # ------------------------------------------------
        # 4. So sánh dự đoán với annotation
        # ------------------------------------------------

        for p in selected:

            candidates = []

            for label in labels:

                iou = iou_xywh(
                    p["box"],
                    label["box"]
                )

                if iou >= iou_missing:
                    candidates.append({
                        "label": label,
                        "iou": iou
                    })

            # Trường hợp 1: Model thấy vật nhưng không có
            # nhãn nào khớp đủ ngưỡng IoU
            if not candidates:

                issues.append(
                    Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_MISSING",
                        stem,
                        (
                            f"Model phát hiện class {p['class_id']} "
                            f"nhưng không tìm thấy nhãn tương ứng "
                            f"(conf={p['conf']:.3f})"
                        ),
                        extra={"conf": p["conf"]}
                    )
                )

                continue

            # Tách các nhãn cùng class và khác class
            same_class = [
                c for c in candidates
                if c["label"]["class_id"] == p["class_id"]
            ]

            other_class = [
                c for c in candidates
                if c["label"]["class_id"] != p["class_id"]
            ]

            # Trường hợp 2: Có nhãn cùng class
            if same_class:

                best = max(
                    same_class,
                    key=lambda c: c["iou"]
                )

                # IoU thấp: box có thể quá lỏng
                if best["iou"] < iou_loose:

                    issues.append(
                        Issue(
                            "second_opinion",
                            WARN,
                            "POSSIBLE_LOOSE_BOX",
                            stem,
                            (
                                f"Box class {p['class_id']} "
                                f"có thể quá lỏng, IoU={best['iou']:.3f}"
                            ),
                            extra={
                                "conf": p["conf"],
                                "iou": best["iou"]
                            }
                        )
                    )

                # Nếu IoU đủ tốt thì không cảnh báo
                continue

            # Trường hợp 3: Không có nhãn cùng class,
            # nhưng có nhãn khác class khớp vị trí
            if other_class:

                best = max(
                    other_class,
                    key=lambda c: c["iou"]
                )

                if best["iou"] >= iou_match:

                    issues.append(
                        Issue(
                            "second_opinion",
                            WARN,
                            "POSSIBLE_WRONG_CLASS",
                            stem,
                            (
                                f"Model dự đoán class {p['class_id']}, "
                                f"annotation là class "
                                f"{best['label']['class_id']}, "
                                f"IoU={best['iou']:.3f}"
                            ),
                            extra={
                                "conf": p["conf"],
                                "iou": best["iou"]
                            }
                        )
                    )

        # ------------------------------------------------
        # 5. Tìm annotation có thể là false positive
        # ------------------------------------------------

        # Dùng fp_conf riêng.
        # Không lọc min_area_px ở bước này.
        fp_predictions = [
            p for p in predictions
            if p["conf"] >= fp_conf
        ]

        for label in labels:

            best_iou = max(
                (
                    iou_xywh(label["box"], p["box"])
                    for p in fp_predictions
                ),
                default=0.0
            )

            if best_iou < iou_missing:

                issues.append(
                    Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_FALSE_POS",
                        stem,
                        (
                            f"Annotation class {label['class_id']} "
                            f"không khớp dự đoán model, "
                            f"IoU tốt nhất={best_iou:.3f}"
                        ),
                        line=label["line"],
                        extra={"iou": best_iou}
                    )
                )

    return issues
