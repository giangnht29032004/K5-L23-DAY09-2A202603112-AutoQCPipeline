"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""

"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework."""

from ..common import ERROR, WARN


def decide(issues, image_stems, cfg, reviewed=frozenset()):
    # Lấy ngưỡng cảnh báo từ cấu hình
    hold_threshold = float(cfg["gate"]["warn_ratio_hold"])

    # Tập hợp ảnh trong dataset
    image_set = set(image_stems)
    n_images = len(image_set)

    # Các ảnh đã được người duyệt
    reviewed_set = set(reviewed)

    # Đếm tất cả lỗi bắt buộc
    # Lỗi ERROR không được bỏ qua kể cả khi ảnh đã reviewed
    n_errors = sum(
        1 for issue in issues
        if issue.severity == ERROR
    )

    # Đếm số ảnh còn cảnh báo chưa được xử lý
    warn_images = {
        issue.image
        for issue in issues
        if issue.severity == WARN
        and issue.image in image_set
        and issue.image not in reviewed_set
    }

    n_warn_images = len(warn_images)

    # Tỷ lệ ảnh còn cảnh báo
    warn_ratio = (
        n_warn_images / n_images
        if n_images > 0
        else 0.0
    )

    # Cây quyết định
    if n_errors > 0:
        decision = "Rework"

        reasons = [
            f"Còn {n_errors} lỗi bắt buộc (ERROR); "
            "phải sửa trước khi phát hành."
        ]

    elif warn_ratio >= hold_threshold:
        decision = "Hold"

        reasons = [
            f"Có {n_warn_images}/{n_images} ảnh còn cảnh báo "
            f"({warn_ratio:.2%}), đạt hoặc vượt ngưỡng "
            f"Hold ({hold_threshold:.2%})."
        ]

    else:
        decision = "Release"

        reasons = [
            f"Không còn lỗi bắt buộc; "
            f"{n_warn_images}/{n_images} ảnh còn cảnh báo "
            f"({warn_ratio:.2%}), thấp hơn ngưỡng "
            f"Hold ({hold_threshold:.2%})."
        ]

    return {
        "decision": decision,
        "n_errors": n_errors,
        "n_warn_images": n_warn_images,
        "n_images": n_images,
        "warn_ratio": round(warn_ratio, 4),
        "reasons": reasons,
    }
