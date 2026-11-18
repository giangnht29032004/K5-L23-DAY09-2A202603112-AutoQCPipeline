"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""

"""BƯỚC 2/4 — Sửa tự động các lỗi bắt buộc an toàn."""

from collections import defaultdict
from pathlib import Path

import yaml

from ..common import ERROR, parse_yolo_line


# Các lỗi được phép xóa dòng nhãn
DROP_CODES = {
    "BAD_FORMAT",
    "BAD_CLASS_ID",
    "BOX_ZERO",
    "DUP_BOX",
}

# Các lỗi yêu cầu loại bỏ cả ảnh
REMOVE_CODES = {
    "LEAK_TEST",
    "CORRUPT_IMAGE",
    "BAD_SIZE",
}


def _format_number(value):
    """Định dạng số gọn, tránh ghi quá nhiều chữ số thập phân."""
    return format(value, ".10g")


def _clip_box(raw_line):
    """
    Cắt bounding box về phạm vi [0, 1].

    Trả về:
        (new_line, True) nếu box hợp lệ sau khi cắt.
        (None, False) nếu box nằm hoàn toàn ngoài ảnh.
    """
    try:
        class_id, cx, cy, w, h = parse_yolo_line(raw_line)
    except ValueError:
        return None, False

    left = max(0.0, cx - w / 2)
    right = min(1.0, cx + w / 2)

    top = max(0.0, cy - h / 2)
    bottom = min(1.0, cy + h / 2)

    new_w = right - left
    new_h = bottom - top

    # Box không còn diện tích sau khi cắt
    if new_w <= 0 or new_h <= 0:
        return None, False

    new_cx = (left + right) / 2
    new_cy = (top + bottom) / 2

    new_line = (
        f"{class_id} "
        f"{_format_number(new_cx)} "
        f"{_format_number(new_cy)} "
        f"{_format_number(new_w)} "
        f"{_format_number(new_h)}"
    )

    return new_line, True


def fix(ds, issues, cfg):
    """
    Sửa lỗi ERROR trong dataset.

    Trả về danh sách thay đổi:
    image, line, action, before, after.
    """

    fixes = []

    # Chỉ xử lý lỗi bắt buộc, không sửa WARNING
    errors = [
        issue for issue in issues
        if issue.severity == ERROR
    ]

    # --------------------------------------------------
    # 1. Xác định các ảnh cần loại bỏ
    # --------------------------------------------------

    remove_stems = set()

    for issue in errors:
        if (
            issue.code.startswith("DUP_IMAGE_")
            or issue.code in REMOVE_CODES
        ):
            remove_stems.add(issue.image)

    # Xóa ảnh và file nhãn trước các thao tác khác
    for stem in sorted(remove_stems):

        image_path = ds.image_path(stem)
        label_path = ds.label_path(stem)

        if image_path is None and not label_path.exists():
            continue

        before = (
            image_path.name
            if image_path is not None
            else "(không có ảnh)"
        )

        if image_path is not None:
            image_path.unlink(missing_ok=True)

        label_path.unlink(missing_ok=True)

        fixes.append({
            "image": stem,
            "line": None,
            "action": "remove_image",
            "before": before,
            "after": "đã xóa ảnh và file nhãn"
        })

    # --------------------------------------------------
    # 2. Sửa tên class trong data.yaml
    # --------------------------------------------------

    if any(i.code == "BAD_CLASS_NAME" for i in errors):

        yaml_path = ds.yaml_path

        if yaml_path.exists():
            before_text = yaml_path.read_text(
                encoding="utf-8"
            )

            data = yaml.safe_load(before_text) or {}

            if not isinstance(data, dict):
                data = {}

            old_names = data.get("names", {})

            data["names"] = {
                i: name
                for i, name in enumerate(cfg["classes"])
            }

            yaml_path.write_text(
                yaml.safe_dump(
                    data,
                    allow_unicode=True,
                    sort_keys=False
                ),
                encoding="utf-8"
            )

            fixes.append({
                "image": "data.yaml",
                "line": None,
                "action": "rename_classes",
                "before": str(old_names),
                "after": str(cfg["classes"])
            })

    # --------------------------------------------------
    # 3. Chuyển ảnh thiếu nhãn cho người gán
    # --------------------------------------------------

    missing_stems = {
        i.image
        for i in errors
        if i.code == "MISSING_LABEL_FILE"
    }

    for stem in sorted(missing_stems):

        if stem in remove_stems:
            continue

        label_path = ds.label_path(stem)

        # Không tạo file nhãn giả
        if label_path.exists():
            continue

        if ds.image_path(stem) is None:
            continue

        fixes.append({
            "image": stem,
            "line": None,
            "action": "needs_label",
            "before": "(không có file)",
            "after": "chuyển người gán"
        })

    # --------------------------------------------------
    # 4. Gom lỗi theo file và số dòng gốc
    # --------------------------------------------------

    by_image = defaultdict(list)

    for issue in errors:
        if issue.line is not None:
            by_image[issue.image].append(issue)

    for stem in sorted(by_image):

        # Ảnh đã bị xóa thì bỏ qua các lỗi dòng
        if stem in remove_stems:
            continue

        label_path = ds.label_path(stem)

        if not label_path.exists():
            continue

        original_text = label_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        original_lines = original_text.splitlines()

        # Dùng số dòng gốc, không đánh lại số dòng
        drop_lines = set()
        clip_lines = set()

        for issue in by_image[stem]:

            if issue.line < 1:
                continue

            if issue.code in DROP_CODES:
                drop_lines.add(issue.line)

            elif issue.code == "BOX_OUT_OF_IMAGE":
                clip_lines.add(issue.line)

        # Nếu một dòng vừa cần xóa vừa cần cắt:
        # XÓA dòng, không thực hiện clip.
        clip_lines -= drop_lines

        new_lines = []

        for line_no, raw_line in enumerate(
            original_lines, start=1
        ):

            # 4a. Xóa dòng lỗi
            if line_no in drop_lines:

                fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "drop_line",
                    "before": raw_line,
                    "after": ""
                })

                continue

            # 4b. Cắt box tràn khỏi ảnh
            if line_no in clip_lines:

                clipped_line, valid = _clip_box(raw_line)

                if not valid:

                    fixes.append({
                        "image": stem,
                        "line": line_no,
                        "action": "drop_line",
                        "before": raw_line,
                        "after": ""
                    })

                    continue

                fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "clip_box",
                    "before": raw_line,
                    "after": clipped_line
                })

                new_lines.append(clipped_line)

            else:
                new_lines.append(raw_line)

        # Ghi lại file một lần duy nhất
        has_trailing_newline = original_text.endswith(
            ("\n", "\r")
        )

        new_text = "\n".join(new_lines)

        if has_trailing_newline and new_lines:
            new_text += "\n"

        if new_text != original_text:
            label_path.write_text(
                new_text,
                encoding="utf-8"
            )

    return fixes
