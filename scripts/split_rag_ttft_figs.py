"""
Tách các figure RAG+TTFT (2 biểu đồ cạnh nhau) thành 2 ảnh riêng: _rag và _ttft.

Tự dò cột trắng (gap) giữa 2 biểu đồ để cắt, thay vì cắt cứng ở giữa.
Áp dụng cho image4/5/6.png (cấu hình 10/3, 15/5, 20/7).
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
FIGS = ROOT / "report_figures"

TARGETS = ["image4.png", "image5.png", "image6.png"]


def find_split_x(arr: np.ndarray) -> int:
    """Tìm cột x để cắt: cột 'trắng nhất' trong vùng giữa ảnh."""
    h, w = arr.shape[:2]
    gray = arr[..., :3].mean(axis=2) if arr.ndim == 3 else arr
    # độ "không trắng" của mỗi cột (giá trị cao = có nội dung)
    darkness = (255 - gray).sum(axis=0)
    # chỉ xét vùng giữa 35%-65% chiều rộng để tìm khe giữa 2 biểu đồ
    lo, hi = int(w * 0.35), int(w * 0.65)
    band = darkness[lo:hi]
    return lo + int(np.argmin(band))


def main():
    for name in TARGETS:
        src = FIGS / name
        if not src.exists():
            print(f"SKIP (not found): {src}")
            continue
        img = Image.open(src).convert("RGB")
        arr = np.asarray(img)
        split_x = find_split_x(arr)

        stem = src.stem
        left = img.crop((0, 0, split_x, img.height))
        right = img.crop((split_x, 0, img.width, img.height))

        left_path = FIGS / f"{stem}_rag.png"
        right_path = FIGS / f"{stem}_ttft.png"
        left.save(left_path)
        right.save(right_path)
        print(f"{name}: split at x={split_x} -> {left_path.name}, {right_path.name}")


if __name__ == "__main__":
    main()
