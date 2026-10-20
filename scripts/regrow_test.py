"""Does SAM 3, given a contrast-enhanced crop and a box widened sideways, grow a feather's mask on
one side? A pale vane left outside the mask on the white card should make it grow there.

Runs on a worker node (SAM 3 weights there), on masks saved by a run (feather_masks/):

    python scripts/regrow_test.py <masks dir> <slides dir> <list.csv: file_id,crop> <out.csv>
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(os.getenv("FEATHER_REPO") or Path(__file__).resolve().parent.parent).expanduser()
sys.path.insert(0, str(REPO))
from src import feather_repair as fr  # noqa: E402


def growth(old: np.ndarray, new: np.ndarray, others: np.ndarray) -> dict:
    """New pixels per side (left/right of the feather's middle), in the middle 15-75% of its length,
    as a share of the old mask's area there."""
    ys, xs = np.nonzero(old)
    y0, y1 = ys.min(), ys.max()
    rows = slice(int(y0 + 0.15 * (y1 - y0)), int(y0 + 0.75 * (y1 - y0)))
    xc = int(np.median(xs))
    gain = (new & ~old & ~others)[rows]
    base = max(1, int(old[rows].sum()))
    return {"left": round(float(gain[:, :xc].sum()) / base, 3), "right": round(float(gain[:, xc:].sum()) / base, 3)}


def main() -> int:
    masks, slides, listing, out = (Path(a) for a in sys.argv[1:5])
    import torch
    from ultralytics.models.sam import SAM3Predictor

    inter = SAM3Predictor(overrides={"model": str(REPO / "sam3.pt"), "imgsz": 1008, "task": "segment", "mode": "predict",
                                     "save": False, "verbose": False,
                                     "device": "mps" if torch.backends.mps.is_available() else "cpu"})
    items = list(csv.DictReader(open(listing)))
    by_slide: dict = {}
    for it in items:
        by_slide.setdefault(it["file_id"], []).append(it["crop"])
    rows = []
    for fid, crops in by_slide.items():
        img = cv2.imread(str(slides / fid))
        all_masks = {c[:-4]: cv2.imread(str(masks / (c[:-4] + ".png")), 0) > 0 for c in crops}
        for c in crops:
            m = all_masks[c[:-4]]
            others = np.zeros_like(m)
            for k, o in all_masks.items():
                if k != c[:-4]:
                    others |= o
            row = {"file_id": fid, "crop": c}
            for variant in ("contrast", "plain"):
                new = fr.resegment(inter, img, m, [o for k, o in all_masks.items() if k != c[:-4]],
                                   ["left", "right"], {"widen_frac": 0.25}, enhance=(variant == "contrast"))
                g = growth(m, new, others) if new is not None else {"left": None, "right": None}
                row[f"{variant}_left"], row[f"{variant}_right"] = g["left"], g["right"]
            rows.append(row)
            print(row, flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
