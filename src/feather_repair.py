"""Second sweep over feathers that failed a fixable check (docs/sam3-segmentation-plan.md, idea 4).

After a feather is checked (src/feather_judge.py), a failed check that a new
mask could fix gets repair attempts:

  holes or gaps          fill the mask's enclosed holes (only when holes were measured)
  outline cut, material  re-segment with SAM 3's interactive predictor on a crop around
  or glue inside, other  the feather: box widened toward the cut sides, positive points
  feather inside         along the feather, negative points on its neighbours. Tried on
                         the contrast-enhanced crop (pale vane on white card stands out)
                         and on the plain crop.

Each attempt is checked again with the same checks, and replaces the first mask
only if it fails fewer checks and nothing new (its failures are a proper subset
of the first mask's). A new mask never takes pixels from a neighbouring
feather's mask. Fragments, pieces that are not a single feather and hidden
feathers are not repaired: a new mask cannot fix them.

Settings: configs/segmentation.json, feather_judge.repair.
"""
from __future__ import annotations

import cv2
import numpy as np

from src import feather_judge

REPAIRABLE = ("outline cut", "holes or gaps", "non-feather material inside", "glue inside", "other feather inside")
NOT_REPAIRABLE = ("fragment", "not a single feather")


def fail_items(rec: dict, prefix: str = "judge_") -> set[str]:
    """Failed checks as single items: 'outline cut: left,top' -> 'outline cut: left', 'outline cut: top'."""
    out = set()
    for f in feather_judge.failed_checks(rec, prefix):
        if f.startswith("outline cut: "):
            out |= {f"outline cut: {s}" for s in f.split(": ", 1)[1].split(",")}
        else:
            out.add(f)
    return out


def worth_repairing(rec: dict) -> set[str]:
    """The fixable failures of a checked feather (empty: leave it as it is)."""
    if rec.get("judge_status") != "ok":
        return set()
    items = fail_items(rec)
    if any(i.startswith(NOT_REPAIRABLE) for i in items):
        return set()
    return {i for i in items if i.startswith(REPAIRABLE)}


def better(old: set[str], new: set[str]) -> bool:
    return new < old


def fill_holes(mask: np.ndarray) -> np.ndarray | None:
    """The mask with its enclosed holes filled; None if it has none."""
    m = mask.astype(np.uint8)
    n, lbl, _, _ = cv2.connectedComponentsWithStats(1 - m, connectivity=4)
    border = set(np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]])).tolist())
    holes = np.isin(lbl, [i for i in range(1, n) if i not in border])
    return (mask | holes) if holes.any() else None


def _points_along(mask: np.ndarray, n: int = 3) -> list[list[float]]:
    """n points on the mask spread along its longer axis (on the feather, near its middle line)."""
    ys, xs = np.nonzero(mask)
    vertical = (ys.max() - ys.min()) >= (xs.max() - xs.min())
    a, b = (ys, xs) if vertical else (xs, ys)
    pts = []
    for q in np.linspace(0.2, 0.8, n):
        t = a.min() + q * (a.max() - a.min())
        near = np.abs(a - t) <= max(2, 0.01 * (a.max() - a.min()))
        if not near.any():
            continue
        c = np.median(b[near])
        k = np.flatnonzero(near)[np.argmin(np.abs(b[near] - c))]
        pts.append([float(xs[k]), float(ys[k])])
    return pts


def _inside_point(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    k = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
    return [float(xs[k]), float(ys[k])]


def _enhance(img: np.ndarray, clip: float = 3.0) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def resegment(inter, img: np.ndarray, mask: np.ndarray, others: list[np.ndarray], sides_cut: list[str],
              cfg: dict, enhance: bool) -> np.ndarray | None:
    """A new mask for one feather from SAM 3's interactive predictor (None if it gives nothing)."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    size = max(x1 - x0, y1 - y0)
    widen = int(float(cfg.get("widen_frac", 0.25)) * size)
    grow = {s: (widen if s in sides_cut else int(0.02 * size)) for s in ("left", "right", "top", "bottom")}
    bx0, by0 = max(0, x0 - grow["left"]), max(0, y0 - grow["top"])
    bx1, by1 = min(W - 1, x1 + grow["right"]), min(H - 1, y1 + grow["bottom"])
    pad = int(float(cfg.get("pad_frac", 0.15)) * size) + 20
    cx0, cy0, cx1, cy1 = max(0, bx0 - pad), max(0, by0 - pad), min(W, bx1 + pad), min(H, by1 + pad)
    pts = [[x - cx0, y - cy0] for x, y in _points_along(mask)]
    lab = [1] * len(pts)
    for o in others:
        sub = o[cy0:cy1, cx0:cx1] & ~mask[cy0:cy1, cx0:cx1]
        p = _inside_point(sub)
        if p is not None and sub.sum() > 0.02 * mask.sum():
            pts.append(p); lab.append(0)
    crop = img[cy0:cy1, cx0:cx1]
    if enhance:
        crop = _enhance(crop, float(cfg.get("contrast_clip", 3.0)))
    inter.set_image(np.ascontiguousarray(crop))
    res = inter(bboxes=[[bx0 - cx0, by0 - cy0, bx1 - cx0, by1 - cy0]], points=[pts], labels=[lab])
    if not res or res[0].masks is None or not len(res[0].masks.data):
        return None
    sub = res[0].masks.data[0].cpu().numpy() > 0.5
    new = np.zeros_like(mask)
    new[cy0:cy1, cx0:cx1] = sub[: cy1 - cy0, : cx1 - cx0]
    for o in others:                       # never take a neighbour's pixels
        new &= ~(o & ~mask)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    return lbl == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))   # the largest piece


def side_growth(old: np.ndarray, new: np.ndarray, others: np.ndarray) -> dict[str, float]:
    """New pixels on each side (left/right of the feather's middle) in the middle 15-75% of its
    length, as a share of the old mask's area there (base and tip left out)."""
    ys, xs = np.nonzero(old)
    y0, y1 = ys.min(), ys.max()
    rows = slice(int(y0 + 0.15 * (y1 - y0)), int(y0 + 0.75 * (y1 - y0)))
    xc = int(np.median(xs))
    gain = (new & ~old & ~others)[rows]
    base = max(1, int(old[rows].sum()))
    return {"left": round(float(gain[:, :xc].sum()) / base, 3), "right": round(float(gain[:, xc:].sum()) / base, 3)}


def regrow_check(inter, img: np.ndarray, mask: np.ndarray, others: list[np.ndarray], cfg: dict) -> dict:
    """Measured 'outline cut' on the sides: SAM 3 re-segments the feather on a crop (contrast-
    enhanced by default) with its box widened left and right. A pale vane left outside the mask on
    the white card, which neither the first pass nor the model sees, makes the new mask grow on
    that side (hand-checked: 58% on the one such feather, at most 3.6% on good ones). Returns
    {'regrow_left', 'regrow_right', 'regrow_cut': [sides]}; empty if SAM gives nothing."""
    new = resegment(inter, img, mask, others, ["left", "right"], {"widen_frac": cfg.get("widen_frac", 0.25)},
                    enhance=cfg.get("variant", "contrast") == "contrast")
    if new is None:
        return {}
    union = np.zeros_like(mask)
    for o in others:
        union |= o
    g = side_growth(mask, new, union)
    cut = [s for s in ("left", "right") if g[s] >= float(cfg.get("min_growth", 0.15))]
    return {"regrow_left": g["left"], "regrow_right": g["right"], "regrow_cut": cut}


def candidates(rec: dict, inter, img: np.ndarray, mask: np.ndarray, others: list[np.ndarray],
               cfg: dict) -> list[tuple[str, np.ndarray]]:
    """(name, new full mask) attempts for one feather, cheapest first, within the area limits."""
    fix = worth_repairing(rec)
    out = []
    if any(f.startswith("holes or gaps") for f in fix) and cfg.get("fill_holes", True):
        m = fill_holes(mask)
        if m is not None:
            out.append(("holes filled", m))
    if inter is not None and fix - {"holes or gaps"}:
        sides = [f.split(": ", 1)[1] for f in fix if f.startswith("outline cut: ")]
        for v in cfg.get("variants", ["contrast", "plain"]):
            m = resegment(inter, img, mask, others, sides, cfg, enhance=(v == "contrast"))
            if m is not None:
                out.append((f"re-segmented ({v}" + (f", widened {','.join(sides)})" if sides else ")"), m))
    lo, hi = float(cfg.get("min_area_ratio", 0.8)), float(cfg.get("max_area_ratio", 1.8))
    a = max(1, int(mask.sum()))
    return [(n, m) for n, m in out if lo <= m.sum() / a <= hi and not np.array_equal(m, mask)]
