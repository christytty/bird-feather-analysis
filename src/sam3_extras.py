"""Optional SAM 3 steps after the text-prompt pass (docs/sam3-segmentation-plan.md).

Both take and return the segmentation in the form FeatherProcessor uses:
(boxes, masks, failure_reason, scores).

exemplar_pass: SAM 3 again on the whole slide with the text prompt plus
    example boxes. Positive: the slide's best-scoring feather-shaped box of a
    typical size (one well-segmented feather). Negative: the label strip and
    the date corner. Aimed at merged and split feathers.
refine_pass: a second pass per feather with SAM 3's interactive predictor on a
    crop around it (its box, a positive point on it, negative points on the
    neighbours in the crop). The crop is far smaller than the slide, so SAM 3
    sees it at a higher effective resolution. A refined mask that barely
    overlaps the first one is discarded (the first mask is kept).

Settings: configs/segmentation.json, sam3.exemplar and sam3.refine.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def feather_shaped(box, min_aspect: float = 2.0) -> bool:
    w, h = box[2] - box[0], box[3] - box[1]
    return w > 0 and h > 0 and max(w, h) / min(w, h) >= min_aspect


def pick_exemplar(boxes, scores, min_aspect: float = 2.0) -> list[list[float]]:
    """The best-scoring feather-shaped box whose area is near the slide's median
    (a typical single feather, not a merged group or a sliver)."""
    if not boxes:
        return []
    areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
    med = float(np.median(areas))
    cands = [(s, b) for b, s, a in zip(boxes, scores, areas)
             if feather_shaped(b, min_aspect) and 0.6 * med <= a <= 1.6 * med]
    return [list(max(cands, key=lambda t: t[0])[1])] if cands else []


def negatives(width: int, height: int) -> list[list[float]]:
    """Label strip at the top and the date corner: never feathers."""
    return [[0, 0, width, 0.16 * height], [0.7 * width, 0.88 * height, width, height]]


def exemplar_pass(sam3, image_path: str, size: tuple[int, int], text: str, first, cfg: dict,
                  nms_iou: float, max_box_area_ratio: float, conf: float | None = None):
    """`first`: the text-prompt result. Returns it unchanged when there is no
    good example or the example pass finds nothing."""
    import torch
    import torchvision

    boxes, masks, reason, scores = first
    pos = pick_exemplar(boxes, scores, cfg.get("min_aspect", 2.0))
    if reason or not pos:
        return first
    W, H = size
    neg = negatives(W, H) if cfg.get("negatives", True) else []
    sam3.set_image(image_path)
    res = sam3(text=[text], bboxes=pos + neg, labels=[1] * len(pos) + [0] * len(neg))
    nb, ns, nm = [], [], []
    for r in res:
        if r.masks is None or r.boxes is None:
            continue
        for b, sc, m in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.masks.data.cpu().numpy()):
            if (b[2] - b[0]) * (b[3] - b[1]) > W * H * max_box_area_ratio:
                continue
            if conf is not None and float(sc) < float(conf):
                continue
            nb.append([float(v) for v in b]); ns.append(float(sc)); nm.append(m)
    if not nb:
        return first
    keep = torchvision.ops.nms(torch.tensor(nb), torch.tensor(ns), nms_iou)
    return [nb[int(i)] for i in keep], [nm[int(i)] for i in keep], "", [ns[int(i)] for i in keep]


def refine_pass(inter, image_path: str, first, cfg: dict) -> tuple[tuple, dict]:
    """Returns (segmentation, {'refined': n, 'kept_first': n})."""
    boxes, masks, reason, scores = first
    if reason or not masks:
        return first, {"refined": 0, "kept_first": 0}
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    full = [cv2.resize(m.astype(np.float32), (W, H)) > 0.5 for m in masks]
    centers = []
    for m in full:
        ys, xs = np.nonzero(m)
        if not len(ys):
            centers.append(None); continue
        cy, cx = ys.mean(), xs.mean()
        k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))   # a pixel on the feather near its middle
        centers.append((float(xs[k]), float(ys[k])))
    pad_frac, min_iou = float(cfg.get("pad_frac", 0.15)), float(cfg.get("min_iou", 0.5))
    out, changed, kept = [], 0, 0
    for i, m in enumerate(full):
        ys, xs = np.nonzero(m)
        if not len(ys):
            out.append(m.astype(np.float32)); continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        pad = int(pad_frac * max(x1 - x0, y1 - y0)) + 20
        cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
        pts, lab = [[centers[i][0] - cx0, centers[i][1] - cy0]], [1]
        for j, c in enumerate(centers):
            if j != i and c is not None and cx0 <= c[0] < cx1 and cy0 <= c[1] < cy1:
                pts.append([c[0] - cx0, c[1] - cy0]); lab.append(0)
        inter.set_image(np.ascontiguousarray(img[cy0:cy1, cx0:cx1]))
        res = inter(bboxes=[[x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0]], points=[pts], labels=[lab])
        if not res or res[0].masks is None:
            out.append(m.astype(np.float32)); kept += 1; continue
        sub = res[0].masks.data[0].cpu().numpy() > 0.5
        new = np.zeros_like(m)
        new[cy0:cy1, cx0:cx1] = sub[: cy1 - cy0, : cx1 - cx0]
        iou = np.logical_and(new, m).sum() / max(1, np.logical_or(new, m).sum())
        if iou < min_iou:
            out.append(m.astype(np.float32)); kept += 1
        else:
            out.append(new.astype(np.float32)); changed += 1
    return (boxes, out, reason, scores), {"refined": changed, "kept_first": kept, "slide": Path(image_path).name}


def _short_side(mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    return float(min(w, h))


def split_merged(sam3, image_path: str, text: str, first, cfg: dict) -> tuple[tuple, dict]:
    """Repairs masks that look like several feathers merged into one: a mask at
    least `width_ratio` times the slide's typical feather width, or `area_ratio`
    times its typical area, is segmented again on a crop around it (text prompt
    only). If that finds `min_parts` or more feather-sized pieces lying mostly
    inside the mask and covering most of it, they replace it; otherwise the mask
    is kept. Returns (segmentation, {'tried': n, 'split': n})."""
    boxes, masks, reason, scores = first
    info = {"tried": 0, "split": 0, "slide": Path(image_path).name}
    if reason or len(masks) < 2:
        return first, info
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    full = [cv2.resize(m.astype(np.float32), (W, H)) > 0.5 for m in masks]
    areas = np.array([m.sum() for m in full], dtype=float)
    widths = np.array([_short_side(m) for m in full], dtype=float)
    med_a, med_w = float(np.median(areas)), float(np.median(widths))
    a_ratio, w_ratio = float(cfg.get("area_ratio", 1.8)), float(cfg.get("width_ratio", 1.6))
    min_piece, min_parts = float(cfg.get("min_piece_frac", 0.35)), int(cfg.get("min_parts", 2))
    min_cover, inside = float(cfg.get("min_cover", 0.6)), float(cfg.get("min_inside", 0.6))
    out_b, out_m, out_s = [], [], []
    for b, m, s, a, w in zip(boxes, full, scores, areas, widths):
        if not (a >= a_ratio * med_a or (med_w and w >= w_ratio * med_w)):
            out_b.append(b); out_m.append(m.astype(np.float32)); out_s.append(s); continue
        info["tried"] += 1
        ys, xs = np.nonzero(m)
        pad = 20
        cx0, cy0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        cx1, cy1 = min(W, xs.max() + pad), min(H, ys.max() + pad)
        sam3.set_image(np.ascontiguousarray(img[cy0:cy1, cx0:cx1]))
        pieces = []
        for r in sam3(text=[text]):
            if r.masks is None or r.boxes is None:
                continue
            for pm, ps in zip(r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                full_p = np.zeros_like(m)
                full_p[cy0:cy1, cx0:cx1] = pm[: cy1 - cy0, : cx1 - cx0] > 0.5
                pa = full_p.sum()
                if pa >= min_piece * med_a and np.logical_and(full_p, m).sum() / max(pa, 1) >= inside:
                    pieces.append((float(ps), full_p))
        # one piece per feather: drop pieces that mostly repeat a better-scoring one
        kept: list[tuple[float, np.ndarray]] = []
        for ps, pm in sorted(pieces, key=lambda t: -t[0]):
            if all(np.logical_and(pm, q).sum() / max(min(pm.sum(), q.sum()), 1) < 0.5 for _, q in kept):
                kept.append((ps, pm))
        union = np.zeros_like(m)
        for _, pm in kept:
            union |= pm
        if len(kept) >= min_parts and np.logical_and(union, m).sum() / max(m.sum(), 1) >= min_cover:
            info["split"] += 1
            for ps, pm in kept:
                yy, xx = np.nonzero(pm)
                out_b.append([float(xx.min()), float(yy.min()), float(xx.max()), float(yy.max())])
                out_m.append(pm.astype(np.float32)); out_s.append(ps)
        else:
            out_b.append(b); out_m.append(m.astype(np.float32)); out_s.append(s)
    return (out_b, out_m, reason, out_s), info
