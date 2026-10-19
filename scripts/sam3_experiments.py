"""SAM 3 segmentation experiments (docs/sam3-segmentation-plan.md), run on one worker node.

Each slide goes through the pipeline's own FeatherProcessor.process_image once
per variant: the same mask clean-up, crops, per-feather checks and
missed-feather check. Only the SAM 3 step differs:

  baseline         the text prompt on the whole slide (what the pipeline does now)
  exemplar         idea 1: text prompt plus example boxes: positive = the slide's
                   best-scoring feather-shaped box from the baseline pass,
                   negative = the label strip and the date corner
  refine           idea 2: the baseline masks, then a second pass per feather with
                   SAM 3's interactive predictor on a crop around it (box, a
                   positive point on the feather, negative points on neighbours)
  exemplar_refine  both

Runs outside the job queue with FEATHER_ENABLE_VLM=0 (no second copy of the
VLM on the node; the per-feather checks use the VLM server). Usage on a node:

    cd <repo> && FEATHER_ENABLE_VLM=0 pixi run python scripts/sam3_experiments.py \\
        --slides "A3818 2002-im1708.jpg" ... --out data/runs/test-sam3exp-<id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(os.getenv("FEATHER_REPO") or Path(__file__).resolve().parent.parent).expanduser()
sys.path.insert(0, str(REPO))

VARIANTS = ("baseline", "exemplar", "refine", "exemplar_refine")
KEEP = ("seq", "crop", "segment_status", "needs_review", "judge_status", "judge_score", "judge_sides_cut",
        "judge_holes_or_gaps", "judge_holes", "judge_hole_frac", "judge_mask_parts", "judge_single_feather",
        "judge_fragment", "judge_other_feather_inside", "judge_other_material_inside", "judge_notes",
        "area_px", "length_px", "width_px", "postprocess", "inventory_status", "inventory_total",
        "feathers_possibly_missed", "missed_between")


def feather_shaped(box, min_aspect: float = 2.0) -> bool:
    w, h = box[2] - box[0], box[3] - box[1]
    return w > 0 and h > 0 and max(w, h) / min(w, h) >= min_aspect


def pick_exemplar(boxes, scores) -> list[list[float]]:
    """The best-scoring feather-shaped box whose area is near the slide's median
    (a typical single feather, not a merged group or a sliver)."""
    if not boxes:
        return []
    areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
    med = float(np.median(areas))
    cands = [(s, b) for b, s, a in zip(boxes, scores, areas) if feather_shaped(b) and 0.6 * med <= a <= 1.6 * med]
    if not cands:
        return []
    return [max(cands, key=lambda t: t[0])[1]]


def negatives(width: int, height: int) -> list[list[float]]:
    """Label strip at the top and the date corner: never feathers."""
    return [[0, 0, width, 0.16 * height], [0.7 * width, 0.88 * height, width, height]]


def make_variants(p, weights: str, stats: dict):
    """Segmentation functions with the signature of FeatherProcessor._segment_sam3."""
    import torch
    import torchvision
    from ultralytics.models.sam import SAM3Predictor

    base = p._segment_sam3
    inter = SAM3Predictor(overrides={"model": weights, "imgsz": 1008, "task": "segment", "mode": "predict",
                                     "device": "mps" if torch.backends.mps.is_available() else "cpu",
                                     "save": False, "verbose": False})

    def exemplar(image_path, img_pil, prompts, nms_iou, max_box_area_ratio, retry=False, conf=None):
        boxes, masks, reason, scores = base(image_path, img_pil, prompts, nms_iou, max_box_area_ratio, retry, conf)
        pos = pick_exemplar(boxes, scores)
        if reason or not pos:
            stats.setdefault("exemplar_skipped", []).append(Path(image_path).name)
            return boxes, masks, reason, scores
        W, H = img_pil.size
        neg = negatives(W, H)
        p.sam3.set_image(image_path)
        res = p.sam3(text=[prompts[0].rstrip(".")], bboxes=pos + neg, labels=[1] * len(pos) + [0] * len(neg))
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
            stats.setdefault("exemplar_empty", []).append(Path(image_path).name)
            return boxes, masks, reason, scores
        keep = torchvision.ops.nms(torch.tensor(nb), torch.tensor(ns), nms_iou)
        stats.setdefault("exemplar_counts", {})[Path(image_path).name] = [len(boxes), len(keep)]
        return [nb[int(i)] for i in keep], [nm[int(i)] for i in keep], "", [ns[int(i)] for i in keep]

    def refine_with(first):
        def refine(image_path, img_pil, prompts, nms_iou, max_box_area_ratio, retry=False, conf=None):
            boxes, masks, reason, scores = first(image_path, img_pil, prompts, nms_iou, max_box_area_ratio,
                                                 retry, conf)
            if reason or not masks:
                return boxes, masks, reason, scores
            img = cv2.imread(image_path)
            H, W = img.shape[:2]
            full = [cv2.resize(m.astype(np.float32), (W, H)) > 0.5 for m in masks]
            centers = []
            for m in full:
                ys, xs = np.nonzero(m)
                # a point on the feather itself, near its middle (the mask pixel closest to the centroid)
                cy, cx = ys.mean(), xs.mean()
                k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)) if len(ys) else 0
                centers.append((float(xs[k]), float(ys[k])) if len(ys) else None)
            out, changed, kept = [], 0, 0
            for i, m in enumerate(full):
                ys, xs = np.nonzero(m)
                if not len(ys) or centers[i] is None:
                    out.append(m.astype(np.float32)); continue
                x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
                pad = int(0.15 * max(x1 - x0, y1 - y0)) + 20
                cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
                crop = np.ascontiguousarray(img[cy0:cy1, cx0:cx1])
                pts = [[centers[i][0] - cx0, centers[i][1] - cy0]]
                lab = [1]
                for j, c in enumerate(centers):
                    if j != i and c is not None and cx0 <= c[0] < cx1 and cy0 <= c[1] < cy1:
                        pts.append([c[0] - cx0, c[1] - cy0]); lab.append(0)
                inter.set_image(crop)
                res = inter(bboxes=[[x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0]], points=[pts], labels=[lab])
                if not res or res[0].masks is None:
                    out.append(m.astype(np.float32)); kept += 1; continue
                sub = res[0].masks.data[0].cpu().numpy() > 0.5
                new = np.zeros_like(m)
                new[cy0:cy1, cx0:cx1] = sub[: cy1 - cy0, : cx1 - cx0]
                inter_ = np.logical_and(new, m).sum()
                iou = inter_ / max(1, np.logical_or(new, m).sum())
                if iou < 0.5:            # the second pass went somewhere else: keep the first mask
                    out.append(m.astype(np.float32)); kept += 1
                else:
                    out.append(new.astype(np.float32)); changed += 1
            stats.setdefault("refine", {})[Path(image_path).name] = {"refined": changed, "kept_first": kept}
            return boxes, out, reason, scores
        return refine

    return {"baseline": base, "exemplar": exemplar, "refine": refine_with(base),
            "exemplar_refine": refine_with(exemplar)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slides", nargs="+", required=True)
    ap.add_argument("--raw-dir", default=str(REPO / "data" / "raw"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    os.environ.setdefault("FEATHER_ENABLE_VLM", "0")
    os.environ.setdefault("FEATHER_ENABLE_LABELS", "0")
    from src.feather_processing import FeatherProcessor

    p = FeatherProcessor()
    if p.segmenter != "sam3":
        print("SAM 3 weights not found; nothing to compare"); return 2
    weights = os.getenv("FEATHER_SAM3_WEIGHTS", str(REPO / "sam3.pt"))
    stats: dict = {}
    fns = make_variants(p, weights, stats)
    out = Path(args.out)
    results: dict = {}
    for v in args.variants.split(","):
        p._segment_sam3 = fns[v]
        results[v] = {}
        for s in args.slides:
            t0 = time.time()
            r = p.process_image(str(Path(args.raw_dir) / s), str(out / v))
            results[v][s] = {"success": r.success, "reason": r.reason, "seconds": round(time.time() - t0, 1),
                             "feathers": [{k: f.get(k) for k in KEEP if k in f} for f in r.feather_ids]}
            n = sum(f.get("segment_status", "segmented") == "segmented" for f in r.feather_ids)
            bad = sum(bool(f.get("needs_review")) for f in r.feather_ids)
            print(f"{v:16} {s:28} feathers {n} flagged {bad} ({results[v][s]['seconds']}s) {r.reason}", flush=True)
            (out / "results.json").write_text(json.dumps({"results": results, "stats": stats}, indent=1, default=str))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
