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

REPO = Path(os.getenv("FEATHER_REPO") or Path(__file__).resolve().parent.parent).expanduser()
sys.path.insert(0, str(REPO))

VARIANTS = ("baseline", "exemplar", "refine", "exemplar_refine")
KEEP = ("seq", "crop", "segment_status", "needs_review", "judge_status", "judge_score", "judge_sides_cut",
        "judge_holes_or_gaps", "judge_holes", "judge_hole_frac", "judge_mask_parts", "judge_single_feather",
        "judge_fragment", "judge_other_feather_inside", "judge_other_material_inside", "judge_notes",
        "area_px", "length_px", "width_px", "postprocess", "inventory_status", "inventory_total",
        "feathers_possibly_missed", "missed_between")


def make_variants(p, weights: str, stats: dict):
    """Segmentation functions with the signature of FeatherProcessor._segment_sam3,
    built on src/sam3_extras.py (the same code the pipeline uses when enabled)."""
    import torch
    from ultralytics.models.sam import SAM3Predictor

    from src import sam3_extras as sx

    base = p._segment_sam3
    inter = SAM3Predictor(overrides={"model": weights, "imgsz": 1008, "task": "segment", "mode": "predict",
                                     "device": "mps" if torch.backends.mps.is_available() else "cpu",
                                     "save": False, "verbose": False})

    def exemplar(image_path, img_pil, prompts, nms_iou, max_box_area_ratio, retry=False, conf=None):
        first = base(image_path, img_pil, prompts, nms_iou, max_box_area_ratio, retry, conf)
        out = sx.exemplar_pass(p.sam3, image_path, img_pil.size, prompts[0].rstrip("."), first, {},
                               nms_iou, max_box_area_ratio, conf)
        stats.setdefault("exemplar_counts", {})[Path(image_path).name] = [len(first[0]), len(out[0]), out is first]
        return out

    def refine_with(first_fn):
        def refine(image_path, img_pil, prompts, nms_iou, max_box_area_ratio, retry=False, conf=None):
            first = first_fn(image_path, img_pil, prompts, nms_iou, max_box_area_ratio, retry, conf)
            out, info = sx.refine_pass(inter, image_path, first, {})
            stats.setdefault("refine", {})[Path(image_path).name] = info
            return out
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
