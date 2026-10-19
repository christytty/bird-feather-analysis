"""Per-feather segmentation checks: measured from the mask, and judged by the VLM.

Rules for a good segment (the same for any feather on any slide):
  1. complete outline: no part of this feather lies outside the mask on any
     side (left, right, top, bottom of the image)
  2. no holes or gaps: the mask is one solid piece with nothing of the
     feather missing inside it
  3. exactly one feather, not a fragment, and nothing else inside the mask

Rule 2 is measured directly on the mask (holes = background enclosed by the
mask; parts = separate pieces of the mask). The VLM checks rule 1 per side,
and rules 2-3 by eye, looking at two panels: LEFT the slide around the
feather with the mask outline, RIGHT exactly the pixels inside the mask (the
untrimmed segmentation; the saved crop's bottom trim is not judged here).

Decisions (needs_review, segment_ok, the retry in full_run_sharded) use the
checks themselves. The 1-10 `score` is computed from them with the fixed
deductions in configs/segmentation.json (feather_judge.score), only for
sorting and summaries; the model's own number is kept, unused, as
`model_score`. A failed or garbled answer is status 'model_error' (no score).
"""
from __future__ import annotations

import json
import re

import cv2
import numpy as np
from PIL import Image

SIDES = ("left", "right", "top", "bottom")
BOOLS = ("holes_or_gaps", "single_feather", "fragment", "other_material_inside", "other_feather_inside")
# used when the config has no feather_judge.score (same values as the shipped config)
DEFAULT_SCORE = {"start": 10, "min": 1, "deduct": {"side_cut": 2, "holes_or_gaps": 3, "fragment": 4,
                                                   "not_single_feather": 4, "other_feather_inside": 3,
                                                   "other_material_inside": 2}}


# ---------------------------------------------------------------- measured on the mask

def mask_metrics(mask: np.ndarray, min_hole_frac: float = 0.002, min_part_frac: float = 0.01) -> dict:
    """Holes and separate pieces of one feather's mask (rule 2), measured.
    hole_frac: enclosed background / filled feather area; mask_parts: pieces
    larger than min_part_frac of the mask."""
    m = mask.astype(np.uint8)
    area = int(m.sum())
    if area == 0:
        return {"hole_frac": None, "holes": 0, "mask_parts": 0}
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    parts = int(sum(stats[i, cv2.CC_STAT_AREA] >= min_part_frac * area for i in range(1, n)))
    # holes: background regions not connected to the image border
    inv = (1 - m).astype(np.uint8)
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = m.shape
    border = set(np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]])))
    holes = [stats2[i, cv2.CC_STAT_AREA] for i in range(1, n2) if i not in border]
    big = [a for a in holes if a >= min_hole_frac * area]
    return {"hole_frac": round(float(sum(big)) / (area + sum(big)), 4), "holes": len(big), "mask_parts": parts}


# ---------------------------------------------------------------- what the model sees

def context_panel(slide_bgr: np.ndarray, mask: np.ndarray, cfg: dict) -> Image.Image:
    """LEFT: slide around the feather with the mask outline (and filled holes
    outlined too); RIGHT: the slide pixels inside the mask on black. Same height."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = int(max(x1 - x0, y1 - y0) * cfg.get("context_margin_frac", 0.15)) + 10
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    left = slide_bgr[y0:y1, x0:x1].copy()
    sub = mask[y0:y1, x0:x1].astype(np.uint8)
    cnts, _ = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)   # outer edges and holes
    cv2.drawContours(left, cnts, -1, (0, 255, 0), max(2, int(min(left.shape[:2]) / 300)))
    right = np.where(sub[:, :, None] > 0, slide_bgr[y0:y1, x0:x1], 0).astype(np.uint8)
    ph = int(cfg.get("panel_height", 900))

    def fit(img):
        s = ph / img.shape[0]
        return cv2.resize(img, (max(1, int(img.shape[1] * s)), ph), interpolation=cv2.INTER_AREA)

    both = np.hstack([fit(left), np.full((ph, 16, 3), 255, np.uint8), fit(right)])
    return Image.fromarray(cv2.cvtColor(both, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------- the answer

def _json(text: str) -> dict:
    if re.search(r"\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}", text or ""):
        raise ValueError("escape sequence in answer")
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError(f"no JSON in answer: {(text or '')[:60]!r}")
    return json.loads(m.group(0))


def derive_score(r: dict, score_cfg: dict | None = None) -> int:
    """start minus a fixed deduction per failed check (per cut side), at least min."""
    sc = {**DEFAULT_SCORE, **(score_cfg or {})}
    d = {**DEFAULT_SCORE["deduct"], **sc.get("deduct", {})}
    s = sc["start"]
    s -= d["side_cut"] * len(r.get("sides_cut") or [])
    s -= d["holes_or_gaps"] * bool(r.get("holes_or_gaps"))
    s -= d["fragment"] * bool(r.get("fragment"))
    s -= d["not_single_feather"] * (r.get("single_feather") is False)
    s -= d["other_feather_inside"] * bool(r.get("other_feather_inside"))
    s -= d["other_material_inside"] * bool(r.get("other_material_inside"))
    return max(sc["min"], s)


def parse(text: str, score_cfg: dict | None = None) -> dict:
    """Model answer -> checks, or {'status': 'model_error', 'error': ...}."""
    try:
        d = _json(text)
        outline = d["outline"]
        sides = {s: str(outline[s]).strip().lower() for s in SIDES}
    except (ValueError, KeyError, TypeError) as exc:
        return {"status": "model_error", "error": f"{type(exc).__name__}: {exc}"[:200]}
    if any(v not in ("complete", "cut") for v in sides.values()):
        return {"status": "model_error", "error": f"outline values must be complete|cut: {sides}"}
    out = {"status": "ok", **{f"outline_{s}": v for s, v in sides.items()},
           "sides_cut": [s for s in SIDES if sides[s] == "cut"]}
    for k in BOOLS:
        v = d.get(k)
        if not isinstance(v, bool):
            return {"status": "model_error", "error": f"{k} must be true|false, got {v!r}"}
        out[k] = v
    ms = d.get("model_score", d.get("score"))
    out["model_score"] = float(ms) if isinstance(ms, (int, float)) and 1 <= ms <= 10 else None
    notes = str(d.get("notes") or "")
    out["notes"] = notes if notes.isascii() else notes.encode("ascii", "ignore").decode()
    _finish(out, score_cfg)
    return out


def _finish(res: dict, score_cfg: dict | None) -> None:
    res["score"] = derive_score(res, score_cfg)
    res["segment_ok"] = not failed_checks(res, prefix="")


_PASS = {"holes_or_gaps": False, "single_feather": True, "fragment": False, "other_feather_inside": False,
         "other_material_inside": False}


def _model_failures(res: dict) -> set[str]:
    """The checks the model's answer fails (not the mask measurements)."""
    out = {f"outline_{s}" for s in SIDES if res.get(f"outline_{s}") == "cut"}
    return out | {k for k, ok in _PASS.items() if res.get(k) is not None and res.get(k) != ok}


def _second_look(client, panel, cfg: dict, res: dict) -> None:
    """Asks again on the same panel at a smaller size when the first answer
    fails a check. A check failed only once is set back to passing and listed
    in `uncertain_checks` (kept for review, not counted as a failure); a check
    failed both times stays failed. If the second request fails, the first
    answer stands: nothing is cleared without a second answer."""
    first = _model_failures(res)
    res["uncertain_checks"] = []
    if not first:
        return
    scale = float((cfg.get("second_look") or {}).get("scale", 0.8))
    small = panel.resize((max(1, int(panel.width * scale)), max(1, int(panel.height * scale))), Image.LANCZOS)
    try:
        raw2 = client.ask(small, cfg["prompt"], max_tokens=cfg.get("max_tokens", 220))
    except Exception as exc:  # noqa: BLE001
        res["second_look"] = f"failed: {type(exc).__name__}"
        return
    res2 = parse(raw2, cfg.get("score"))
    res["second_look"] = (raw2 or "")[:300]
    if res2.get("status") != "ok":
        return
    uncertain = sorted(first - _model_failures(res2))
    for c in uncertain:
        res[c] = "complete" if c.startswith("outline_") else _PASS[c]
    res["sides_cut"] = [s for s in SIDES if res.get(f"outline_{s}") == "cut"]
    res["uncertain_checks"] = uncertain


def judge(client, slide_bgr, mask, crop_bgr, cfg: dict) -> tuple[dict, Image.Image]:
    """Measures and asks about one feather. Returns (result, the panel image)."""
    metrics = mask_metrics(mask, cfg.get("min_hole_frac", 0.002))
    panel = context_panel(slide_bgr, mask, cfg)
    try:
        raw = client.ask(panel, cfg["prompt"], max_tokens=cfg.get("max_tokens", 220))
    except Exception as exc:  # noqa: BLE001
        return {"status": "model_error", "error": f"{type(exc).__name__}: {exc}", **metrics}, panel
    res = parse(raw, cfg.get("score"))
    res.update(metrics)
    if res["status"] == "ok" and (cfg.get("second_look") or {}).get("enabled", False):
        _second_look(client, panel, cfg, res)
        _finish(res, cfg.get("score"))
    fail_frac = float(cfg.get("fail_hole_frac", 0.02))
    if res["status"] == "ok" and ((metrics["hole_frac"] or 0) >= fail_frac or metrics["mask_parts"] > 1):
        # a mask in pieces, or measured holes of fail_hole_frac or more, count as rule-2
        # failures even if the model missed them; smaller measured holes are mostly natural
        # vane splits (background showing through) and are only recorded
        res["holes_or_gaps"] = True
        _finish(res, cfg.get("score"))
    res["raw"] = (raw or "")[:400]
    return res, panel


def failed_checks(rec: dict, prefix: str = "judge_") -> list[str]:
    """The checks one feather failed (model answer plus mask measurements);
    keys are judge_* in a feather record, bare in a judge result (prefix='')."""
    g = lambda k: rec.get(prefix + k)  # noqa: E731
    why = []
    sides = g("sides_cut")
    if sides:
        why.append("outline cut: " + ",".join(sides if isinstance(sides, list) else [str(sides)]))
    if g("holes_or_gaps") or (g("mask_parts") or 0) > 1:
        why.append("holes or gaps")
    if g("fragment"):
        why.append("fragment")
    if g("single_feather") is False:
        why.append("not a single feather")
    if g("other_feather_inside"):
        why.append("other feather inside")
    if g("other_material_inside"):
        why.append("non-feather material inside")
    return why


def needs_review(rec: dict) -> str:
    """Why a feather should be looked at ('' if it passed every check). Rows
    that are only review flags (a possible missed feather) keep their reason."""
    if rec.get("segment_status", "segmented") != "segmented":
        return rec.get("needs_review") or str(rec.get("segment_status"))
    why = ["judge failed"] if rec.get("judge_status") == "model_error" else []
    why += failed_checks(rec)
    if "possible fragment" in str(rec.get("postprocess") or ""):
        why.append("small/narrow mask")
    return "; ".join(why)


def is_fragment(rec: dict, prefix: str = "judge_") -> bool:
    """A piece, not a feather: the mask clean-up marked it small or narrow, and
    the model (on two looks) sees a fragment or not a single feather. Such
    pieces get no letter or feather ID; they stay as review rows. Without a
    model answer the piece stays a feather (flagged)."""
    return ("possible fragment" in str(rec.get("postprocess") or "") and rec.get(prefix + "status") == "ok"
            and (rec.get(prefix + "fragment") is True or rec.get(prefix + "single_feather") is False))


def uncertain_review(rec: dict, prefix: str = "judge_") -> str:
    """Checks failed on one look only (second_look): listed for review after
    the confirmed failures, not counted in the score or the retry."""
    names = {"outline_left": "outline cut: left", "outline_right": "outline cut: right",
             "outline_top": "outline cut: top", "outline_bottom": "outline cut: bottom",
             "holes_or_gaps": "holes or gaps", "single_feather": "not a single feather", "fragment": "fragment",
             "other_feather_inside": "other feather inside", "other_material_inside": "non-feather material inside"}
    return "; ".join(names.get(c, c) for c in rec.get(prefix + "uncertain_checks") or [])


def slide_checks(records: list[dict]) -> dict:
    """Slide-level view of the per-feather checks and the missed-feather flags
    (what the retired slide judge used to guess from the overlay)."""
    seg = [r for r in records if r.get("segment_status", "segmented") == "segmented"]
    judged = [r for r in seg if r.get("judge_status") == "ok"]
    inv = next((r.get("inventory_status") for r in records if r.get("inventory_status")), None)
    scores = [r["judge_score"] for r in judged if isinstance(r.get("judge_score"), (int, float))]
    return {
        "feathers_failing": sum(bool(failed_checks(r)) for r in judged),
        "checks_failed": sum(len(failed_checks(r)) for r in judged),
        "judge_errors": sum(r.get("judge_status") == "model_error" for r in seg),
        "possible_missed": sum(r.get("segment_status") == "possible_missed" for r in records),
        "fragments": sum(r.get("segment_status") == "fragment" for r in records),
        "all_feathers_covered": {"complete": True, "missed": False}.get(inv),
        "merged_feathers": (any(r.get("judge_single_feather") is False or r.get("judge_other_feather_inside")
                                for r in judged) if judged else None),
        "other_material": any(bool(r.get("judge_other_material_inside")) for r in judged) if judged else None,
        "score_mean": round(sum(scores) / len(scores), 2) if scores else None,
    }
