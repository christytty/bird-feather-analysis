"""Re-run the per-feather checks of a finished run with other settings or another model, on
exactly the run's masks (feather_masks/, saved by the pipeline), and score them against your
hand verdicts (notebooks/feather_verdicts.ipynb).

    python scripts/rejudge.py slice100-masks-20261019 --variant saved \\
        --set '{"judge_mask": "saved", "prompt_add": "trim"}' --sample data/review/verdict_sample_<run>.csv
    python scripts/rejudge.py <run> --compare          # every variant run so far vs the verdicts

--set takes feather_judge overrides (judge_mask, enhance_contrast, panel_height, prompt, ...).
"prompt_add": names from PROMPT_ADD below, added to the configured prompt. --endpoint and --model
point at another VLM server (e.g. a second model on another node). Results: data/review/rejudge/<run>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src import feather_judge as fj, label_extract as lx  # noqa: E402
from src.feather_processing import equalized, load_seg_config, make_feather  # noqa: E402

TRIM = ("The feather's base is trimmed off on purpose: the bottom of the outline may be a straight horizontal line "
        "across the feather. That straight cut is not a fault: answer \"complete\" for the bottom when the only cut "
        "there is that straight line, and ignore everything below it (base, glue, card).\n\n")
PROMPT_ADD = {"trim": ("1. Outline, side by side:", TRIM + "1. Outline, side by side:")}


def judge_cfg(overrides: dict) -> dict:
    cfg = dict(load_seg_config()["feather_judge"])
    adds = overrides.pop("prompt_add", [])
    cfg.update(overrides)
    for name in ([adds] if isinstance(adds, str) else adds):
        old, new = PROMPT_ADD[name]
        assert old in cfg["prompt"], name
        cfg["prompt"] = cfg["prompt"].replace(old, new)
    return cfg


def rejudge(run: str, variant: str, overrides: dict, sample: Path | None, endpoint: str | None,
            model: str | None) -> Path:
    seg = REPO / "data/sample_runs" / run
    fdf = pd.read_csv(REPO / "data/label_runs" / run / "feathers.csv")
    fdf = fdf[(fdf.segment_status == "segmented") & fdf.mask_image.notna()]
    if sample is not None:
        fdf = fdf[fdf.crop.isin(set(pd.read_csv(sample).crop))]
    cfg = judge_cfg(dict(overrides))
    seg_cfg = load_seg_config()
    trim = float(seg_cfg["crop"].get("trim_bottom_frac", 0.12))
    vcfg = dict(lx.load_config()["vlm"])
    if model:
        vcfg["model"] = model
    if endpoint:
        vcfg["endpoints"] = [endpoint]
    client = lx.VLMClient(vcfg)
    if endpoint:
        client._endpoints = iter(lambda: endpoint, None)   # ignore FEATHER_VLM_ENDPOINTS
    out_dir = REPO / "data/review/rejudge" / run / variant
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)
    (out_dir / "settings.json").write_text(json.dumps({"overrides": overrides, "endpoint": endpoint,
                                                       "model": vcfg["model"]}, indent=1))
    done = {}
    res_path = out_dir / "results.jsonl"
    if res_path.exists():
        done = {json.loads(l)["crop"]: l for l in res_path.read_text().splitlines() if l.strip()}
    slides: dict = {}
    with open(res_path, "a") as fh:
        for f in fdf.itertuples():
            if f.crop in done:
                continue
            if f.file_id not in slides:
                slides.clear()
                slides[f.file_id] = cv2.imread(str(REPO / "data/raw_cache" / f.file_id))
            img = slides[f.file_id]
            full = cv2.imread(str(seg / f.mask_image), 0) > 0
            feather = make_feather(img, full, trim, 0)
            res, panel = fj.judge(client, img, feather["mask"] if cfg.get("judge_mask") == "saved" else full,
                                  equalized(feather["crop"]), cfg)
            panel.save(out_dir / "panels" / (Path(f.crop).stem + ".jpg"), quality=85)
            rec = {"judge_" + k: v for k, v in res.items() if k != "raw"}
            row = {"crop": f.crop, "file_id": f.file_id, "status": res.get("status"), "score": res.get("score"),
                   "needs_review": fj.needs_review({**rec, "postprocess": f.postprocess}),
                   "uncertain": fj.uncertain_review(rec), "notes": res.get("notes"), "raw": res.get("raw")}
            fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
            print(variant, f.crop, row["score"], row["needs_review"] or "pass", flush=True)
    return res_path


def same_crop(a: Path, b: Path) -> bool:
    """The same feather crop in two runs (verdicts given on one run apply to the other)."""
    x, y = cv2.imread(str(a)), cv2.imread(str(b))
    return x is not None and y is not None and x.shape == y.shape and float(np.abs(x.astype(int) - y).mean()) < 2


def compare(run: str, verdicts: Path) -> pd.DataFrame:
    """Per variant, on the feathers you judged: bad feathers it flagged (caught), bad ones it passed
    (missed, the costly kind) and good ones it flagged (false alarms). Verdicts from another run
    (verdicts_<other run>.csv) count only where the crop is identical in both runs."""
    v = pd.read_csv(verdicts, dtype=str).fillna("")
    v = v[v.verdict.isin(["good", "bad"])].set_index("crop")
    other = verdicts.stem.removeprefix("verdicts_")
    if other != run:
        keep = [c for c in v.index if same_crop(REPO / "data/sample_runs" / other / c, REPO / "data/sample_runs" / run / c)]
        print(f"verdicts from {other}: {len(keep)} of {len(v)} crops identical in {run}")
        v = v.loc[keep]
    rows = []
    variants = {"run (as shipped)": None, **{p.name: p for p in sorted((REPO / "data/review/rejudge" / run).glob("*"))
                                             if (p / "results.jsonl").exists()}}
    fdf = pd.read_csv(REPO / "data/label_runs" / run / "feathers.csv").fillna({"needs_review": ""})
    for name, p in variants.items():
        if p is None:
            flags = dict(zip(fdf.crop, fdf.needs_review != ""))
        else:
            r = [json.loads(l) for l in (p / "results.jsonl").read_text().splitlines() if l.strip()]
            flags = {x["crop"]: bool(x["needs_review"]) for x in r}
        common = [c for c in v.index if c in flags]
        bad = [c for c in common if v.loc[c, "verdict"] == "bad"]
        good = [c for c in common if v.loc[c, "verdict"] == "good"]
        rows.append({"variant": name, "judged": len(common), "bad caught": sum(flags[c] for c in bad),
                     "bad missed": sum(not flags[c] for c in bad), "good flagged (false alarm)": sum(flags[c] for c in good),
                     "good passed": sum(not flags[c] for c in good),
                     "agreement": f"{sum(flags[c] == (c in bad) for c in common)}/{len(common)}"})
    return pd.DataFrame(rows).set_index("variant")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--variant")
    ap.add_argument("--set", default="{}", help="feather_judge overrides as JSON")
    ap.add_argument("--sample", help="CSV with a crop column: judge only these feathers")
    ap.add_argument("--endpoint")
    ap.add_argument("--model")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--verdicts", help="verdicts CSV (default: data/review/verdicts_<run>.csv)")
    args = ap.parse_args()
    verdicts = Path(args.verdicts) if args.verdicts else REPO / "data/review" / f"verdicts_{args.run}.csv"
    if args.compare:
        print(compare(args.run, verdicts).to_string())
        return 0
    rejudge(args.run, args.variant, json.loads(args.set), Path(args.sample) if args.sample else None,
            args.endpoint, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
