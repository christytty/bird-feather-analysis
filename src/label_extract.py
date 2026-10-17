"""Specimen label extraction: the second output table.

Reads the label fields (individual ID, species, age, sex, feather line, date)
from the ORIGINAL slide image, not the segmentation overlay. Each field gets
its own crop (configs/label_fields.json) and its own constrained VLM prompt, so
one hard field (the feather line, often under the black board) cannot spoil
the others.

Every field comes back with a status, so a missing value always says why:
  ok              read and passed the format check
  partial         only part of the text is visible (feather line)
  not_visible     the model saw no such text in the crop
  invalid_format  something was read but fails the field's pattern -> review
  mismatch_filename  disagrees with the ID/year in the file name -> review
  error           the request failed; this is NOT evidence the label is missing

The VLM runs on an Apple-silicon (MLX) node behind an OpenAI-compatible server
(`pixi run vlm-server`), so this works from any machine. Endpoints come from
FEATHER_VLM_ENDPOINTS (comma-separated) or configs/label_fields.json.

CLI:
    python -m src.label_extract --sample 10 --seed 1          # random sample
    python -m src.label_extract --match "A0012 1996" --match BB17692
    python -m src.label_extract --all --workers 4 --out data/label_runs/full
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from itertools import cycle
from pathlib import Path
from threading import Lock

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "configs" / "label_fields.json"
FIELDS_ORDER = ["individual_id", "species", "age", "sex", "feather", "date"]


@dataclass
class FieldResult:
    value: str | None
    status: str
    confidence: float | None = None
    raw: str = ""
    seconds: float = 0.0


@dataclass
class LabelResult:
    image: str
    fields: dict[str, FieldResult] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return any(f.status in ("invalid_format", "error", "partial", "mismatch_filename")
                   for f in self.fields.values())

    def row(self) -> dict:
        out = {"image": Path(self.image).name, "needs_review": self.needs_review}
        for name in FIELDS_ORDER:
            f = self.fields.get(name)
            if f is None:
                continue
            out[name] = f.value
            out[f"{name}_status"] = f.status
            out[f"{name}_confidence"] = f.confidence
        return out


def load_config(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_CONFIG).read_text())


def crop_field(image: Image.Image, region: list[float], max_side: int) -> Image.Image:
    w, h = image.size
    l, t, r, b = region
    crop = image.crop((int(l * w), int(t * h), int(r * w), int(b * h))).convert("RGB")
    # Upscale small crops (the model reads small print better), cap big ones.
    scale = max_side / max(crop.size)
    if scale > 1 or max(crop.size) > max_side:
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
    return crop


def _data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class VLMClient:
    """Round-robins requests over the configured endpoints."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        env = [e.strip() for e in os.getenv("FEATHER_VLM_ENDPOINTS", "").split(",") if e.strip()]
        self._endpoints = cycle(env or cfg["endpoints"])
        self._lock = Lock()

    def ask(self, image: Image.Image, prompt: str, timeout: float = 180) -> str:
        with self._lock:
            base = next(self._endpoints)
        body = {
            "model": self.cfg["model"],
            "max_tokens": self.cfg.get("max_tokens", 120),
            "temperature": self.cfg.get("temperature", 0),
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _data_url(image)}},
                {"type": "text", "text": prompt},
            ]}],
        }
        req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"]


def normalize(value: str | None, how: str | None) -> str | None:
    if value is None:
        return None
    v = str(value).strip()
    if how == "remove_spaces_upper":
        v = re.sub(r"\s+", "", v).upper()
    elif how == "date":
        m = re.search(r"(\d{4})\D?(\d{1,2})\D?(\d{1,2})", v)
        if m:
            v = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return v or None


def parse_answer(text: str) -> tuple[str | None, str, float | None]:
    """Returns (value, visible, confidence) from the model's JSON answer."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            conf = d.get("confidence")
            return (d.get("value"), str(d.get("visible", "yes")).lower(),
                    float(conf) if isinstance(conf, (int, float)) else None)
        except (ValueError, TypeError):
            pass
    # Not JSON: treat the whole reply as the value, confidence unknown.
    t = text.strip().strip("`").strip()
    return (t or None, "yes" if t else "no", None)


def read_field(client: VLMClient, image: Image.Image, name: str, spec: dict, cfg: dict) -> FieldResult:
    crop = crop_field(image, spec["region"], cfg["vlm"].get("crop_max_side", 1024))
    prompt = spec["prompt"] + "\n\n" + cfg["answer_rules"]
    t = time.monotonic()
    try:
        raw = client.ask(crop, prompt)
    except Exception as exc:  # noqa: BLE001
        return FieldResult(None, "error", raw=f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - t)
    secs = time.monotonic() - t
    value, visible, conf = parse_answer(raw)
    value = normalize(value, spec.get("normalize"))
    if value is None or visible == "no":
        return FieldResult(None, "not_visible", conf, raw, secs)
    if visible == "partly" or "..." in value:
        return FieldResult(value, "partial", conf, raw, secs)
    pattern = spec.get("pattern")
    if pattern and not re.match(pattern, value):
        return FieldResult(value, "invalid_format", conf, raw, secs)
    return FieldResult(value, "ok", conf, raw, secs)


FILENAME_RX = re.compile(r"^([A-Z]{1,2})\s*(\d{3,5})\s+((?:19|20)\d{2})")


def check_against_filename(result: LabelResult) -> None:
    """Slide files are named like 'A0012 1996-im1058.jpg' (ID, year): an
    independent source for the ID and the date's year. Disagreement means one
    of them is misread, so the field goes to review instead of passing."""
    m = FILENAME_RX.match(Path(result.image).name)
    if not m:
        return
    fid, year = m.group(1) + m.group(2), m.group(3)
    f = result.fields.get("individual_id")
    if f and f.status == "ok" and not (f.value or "").startswith(fid):
        f.status = "mismatch_filename"
    f = result.fields.get("date")
    if f and f.status == "ok" and not (f.value or "").startswith(year):
        f.status = "mismatch_filename"


def extract(image_path: str | Path, cfg: dict | None = None, client: VLMClient | None = None,
            fields: list[str] | None = None) -> LabelResult:
    cfg = cfg or load_config()
    client = client or VLMClient(cfg["vlm"])
    image = Image.open(image_path)
    result = LabelResult(str(image_path))
    for name in fields or FIELDS_ORDER:
        spec = cfg["fields"].get(name)
        if spec:
            result.fields[name] = read_field(client, image, name, spec, cfg)
    check_against_filename(result)
    return result


def field_crops(image_path: str | Path, cfg: dict | None = None) -> dict[str, Image.Image]:
    """The exact crops the model sees, for inspecting/tuning regions."""
    cfg = cfg or load_config()
    image = Image.open(image_path)
    return {n: crop_field(image, s["region"], cfg["vlm"].get("crop_max_side", 1024))
            for n, s in cfg["fields"].items() if not n.startswith("_")}


def write_csv(results: list[LabelResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.row() for r in results]
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def score_against_gold(results: list[LabelResult], gold_csv: str | Path) -> dict[str, dict]:
    """Per-field accuracy against a hand-labelled CSV (image + field columns)."""
    gold = {}
    with open(gold_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            gold[row["image"]] = row
    stats: dict[str, dict] = {}
    for r in results:
        g = gold.get(Path(r.image).name)
        if not g:
            continue
        for name, f in r.fields.items():
            want = (g.get(name) or "").strip()
            if not want:
                continue
            s = stats.setdefault(name, {"n": 0, "correct": 0, "wrong": []})
            s["n"] += 1
            want_norm = normalize(want, load_config()["fields"][name].get("normalize"))
            if (f.value or "") == (want_norm or ""):
                s["correct"] += 1
            else:
                s["wrong"].append((Path(r.image).name, f.value, want_norm, f.status))
    for s in stats.values():
        s["accuracy"] = round(s["correct"] / s["n"], 3) if s["n"] else None
    return stats


def _select(images: list[str], args) -> list[str]:
    if args.match:
        return [p for p in images if any(m in Path(p).name for m in args.match)]
    if args.all:
        return images
    rnd = random.Random(args.seed)
    return rnd.sample(images, min(args.sample, len(images)))


def main(argv: list[str] | None = None) -> int:
    from src.notebook_remote_runtime import RemoteClusterIO, load_notebook_config_from_env

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match", action="append", help="substring of the image name (repeatable)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--workers", type=int, default=2, help="images in flight at once")
    ap.add_argument("--out", default=None, help="output dir (default data/label_runs/<timestamp>)")
    ap.add_argument("--gold", default=None, help="hand-labelled CSV to score against")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    client = VLMClient(cfg["vlm"])
    rc = load_notebook_config_from_env()
    io_ = RemoteClusterIO(rc)
    chosen = _select(io_.list_remote_images(), args)
    out = Path(args.out or REPO / "data" / "label_runs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    cache = REPO / "data" / "raw_cache"
    print(f"{len(chosen)} images -> {out}")

    def one(p: str) -> LabelResult:
        local = cache / Path(p).name
        if not local.exists():
            io_.fetch_remote_file(rc.head_host, p, cache)
        r = extract(local, cfg, client)
        flag = " REVIEW" if r.needs_review else ""
        print(f"  {local.name}: " + ", ".join(f"{k}={v.value!r}/{v.status}" for k, v in r.fields.items()) + flag,
              flush=True)
        return r

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(one, chosen))
    write_csv(results, out / "labels.csv")
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out / 'labels.csv'}  ({sum(r.needs_review for r in results)} need review)")
    if args.gold:
        for name, s in score_against_gold(results, args.gold).items():
            print(f"  {name:14} {s['correct']}/{s['n']}  {s['accuracy']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
