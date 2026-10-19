"""Label table wiring in the distributed pipeline, with a stub processor
(no SAM/MLX models needed)."""
import json
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from src import label_extract as lx

feather_pkgs = pytest.importorskip("redis")


class StubResult:
    def __init__(self, path, success):
        self.image_path, self.success = path, success
        self.reason = "" if success else "no_boxes"
        self.feathers_saved = 3 if success else 0
        self.vlm_score = None
        self.metadata_source = "filename"
        self.bird_id, self.date = "A0012", "1996"
        self.vlm_all_feathers_covered = None
        self.vlm_background_leakage_detected = None
        self.vlm_green_boxes_grouped_feathers = None
        self.vlm_notes = ""
        self.processing_profile = "default"
        self.feather_ids = []
        self.qa_checks = {}


class StubProcessor:
    """Segmentation fails on purpose: labels must still be persisted."""
    calls = []

    def __init__(self):
        self.vlm_model_id = "stub"

    def extract_labels(self, image_path, output_dir):
        StubProcessor.calls.append(("labels", Path(image_path).name))
        r = lx.LabelResult(image_path, filename=lx.parse_filename(Path(image_path).name), node="stub-node")
        r.fields["bird_id"] = lx.FieldResult("A0012H", "readable", source="primary")
        r.fields["sex"] = lx.FieldResult(None, "model_error")
        return r

    def process_image(self, image_path, output_dir, profile="default", labels=None, options=None):
        StubProcessor.calls.append(("segment", Path(image_path).name))
        ok = Path(image_path).name.startswith("BB")   # first slide fails, second gets 2 feathers
        res = StubResult(image_path, success=ok)
        if ok and labels is not None:
            res.feather_ids = lx.assign_feather_ids(labels, ["c_Feather_1.jpg", "c_Feather_2.jpg"])
        return res


class DateClient:
    """Fake VLM: a fixed date for label prompts, a fixed verdict for the feather
    judge, and a fixed inventory (total feathers + missed list) for the slide check."""
    model_name = "fake"

    def __init__(self, date="2000-01-01", cut=(), holes=False, total=None, missed=(), fragment=False):
        self.date, self.cut, self.holes, self.total, self.missed = date, list(cut), holes, total, list(missed)
        self.fragment = fragment
        self.judge_calls = 0

    def ask(self, image, prompt, timeout=None, max_tokens=None):
        if "Outline, side by side" in prompt:
            self.judge_calls += 1
            outline = {s: ("cut" if s in self.cut else "complete") for s in ("left", "right", "top", "bottom")}
            return json.dumps({"outline": outline, "holes_or_gaps": self.holes, "single_feather": True,
                               "fragment": self.fragment, "other_feather_inside": False, "other_material_inside": False,
                               "score": 9, "notes": "ok"})
        if "total_feathers" in prompt:
            return json.dumps({"total_feathers": self.total, "missed": self.missed, "notes": ""})
        return json.dumps({"value": self.date, "visible": "yes", "confidence": 0.99})


def test_sharded_run_persists_labels_when_segmentation_fails(tmp_path, monkeypatch):
    from src import full_run_sharded as frs

    raw = tmp_path / "raw"
    raw.mkdir()
    for n in ("A0012 1996-im1058.jpg", "BB17280 1992-im0412.jpg"):
        Image.new("RGB", (40, 46)).save(raw / n)
    monkeypatch.setattr(frs, "FeatherProcessor", StubProcessor)
    monkeypatch.setattr(frs, "_connect_metrics_redis", lambda: None)
    monkeypatch.setenv("FEATHER_RUN_ID", "t1")
    monkeypatch.setenv("FEATHER_NODE_ID", "node-a")
    monkeypatch.setenv("FEATHER_METRICS_DIR", str(tmp_path / "metrics"))
    frs.run_shard(str(raw), str(tmp_path / "out"), 0, 1, 0, None)

    assert StubProcessor.calls[:2] == [("labels", "A0012 1996-im1058.jpg"), ("segment", "A0012 1996-im1058.jpg")]
    db = sqlite3.connect(tmp_path / "metrics" / "run_stats.sqlite3")
    rows = db.execute("select run_id, node_id, file_id, bird_id, sex, sex_status "
                      "from specimen_labels order by file_id").fetchall()
    assert rows == [("t1", "node-a", "A0012 1996-im1058.jpg", "A0012H", "", "model_error"),
                    ("t1", "node-a", "BB17280 1992-im0412.jpg", "A0012H", "", "model_error")]
    assert db.execute("select count(*) from image_stats where success=0").fetchone()[0] == 1
    feathers = db.execute("select file_id, seq, feather_id, year_source from specimen_feathers order by seq").fetchall()
    assert feathers == [("BB17280 1992-im0412.jpg", "A", "A0012H_1992_A", "filename"),
                        ("BB17280 1992-im0412.jpg", "B", "A0012H_1992_B", "filename")]
    assert db.execute("select feather_ids from specimen_labels where file_id like 'BB%'").fetchone()[0] \
        == "A0012H_1992_A;A0012H_1992_B"
    state = json.loads((tmp_path / "metrics" / "t1_node-a.json").read_text())
    assert state["labels_done"] == 2 and state["labels_model_error"] == 2
    assert state["labels_bird_id_readable"] == 2
    events = (tmp_path / "metrics" / "t1_node-a_events.jsonl").read_text().splitlines()
    assert json.loads(events[0])["labels"]["sex"] == "model_error"


def test_celery_process_image_returns_labels(monkeypatch, tmp_path):
    from src import celery_tasks as ct

    monkeypatch.setattr(ct, "_PROCESSOR", StubProcessor())
    out = ct.process_image.run("/x/A0012 1996-im1058.jpg", str(tmp_path))
    assert out["success"] is False and out["labels"]["fields"]["bird_id"]["value"] == "A0012H"


def test_distributed_driver_writes_label_table(tmp_path):
    from src.full_run_distributed import _write_labels

    r = StubProcessor().extract_labels("/x/A0012 1996-im1058.jpg", None)
    _write_labels([{"success": False, "labels": r.to_json()}, {"success": True, "labels": None}],
                  str(tmp_path / "labels"))
    rows = (tmp_path / "labels" / "labels.csv").read_text().splitlines()
    assert len(rows) == 2 and "A0012H" in rows[1]
    assert (tmp_path / "labels" / "labels.sqlite3").exists()


def test_process_image_writes_new_names_left_to_right(tmp_path):
    """Real FeatherProcessor.process_image with fake masks instead of SAM."""
    import numpy as np
    from src.feather_processing import FeatherProcessor

    raw = tmp_path / "raw"
    raw.mkdir()
    slide = raw / "CE01168 2008-im2346.jpg"
    Image.new("RGB", (300, 200), "white").save(slide)
    (raw / "A0012 1996-im1058.jpg").write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()
    # leftovers from an earlier attempt (old and new names) must be replaced
    (out / "CE01168 2008-im2346_CE01168_2008_Feather_7.jpg").write_bytes(b"x")
    (out / "CE01168_2008_Feather_E.jpg").write_bytes(b"x")

    def masks_at(xs):
        ms = []
        for x in xs:
            m = np.zeros((200, 300), dtype=np.float32)
            m[20:180, x:x + 30] = 1
            ms.append(m)
        return ms

    p = object.__new__(FeatherProcessor)
    p.segmenter, p.has_vlm = "sam3", False
    p._label_client = DateClient()
    p._segment_sam3 = lambda *a, **k: ([[200, 20, 230, 180], [20, 20, 50, 180], [110, 20, 140, 180]],
                                       masks_at([200, 20, 110]), "")
    labels = lx.LabelResult(str(slide), filename=lx.parse_filename(slide.name))
    labels.fields["bird_id"] = lx.FieldResult("CE01168V", "readable")
    labels.fields["date"] = lx.FieldResult("2008-05-22", "readable")
    res = p.process_image(str(slide), str(out), labels=labels)

    assert res.success and res.feathers_saved == 3
    names = sorted(f.name for f in out.iterdir() if f.is_file())
    assert sorted(p.name for p in (out / "feather_masks").iterdir()) == [
        "CE01168_2008_Feather_A.png", "CE01168_2008_Feather_B.png", "CE01168_2008_Feather_C.png"]
    assert names == ["CE01168 2008_BoundingBoxes.jpg", "CE01168_2008_Feather_A.jpg",
                     "CE01168_2008_Feather_B.jpg", "CE01168_2008_Feather_C.jpg"]
    assert all(f["judge_status"] == "ok" and f["judge_score"] == 10 and f["needs_review"] == "" for f in res.feather_ids)
    assert all((out / f["qc_image"]).exists() for f in res.feather_ids)
    import cv2
    m = cv2.imread(str(out / res.feather_ids[0]["mask_image"]), 0)       # the untrimmed mask, slide-sized
    assert m.shape == (200, 300) and (m > 0).sum() == 160 * 30
    assert [(f["feather_id"], f["crop"]) for f in res.feather_ids] == [
        ("CE01168V_2008_A", "CE01168_2008_Feather_A.jpg"), ("CE01168V_2008_B", "CE01168_2008_Feather_B.jpg"),
        ("CE01168V_2008_C", "CE01168_2008_Feather_C.jpg")]
    # leftmost mask (x=20) became crop A
    import cv2
    a = cv2.imread(str(out / "CE01168_2008_Feather_A.jpg"))
    assert a is not None and a.shape[1] <= 31


@pytest.mark.parametrize("sibling_date,expect_prefix", [
    ("1998-06-20", "BB17692_19980612"),           # same year, other day -> date only
    ("1998-06-12", "BB17692_19980612-im0513"),    # same day -> date + image number
])
def test_process_image_names_same_year_slides_by_label_date(tmp_path, sibling_date, expect_prefix):
    import numpy as np
    from src.feather_processing import FeatherProcessor

    raw = tmp_path / "raw"
    raw.mkdir()
    for n in ("BB17692 1998-im0513.jpg", "BB17692 1998-im0514.jpg"):
        Image.new("RGB", (300, 200), "white").save(raw / n)
    out = tmp_path / "out"
    m = np.zeros((200, 300), dtype=np.float32)
    m[20:180, 40:70] = 1
    p = object.__new__(FeatherProcessor)
    p.segmenter, p.has_vlm = "sam3", False
    p._label_client = DateClient()
    p._segment_sam3 = lambda *a, **k: ([[40, 20, 70, 180]], [m], "")
    p._label_client = DateClient(sibling_date)          # the sibling's date reading
    name = "BB17692 1998-im0513.jpg"
    labels = lx.LabelResult(str(raw / name), filename=lx.parse_filename(name))
    labels.fields["bird_id"] = lx.FieldResult("BB17692H", "readable")
    labels.fields["date"] = lx.FieldResult("1998-06-12", "readable")
    res = p.process_image(str(raw / name), str(out), labels=labels)
    assert res.success, res.reason
    assert (out / f"{expect_prefix}_Feather_A.jpg").exists()
    assert res.feather_ids[0]["feather_id"] == f"BB17692H_{expect_prefix.split('_', 1)[1]}_A"
    assert (out / "labels" / "BB17692 1998-im0514.date.json").exists()   # sibling date cached


def test_process_image_unreadable_own_date_uses_image_number(tmp_path):
    import numpy as np
    from src.feather_processing import FeatherProcessor

    raw = tmp_path / "raw"
    raw.mkdir()
    for n in ("BB17692 1998-im0513.jpg", "BB17692 1998-im0514.jpg"):
        Image.new("RGB", (300, 200), "white").save(raw / n)
    m = np.zeros((200, 300), dtype=np.float32)
    m[20:180, 40:70] = 1
    p = object.__new__(FeatherProcessor)
    p.segmenter, p.has_vlm = "sam3", False
    p._label_client = DateClient()
    p._segment_sam3 = lambda *a, **k: ([[40, 20, 70, 180]], [m], "")
    name = "BB17692 1998-im0514.jpg"
    labels = lx.LabelResult(str(raw / name), filename=lx.parse_filename(name))
    labels.fields["bird_id"] = lx.FieldResult("BB17692H", "readable")
    labels.fields["date"] = lx.FieldResult(None, "not_found")
    res = p.process_image(str(raw / name), str(tmp_path / "out"), labels=labels)
    assert res.feather_ids[0]["feather_id"] == "BB17692H_1998-im0514_A"
    assert (tmp_path / "out" / "BB17692_1998-im0514_Feather_A.jpg").exists()



def test_process_image_merges_a_sliver_into_its_feather(tmp_path):
    """A3818 case: SAM returns one vane as a separate thin mask -> one crop, not two."""
    import numpy as np
    from src.feather_processing import FeatherProcessor

    raw = tmp_path / "raw"
    raw.mkdir()
    slide = raw / "A3818 2002-im1708.jpg"
    Image.new("RGB", (400, 300), "white").save(slide)
    full = [np.zeros((300, 400), np.float32) for _ in range(3)]
    full[0][30:270, 40:80] = 1      # feather A
    full[1][30:270, 200:240] = 1    # feather B
    full[2][60:220, 194:200] = 1    # sliver touching B's left edge
    p = object.__new__(FeatherProcessor)
    p.segmenter, p.has_vlm = "sam3", False
    p._label_client = client = DateClient(cut=["left"], total=2)
    p._segment_sam3 = lambda *a, **k: ([[40, 30, 80, 270], [200, 30, 240, 270], [194, 60, 200, 220]],
                                       full, "", [0.9, 0.9, 0.4])
    res = p.process_image(str(slide), str(tmp_path / "out"))
    # each feather fails a check on the first look, so each gets a second look (confirmed)
    assert res.feathers_saved == 2 and client.judge_calls == 4
    b = res.feather_ids[1]
    assert b["sam_masks"] == 2 and "fragment touching" in b["postprocess"]
    assert b["judge_score"] == 8 and b["needs_review"] == "outline cut: left"   # surfaced per feather


def test_options_override_the_config(tmp_path):
    import numpy as np
    from src.feather_processing import FeatherProcessor

    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (400, 300), "white").save(raw / "A0001 1999-im0001.jpg")
    m = np.zeros((300, 400), np.float32)
    m[30:270, 40:80] = 1
    p = object.__new__(FeatherProcessor)
    p.segmenter, p.has_vlm = "sam3", False
    p._label_client = client = DateClient()
    seen = {}
    def seg(*a, **k):
        seen["conf"] = k.get("conf")
        return [[40, 30, 80, 270]], [m], "", [0.9]
    p._segment_sam3 = seg
    res = p.process_image(str(raw / "A0001 1999-im0001.jpg"), str(tmp_path / "out"),
                          options={"segmentation": {"sam3": {"conf": 0.4}, "feather_judge": {"enabled": False}}})
    assert seen["conf"] == 0.4 and client.judge_calls == 0 and res.feather_ids[0]["judge_status"] == "disabled"


def test_sharded_run_writes_per_feather_table(tmp_path, monkeypatch):
    import csv as _csv
    from src import full_run_sharded as frs

    raw = tmp_path / "raw"
    raw.mkdir()
    for n in ("A0012 1996-im1058.jpg", "BB17280 1992-im0412.jpg"):
        Image.new("RGB", (40, 46)).save(raw / n)
    monkeypatch.setattr(frs, "FeatherProcessor", StubProcessor)
    monkeypatch.setattr(frs, "_connect_metrics_redis", lambda: None)
    monkeypatch.setenv("FEATHER_RUN_ID", "t2")
    monkeypatch.setenv("FEATHER_NODE_ID", "node-a")
    monkeypatch.setenv("FEATHER_METRICS_DIR", str(tmp_path / "metrics"))
    frs.run_shard(str(raw), str(tmp_path / "out"), 0, 1, 0, None)
    rows = list(_csv.DictReader(open(tmp_path / "metrics" / "t2_node-a_feathers.csv")))
    assert [r["feather_id"] for r in rows] == ["A0012H_1992_A", "A0012H_1992_B"]
    assert rows[0]["file_id"] == "BB17280 1992-im0412.jpg" and rows[0]["bird_id"] == "A0012H"
    assert rows[0]["run_id"] == "t2" and rows[0]["node_id"] == "node-a"
    assert set(lx.FEATHER_TABLE_COLS) == set(rows[0])


def test_feather_table_row_has_label_and_segmentation_fields():
    r = lx.LabelResult("A0012 1996-im1058.jpg", filename=lx.parse_filename("A0012 1996-im1058.jpg"))
    r.fields["bird_id"] = lx.FieldResult("A0012H", "readable")
    r.fields["species"] = lx.FieldResult("HAFLU", "readable")
    r.fields["sex"] = lx.FieldResult(None, "not_found")
    r.fields["date"] = lx.FieldResult("1996-06-04", "readable")
    ids = lx.assign_feather_ids(r, ["A0012_1996_Feather_A.jpg"])
    r.feathers = [{**ids[0], "bbox": [1, 2, 3, 4], "judge_status": "ok", "judge_score": 8.0,
                   "needs_review": "", "postprocess": "merged fragment touching [2]"}]
    row = lx.feather_table_rows(r, "run1")[0]
    assert row["feather_id"] == "A0012H_1996_A" and row["species"] == "HAFLU" and row["sex"] == "N/A"
    assert (row["year"], row["month"], row["day"]) == ("1996", "06", "04")
    assert row["bbox"] == "[1, 2, 3, 4]" and row["judge_score"] == 8.0 and row["postprocess"].startswith("merged")


def test_distributed_driver_writes_feathers_without_labels(tmp_path):
    import csv as _csv
    from src.full_run_distributed import _write_labels

    _write_labels([{"image_path": "/x/A0012 1996-im1058.jpg", "success": True, "labels": None,
                    "feather_ids": [{"seq": "A", "crop": "A0012_1996_Feather_A.jpg", "judge_score": 9.0}]}],
                  str(tmp_path / "labels"))
    rows = list(_csv.DictReader(open(tmp_path / "labels" / "feathers.csv")))
    assert rows[0]["crop"] == "A0012_1996_Feather_A.jpg" and rows[0]["judge_score"] == "9.0"
    assert rows[0]["bird_id"] == "A0012"        # backup from the file name



def test_missed_feather_is_a_review_row_and_letters_stay(tmp_path):
    """Slide with 3 feathers, segmentation found the 1st and 3rd. The model's
    inventory adds a review row between A and B; letters, crop names and
    feather IDs come from the segmentation alone (A, B)."""
    import numpy as np
    from src.feather_processing import FeatherProcessor

    raw = tmp_path / "raw"
    raw.mkdir()
    slide = raw / "CE01168 2008-im2346.jpg"
    Image.new("RGB", (400, 300), "white").save(slide)
    ms = [np.zeros((300, 400), np.float32) for _ in range(2)]
    ms[0][30:270, 40:80] = 1
    ms[1][30:270, 300:340] = 1
    p = object.__new__(FeatherProcessor)
    p.segmenter, p.has_vlm = "sam3", False
    p._label_client = DateClient(total=3, missed=[{"between": [1, 2], "x_center": 0.48, "partly_hidden": False}])
    p._segment_sam3 = lambda *a, **k: ([[40, 30, 80, 270], [300, 30, 340, 270]], ms, "", [0.9, 0.9])
    labels = lx.LabelResult(str(slide), filename=lx.parse_filename(slide.name))
    labels.fields["bird_id"] = lx.FieldResult("CE01168V", "readable")
    labels.fields["date"] = lx.FieldResult("2008-05-22", "readable")
    res = p.process_image(str(slide), str(tmp_path / "out"), labels=labels)
    got = [(f["seq"], f["segment_status"], f["crop"], f["feather_id"]) for f in res.feather_ids]
    assert got == [("A", "segmented", "CE01168_2008_Feather_A.jpg", "CE01168V_2008_A"),
                   ("B", "segmented", "CE01168_2008_Feather_B.jpg", "CE01168V_2008_B"),
                   ("missed_1", "possible_missed", "", "")]
    flag = res.feather_ids[2]
    assert "between A and B" in flag["needs_review"] and flag["inventory_status"] == "missed"
    assert res.feathers_saved == 2 and res.qa_checks["possible_missed"] == 1
    assert res.vlm_all_feathers_covered is False             # slide-level field derived from the inventory
    assert sorted(f.name for f in (tmp_path / "out").glob("*_Feather_*.jpg")) == \
        ["CE01168_2008_Feather_A.jpg", "CE01168_2008_Feather_B.jpg"]
    row = labels.row()
    assert row["feather_count"] == 2 and row["feather_ids"] == "CE01168V_2008_A;CE01168V_2008_B"
    assert row["possible_missed"] == 1
    rows = lx.feather_table_rows(labels, "r")
    assert rows[2]["segment_status"] == "possible_missed" and rows[2]["crop"] == ""
    assert rows[2]["feathers_possibly_missed"] == 1


def _fragment_slide(tmp_path, client):
    """Three feathers and a small isolated piece between A and B."""
    import numpy as np
    from src.feather_processing import FeatherProcessor

    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    slide = raw / "CE01168 2008-im2346.jpg"
    Image.new("RGB", (400, 300), "white").save(slide)
    ms = [np.zeros((300, 400), np.float32) for _ in range(4)]
    boxes = [[40, 30, 80, 270], [140, 120, 160, 160], [200, 30, 240, 270], [300, 30, 340, 270]]
    for m, (x0, y0, x1, y1) in zip(ms, boxes):
        m[y0:y1, x0:x1] = 1
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    (out / "CE01168_2008_Fragment_2.jpg").write_bytes(b"x")     # from an earlier attempt
    p = object.__new__(FeatherProcessor)
    p.segmenter, p.has_vlm = "sam3", False
    p._label_client = client
    p._segment_sam3 = lambda *a, **k: (boxes, ms, "", [0.9] * 4)
    labels = lx.LabelResult(str(slide), filename=lx.parse_filename(slide.name))
    labels.fields["bird_id"] = lx.FieldResult("CE01168V", "readable")
    labels.fields["date"] = lx.FieldResult("2008-05-22", "readable")
    return p.process_image(str(slide), str(out), labels=labels), labels, out


def test_fragment_gets_no_letter_or_feather_id(tmp_path):
    """A piece the clean-up marks small and the model (twice) calls a fragment
    is set aside: letters A, B, C skip it, it keeps a crop and QC image."""
    res, labels, out = _fragment_slide(tmp_path, DateClient(fragment=True))
    got = [(f["seq"], f["segment_status"], f["crop"], f["feather_id"]) for f in res.feather_ids]
    assert got == [("A", "segmented", "CE01168_2008_Feather_A.jpg", "CE01168V_2008_A"),
                   ("B", "segmented", "CE01168_2008_Feather_B.jpg", "CE01168V_2008_B"),
                   ("C", "segmented", "CE01168_2008_Feather_C.jpg", "CE01168V_2008_C"),
                   ("fragment_1", "fragment", "CE01168_2008_Fragment_1.jpg", "")]
    frag = res.feather_ids[3]
    assert frag["needs_review"].startswith("fragment") and (out / frag["qc_image"]).exists()
    assert frag["qc_image"] == "feather_qc/CE01168_2008_Fragment_1_qc.jpg"
    # whole feathers are never fragments, even when the model says so (flagged instead)
    assert res.feather_ids[0]["needs_review"] == "fragment"
    assert res.feathers_saved == 3 and res.qa_checks["fragments"] == 1
    assert sorted(f.name for f in out.glob("*_Fragment_*.jpg")) == ["CE01168_2008_Fragment_1.jpg"]
    row = labels.row()
    assert row["feather_count"] == 3 and row["feather_ids"] == "CE01168V_2008_A;CE01168V_2008_B;CE01168V_2008_C"
    rows = lx.feather_table_rows(labels, "r")
    assert [(r["seq"], r["usable"], r["excluded_because"]) for r in rows] == [
        ("A", "no", "check: fragment"), ("B", "no", "check: fragment"), ("C", "no", "check: fragment"),
        ("fragment_1", "no", "fragment")]
    assert [p["kind"] for p in lx.slide_problems(labels, None, {})].count("fragment") == 1


def test_small_piece_the_model_calls_a_feather_stays_a_feather(tmp_path):
    res, labels, out = _fragment_slide(tmp_path, DateClient())
    assert [f["seq"] for f in res.feather_ids] == ["A", "B", "C", "D"]
    b = res.feather_ids[1]
    assert b["segment_status"] == "segmented" and b["needs_review"] == "small/narrow mask"
    assert [r["usable"] for r in lx.feather_table_rows(labels, "r")] == ["yes", "no", "yes", "yes"]
    assert not list(out.glob("*_Fragment_*.jpg"))      # the old fragment crop was removed


def test_usable_feathers_and_filtered_export(tmp_path):
    r = lx.LabelResult("raw/A0012 1996-im1058.jpg", filename=lx.parse_filename("A0012 1996-im1058.jpg"))
    r.feathers = [
        {"seq": "A", "feather_id": "A0012_1996_A", "segment_status": "segmented", "needs_review": "",
         "review_uncertain": "holes or gaps", "crop": "a.jpg"},
        {"seq": "B", "feather_id": "A0012_1996_B", "segment_status": "segmented", "needs_review": "",
         "duplicate_of": "A0012_1996_A", "crop": "b.jpg"},
        {"seq": "C", "feather_id": "A0012_1996_C", "segment_status": "segmented", "needs_review": "outline cut: top",
         "crop": "c.jpg"},
        {"seq": "fragment_1", "feather_id": "", "segment_status": "fragment", "needs_review": "fragment: ...",
         "crop": "f.jpg"},
        {"seq": "missed_1", "feather_id": "", "segment_status": "possible_missed", "needs_review": "possible ...",
         "crop": ""},
    ]
    assert [lx.usable(f) for f in r.feathers] == [
        (True, ""), (False, "duplicate of A0012_1996_A"), (False, "check: outline cut: top"),
        (False, "fragment"), (False, "review row: possible missed feather")]
    lx.write_feathers_csv([r], tmp_path / "all.csv", "run")
    lx.write_feathers_csv([r], tmp_path / "usable.csv", "run", usable_only=True)
    import csv
    assert len(list(csv.DictReader(open(tmp_path / "all.csv")))) == 5
    kept = list(csv.DictReader(open(tmp_path / "usable.csv")))
    assert [k["feather_id"] for k in kept] == ["A0012_1996_A"] and kept[0]["usable"] == "yes"


def test_bare_result_gives_ids_only_to_feathers():
    feathers = [{"seq": "A", "crop": "A0012_1996_Feather_A.jpg", "segment_status": "segmented"},
                {"seq": "fragment_1", "crop": "A0012_1996_Fragment_1.jpg", "segment_status": "fragment",
                 "feather_id": ""},
                {"seq": "B", "crop": "A0012_1996_Feather_B.jpg", "segment_status": "segmented"},
                {"seq": "missed_1", "crop": "", "segment_status": "possible_missed", "feather_id": ""}]
    r = lx.bare_result("raw/A0012 1996-im1058.jpg", feathers)
    assert [(f["seq"], f.get("feather_id")) for f in r.feathers] == [
        ("A", "A0012_1996_A"), ("fragment_1", ""), ("B", "A0012_1996_B"), ("missed_1", "")]


def test_repair_fills_a_measured_hole_and_keeps_the_better_mask(tmp_path):
    """A feather whose mask has a large hole fails 'holes or gaps' (measured); the second
    sweep fills it, the filled mask passes, and it replaces the first."""
    import numpy as np
    from src.feather_processing import FeatherProcessor

    raw = tmp_path / "raw"
    raw.mkdir()
    slide = raw / "A0012 1996-im1058.jpg"
    Image.new("RGB", (400, 300), "white").save(slide)
    ms = [np.zeros((300, 400), np.float32) for _ in range(2)]
    ms[0][30:270, 40:100] = 1
    ms[0][120:160, 55:85] = 0          # hole: 1200 px of 14400, above fail_hole_frac
    ms[1][30:270, 300:340] = 1
    p = object.__new__(FeatherProcessor)
    p.segmenter, p.has_vlm = "sam3", False
    p._label_client = client = DateClient()
    p._segment_sam3 = lambda *a, **k: ([[40, 30, 100, 270], [300, 30, 340, 270]], ms, "", [0.9, 0.9])
    opts = {"segmentation": {"feather_judge": {"repair": {"enabled": True}}}}
    res = p.process_image(str(slide), str(tmp_path / "out"), options=opts)
    a = res.feather_ids[0]
    assert a["needs_review"] == "" and a["repair"].startswith("holes filled (before: holes or gaps)")
    assert a["judge_hole_frac"] == 0 and client.judge_calls == 3      # A, its repair, B
    assert "repair" not in res.feather_ids[1]
    off = p.process_image(str(slide), str(tmp_path / "out2"))
    assert off.feather_ids[0]["needs_review"] == "holes or gaps"      # repair is off by default


def test_retry_and_choice_follow_the_checks():
    from types import SimpleNamespace as NS

    from src import full_run_sharded as frs

    clean = NS(success=True, qa_checks={"feathers_failing": 0, "checks_failed": 0, "judge_errors": 0,
                                         "possible_missed": 0, "merged_feathers": False, "score_mean": 4.0})
    cut = NS(success=True, qa_checks={"feathers_failing": 1, "checks_failed": 2, "judge_errors": 0,
                                       "possible_missed": 0, "merged_feathers": False, "score_mean": 9.5})
    missed = NS(success=True, qa_checks={"feathers_failing": 0, "checks_failed": 0, "possible_missed": 1})
    assert not frs.should_retry(clean)            # a low mean score alone never triggers a retry
    assert frs.should_retry(cut) and frs.should_retry(missed)
    assert not frs.should_retry(NS(success=False, qa_checks={"feathers_failing": 3}))
    assert frs.qa_penalty(clean) == 0 and frs.qa_penalty(cut) == 2 and frs.qa_penalty(missed) == 2
    assert frs.qa_penalty(NS(success=True)) == 0  # results without checks (old workers)
    frag = NS(success=True, qa_checks={"feathers_failing": 0, "checks_failed": 0, "fragments": 1})
    assert frs.should_retry(frag) and frs.qa_penalty(frag) == 1
