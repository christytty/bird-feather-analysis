"""Optional SAM 3 steps (src/sam3_extras.py) with stand-in predictors."""
from types import SimpleNamespace as NS

import cv2
import numpy as np
import torch

from src import sam3_extras as sx


def _res(boxes, scores, masks):
    return [NS(boxes=NS(xyxy=torch.tensor(boxes, dtype=torch.float32), conf=torch.tensor(scores)),
               masks=NS(data=torch.tensor(np.array(masks, dtype=np.float32))))]


class FakeSemantic:
    def __init__(self, result):
        self.result, self.calls = result, []

    def set_image(self, path):
        self.calls.append(("set_image", path))

    def __call__(self, **kw):
        self.calls.append(("call", kw))
        return self.result


def test_exemplar_pass_prompts_with_a_typical_feather_and_negatives():
    first = ([[0, 0, 100, 400], [200, 0, 300, 380], [400, 0, 700, 400]], [np.zeros((10, 10))] * 3, "", [0.9, 0.95, 0.99])
    m = np.zeros((10, 10)); m[2:8, 4:6] = 1
    sam = FakeSemantic(_res([[0, 0, 100, 400], [120, 0, 200, 400]], [0.9, 0.8], [m, m]))
    out = sx.exemplar_pass(sam, "s.jpg", (1000, 1000), "bird feather", first, {}, 0.3, 0.45)
    kw = sam.calls[-1][1]
    assert kw["bboxes"][0] == [200, 0, 300, 380]                 # the merged-looking wide box is not the example
    assert kw["labels"] == [1, 0, 0] and kw["text"] == ["bird feather"]
    assert len(out[0]) == 2 and out[2] == ""


def test_exemplar_pass_keeps_the_first_result_without_a_good_example():
    first = ([[0, 0, 300, 300]], [np.zeros((5, 5))], "", [0.9])      # not feather-shaped
    sam = FakeSemantic(_res([], [], []))
    assert sx.exemplar_pass(sam, "s.jpg", (1000, 1000), "bird feather", first, {}, 0.3, 0.45) is first


class FakeInteractive:
    """Returns a fixed mask (crop coordinates) for every prompt."""
    def __init__(self, make_mask):
        self.make_mask, self.calls = make_mask, []

    def set_image(self, crop):
        self.shape = crop.shape[:2]

    def __call__(self, **kw):
        self.calls.append(kw)
        return [NS(masks=NS(data=torch.tensor(self.make_mask(self.shape)[None].astype(np.float32))))]


def test_refine_pass_replaces_overlapping_masks_and_keeps_strays(tmp_path):
    img = np.full((200, 300, 3), 200, np.uint8)
    path = str(tmp_path / "s.jpg"); cv2.imwrite(path, img)
    a = np.zeros((200, 300), np.float32); a[50:150, 40:60] = 1
    b = np.zeros((200, 300), np.float32); b[50:150, 200:220] = 1
    first = ([[40, 50, 60, 150], [200, 50, 220, 150]], [a, b], "", [0.9, 0.9])

    def wider(shape):                       # the refined feather: same place, a bit wider
        m = np.zeros(shape, bool); h, w = shape; m[h // 4: 3 * h // 4, w // 2 - 14: w // 2 + 14] = True
        return m
    (boxes, masks, reason, scores), info = sx.refine_pass(FakeInteractive(wider), path, first, {"pad_frac": 0.15})
    assert info["refined"] == 2 and masks[0].sum() > a.sum()

    def elsewhere(shape):                   # the second pass lands somewhere else: keep the first mask
        m = np.zeros(shape, bool); m[:5, :5] = True
        return m
    (_, masks2, _, _), info2 = sx.refine_pass(FakeInteractive(elsewhere), path, first, {})
    assert info2 == {"refined": 0, "kept_first": 2, "slide": "s.jpg"} and np.array_equal(masks2[0] > 0, a > 0)


def test_processor_runs_example_boxes_when_enabled(monkeypatch):
    from PIL import Image

    from src.feather_processing import FeatherProcessor

    p = object.__new__(FeatherProcessor)
    p.sam3 = object()
    seen = {}

    def fake_exemplar(sam3, image_path, size, text, first, cfg, nms_iou, max_ratio, conf=None):
        seen.update(text=text, cfg=cfg)
        return ([[1, 1, 2, 2]], [np.ones((2, 2))], "", [0.9])
    monkeypatch.setattr(sx, "exemplar_pass", fake_exemplar)
    first = ([[0, 0, 1, 1]], [np.ones((2, 2))], "", [0.8])
    seg = {"exemplar": {"enabled": True}, "refine": {"enabled": False}, "conf": 0.25}
    out = p._sam3_extras("s.jpg", Image.new("RGB", (10, 10)), first, seg, ["bird feather"], 0.3, 0.45)
    assert out[0] == [[1, 1, 2, 2]] and seen["text"] == "bird feather"
    off = {"exemplar": {"enabled": False}, "refine": {"enabled": False}}
    assert p._sam3_extras("s.jpg", Image.new("RGB", (10, 10)), first, off, ["bird feather"], 0.3, 0.45) is first


class CropSemantic:
    """Stand-in for the semantic predictor on a crop: returns `make(shape)` -> list of (mask, score)."""
    def __init__(self, make):
        self.make, self.calls = make, 0

    def set_image(self, crop):
        self.shape = crop.shape[:2]

    def __call__(self, text):
        self.calls += 1
        pieces = self.make(self.shape)
        return _res([[0, 0, 1, 1]] * len(pieces), [s for _, s in pieces], [m for m, _ in pieces])


def test_split_merged_replaces_a_wide_mask_with_its_feathers(tmp_path):
    img = np.full((300, 400, 3), 200, np.uint8)
    path = str(tmp_path / "s.jpg"); cv2.imwrite(path, img)
    def mask(x0, x1):
        m = np.zeros((300, 400), np.float32); m[40:260, x0:x1] = 1; return m
    normal = [mask(20, 50), mask(70, 100), mask(120, 150)]
    merged = mask(200, 290)                        # three feathers wide
    first = ([[0, 0, 1, 1]] * 4, normal + [merged], "", [0.9] * 4)

    def two_feathers(shape):                       # crop = merged mask + 20 px margin
        h, w = shape
        a = np.zeros(shape, bool); a[20:h - 20, 20:50] = True
        b = np.zeros(shape, bool); b[20:h - 20, 60:110] = True
        return [(a, 0.9), (b, 0.8)]
    (boxes, masks, _, scores), info = sx.split_merged(CropSemantic(two_feathers), path, "bird feather", first, {})
    assert info["tried"] == 1 and info["split"] == 1 and len(masks) == 5
    assert all(np.array_equal(masks[i] > 0, normal[i] > 0) for i in range(3))   # normal masks untouched

    def one_feather(shape):
        h, w = shape
        a = np.zeros(shape, bool); a[20:h - 20, 20:w - 20] = True
        return [(a, 0.9)]
    (_, masks2, _, _), info2 = sx.split_merged(CropSemantic(one_feather), path, "bird feather", first, {})
    assert info2 == {"tried": 1, "split": 0, "slide": "s.jpg"} and len(masks2) == 4
