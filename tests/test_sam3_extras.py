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
