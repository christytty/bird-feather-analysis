"""Second sweep over feathers that failed a fixable check (src/feather_repair.py)."""
import numpy as np

from src import feather_repair as fr


def _rec(**checks):
    base = {"judge_status": "ok", "judge_sides_cut": [], "judge_holes_or_gaps": False, "judge_single_feather": True,
            "judge_fragment": False, "judge_other_feather_inside": False, "judge_other_material_inside": False}
    return {**base, **{"judge_" + k: v for k, v in checks.items()}}


def test_fail_items_split_the_cut_sides():
    assert fr.fail_items(_rec(sides_cut=["left", "top"], glue_inside=True)) == {
        "outline cut: left", "outline cut: top", "glue inside"}


def test_only_fixable_failures_are_repaired():
    assert fr.worth_repairing(_rec(sides_cut=["left"])) == {"outline cut: left"}
    assert fr.worth_repairing(_rec(holes_or_gaps=True, hidden="part")) == {"holes or gaps"}
    assert fr.worth_repairing(_rec(hidden="most")) == set()                       # nothing a mask can fix
    assert fr.worth_repairing(_rec(sides_cut=["left"], fragment=True)) == set()   # a piece, not a feather
    assert fr.worth_repairing(_rec()) == set()
    assert fr.worth_repairing({"judge_status": "model_error"}) == set()


def test_a_repair_must_fail_fewer_checks_and_nothing_new():
    old = {"outline cut: left", "holes or gaps"}
    assert fr.better(old, {"holes or gaps"}) and fr.better(old, set())
    assert not fr.better(old, old)
    assert not fr.better(old, {"holes or gaps", "glue inside"})        # fixed one, broke another


def test_fill_holes():
    m = np.zeros((50, 50), bool)
    m[5:45, 10:40] = True
    assert fr.fill_holes(m) is None
    m[20:25, 20:25] = False
    filled = fr.fill_holes(m)
    assert filled is not None and filled[22, 22] and filled.sum() == 40 * 30


def test_side_growth_counts_new_pixels_per_side_and_skips_neighbours():
    old = np.zeros((200, 100), bool)
    old[20:180, 40:60] = True
    new = old.copy()
    new[20:180, 10:40] = True                  # grew left (a pale vane)
    others = np.zeros_like(old)
    others[20:180, 60:90] = True
    new |= others                              # the neighbour's pixels never count
    g = fr.side_growth(old, new, others)
    assert g["left"] > 1.0 and g["right"] == 0.0


def test_measured_cut_sides_fail_the_outline_check():
    from src import feather_judge as fj

    rec = {**_rec(), "judge_outline_left": "complete", "judge_outline_right": "complete",
           "judge_outline_top": "complete", "judge_outline_bottom": "complete", "judge_score": 10}
    fj.add_cut_sides(rec, ["left"], {})
    assert fj.failed_checks(rec) == ["outline cut: left"] and rec["judge_score"] == 8
