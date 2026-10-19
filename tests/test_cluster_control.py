"""Run control helpers (no cluster needed)."""
import json

from src import cluster_control as cc

LOG = """slice100-x: 5 slides -> workers
slice100-x: 5 slides -> workers
  ok  A0012 1996-im1058.jpg on node-4
  FAIL BX40 368 2004-im1800.jpg on node-2 no_boxes
  ok  A3681 2001-im1686.jpg on node-4
"""


def test_parse_log_counts_and_phase():
    p = cc.parse_log(LOG)
    assert (p["total"], p["done"], p["failed"], p["phase"]) == (5, 3, 1, "running")
    assert p["by_worker"] == {"node-4": 2, "node-2": 1}
    assert p["failures"] == [("BX40 368 2004-im1800.jpg", "node-2")]
    assert cc.parse_log(LOG + "cluster part done in 3 min\n")["phase"] == "collecting outputs"
    assert cc.parse_log(LOG + "wrote data/label_runs/x/labels.csv, review.csv\n")["phase"] == "finished"
    assert cc.parse_log("")["phase"] == "starting"


def test_progress_and_eta_from_a_registered_run(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "CONTROL", tmp_path)
    monkeypatch.setattr(cc, "hosts", lambda: ["h1", "h2"])
    log = tmp_path / "r.log"
    log.write_text(LOG)
    cc.register_run("r", log, 5, None, "2026-01-01T00:00:00+00:00")
    p = cc.progress("r", live=False)
    assert p["done"] == 3 and p["total"] == 5 and not p["driver_running"] and p["slides_per_hour"] is not None
    assert cc.progress_bar(p, width=10).startswith("[######....] 3/5")


def test_launch_dry_run_builds_the_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "CONTROL", tmp_path)
    monkeypatch.setattr(cc, "hosts", lambda: ["h1", "h2", "h3", "h4"])
    rec = cc.launch_run(["a.jpg", "b.jpg"], "test-x", upload=False, dry_run=True)
    assert "--no-upload" in rec["cmd"] and rec["slides"] == 2
    assert (tmp_path / "test-x.manifest.txt").read_text() == "a.jpg\nb.jpg\n"
    assert not (tmp_path / "test-x.json").exists()          # dry run starts nothing
