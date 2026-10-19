"""Output collection from the nodes (a local folder stands in for a node)."""
import shlex

from src.sample_run import collect_outputs


class LocalIO:
    """'ssh host cmd' becomes 'bash -c cmd' on this machine."""
    def _ssh_base_cmd(self, host):
        return ["bash", "-c"]

    def _remote_shell_path(self, path):
        return shlex.quote(path)


def test_collects_everything_in_the_run_folder(tmp_path):
    node = tmp_path / "node" / "processed"
    (node / "feather_qc").mkdir(parents=True)
    (node / "labels" / "evidence").mkdir(parents=True)
    for f in ("A0012 1996_BoundingBoxes.jpg", "A0012_1996_Feather_A.jpg",
              "A0514_20000521-im1163_Feather_A.jpg", "A0514 20000521-im1163_BoundingBoxes.jpg"):  # date-named
        (node / f).write_bytes(b"jpg")
    (node / "feather_qc" / "A0514_20000521-im1163_Feather_A_qc.jpg").write_bytes(b"jpg")
    (node / "labels" / "A0514 2000-im1163.json").write_text("{}")
    (node / "labels" / "evidence" / "big_crop.jpg").write_bytes(b"x" * 1000)
    (node / "._A0012_1996_Feather_A.jpg").write_bytes(b"apple double")
    out = tmp_path / "local"
    n = collect_outputs(LocalIO(), ["node-1", "node-2"], str(node), out)
    got = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert "A0514_20000521-im1163_Feather_A.jpg" in got and "feather_qc/A0514_20000521-im1163_Feather_A_qc.jpg" in got
    assert "labels/A0514 2000-im1163.json" in got
    assert not any("evidence" in g or g.startswith("._") for g in got)
    assert n == 2 * 6            # both "nodes" serve the same folder here


def test_node_without_the_run_folder_is_skipped(tmp_path):
    assert collect_outputs(LocalIO(), ["node-1"], str(tmp_path / "missing"), tmp_path / "out") == 0
