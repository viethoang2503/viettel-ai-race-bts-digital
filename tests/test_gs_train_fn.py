from pathlib import Path

from src.common.config import SceneConfig
from src.training import gs_train_fn


def _chair_scene():
    return SceneConfig(
        name="chair",
        root=Path("VAI_NVS_DATA_ROUND2/chair"),
        train_images_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/images"),
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv"),
        submission_dir="chair",
    )


def test_real_train_fn_skips_subprocess_when_checkpoint_already_at_target(tmp_path, monkeypatch):
    (tmp_path / "chkpnt30000.pth").touch()
    calls = []
    monkeypatch.setattr(gs_train_fn.subprocess, "run", lambda *a, **k: calls.append((a, k)))

    result = gs_train_fn.real_train_fn(_chair_scene(), tmp_path, iterations=30000)

    assert calls == []
    assert result == tmp_path / "chkpnt30000.pth"


def test_real_train_fn_runs_subprocess_from_scratch_when_no_checkpoint(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, cwd, check):
        calls.append((argv, cwd, check))
        (tmp_path / "chkpnt30000.pth").touch()

    monkeypatch.setattr(gs_train_fn.subprocess, "run", fake_run)

    result = gs_train_fn.real_train_fn(_chair_scene(), tmp_path, iterations=30000)

    assert len(calls) == 1
    argv, cwd, check = calls[0]
    assert cwd == "third_party/gaussian-splatting"
    assert check is True
    assert "--start_checkpoint" not in argv
    assert "--checkpoint_iterations" in argv
    assert argv[argv.index("--checkpoint_iterations") + 1] == "30000"
    assert result == tmp_path / "chkpnt30000.pth"


def test_real_train_fn_resumes_from_partial_checkpoint(tmp_path, monkeypatch):
    partial = tmp_path / "chkpnt15000.pth"
    partial.touch()
    calls = []

    def fake_run(argv, cwd, check):
        calls.append(argv)
        (tmp_path / "chkpnt30000.pth").touch()

    monkeypatch.setattr(gs_train_fn.subprocess, "run", fake_run)

    result = gs_train_fn.real_train_fn(_chair_scene(), tmp_path, iterations=30000)

    argv = calls[0]
    assert "--start_checkpoint" in argv
    assert str(partial.resolve()) == argv[argv.index("--start_checkpoint") + 1]
    assert result == tmp_path / "chkpnt30000.pth"


def test_real_train_fn_raises_if_subprocess_produces_no_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(gs_train_fn.subprocess, "run", lambda *a, **k: None)

    try:
        gs_train_fn.real_train_fn(_chair_scene(), tmp_path, iterations=30000)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "30000" in str(e)
