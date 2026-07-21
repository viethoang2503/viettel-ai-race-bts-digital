import hashlib
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

    captured_env = {}

    def fake_run(argv, cwd, check, env):
        calls.append((argv, cwd, check))
        captured_env.update(env)
        (tmp_path / "chkpnt30000.pth").touch()

    monkeypatch.setattr(gs_train_fn.subprocess, "run", fake_run)

    result = gs_train_fn.real_train_fn(_chair_scene(), tmp_path, iterations=30000)

    assert len(calls) == 1
    argv, cwd, check = calls[0]
    assert cwd == str(gs_train_fn.GS_ROOT)
    assert check is True
    assert "--start_checkpoint" not in argv
    assert "--checkpoint_iterations" in argv
    # Saved every 5000 iterations, not just at the end — a Colab disconnect
    # partway through must be able to resume from the latest interval
    # instead of losing all progress from that run.
    ckpt_idx = argv.index("--checkpoint_iterations")
    checkpoint_values = argv[ckpt_idx + 1:ckpt_idx + 7]
    assert checkpoint_values == ["5000", "10000", "15000", "20000", "25000", "30000"]
    # train.py's tqdm progress bar must flush live instead of block-
    # buffering, or a genuinely-running multi-hour training looks stuck in
    # Colab's output pane.
    assert captured_env.get("PYTHONUNBUFFERED") == "1"
    assert result == tmp_path / "chkpnt30000.pth"


def test_real_train_fn_resumes_from_partial_checkpoint(tmp_path, monkeypatch):
    partial = tmp_path / "chkpnt15000.pth"
    partial.touch()
    calls = []

    def fake_run(argv, cwd, check, env):
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


def test_real_train_fn_retrains_when_scene_contents_changed(tmp_path, monkeypatch):
    scene_dir = tmp_path / "scene_images"
    scene_dir.mkdir()
    (scene_dir / "0001.jpg").touch()
    scene = SceneConfig(
        name="chair", root=Path("VAI_NVS_DATA_ROUND2/chair"),
        train_images_dir=scene_dir,
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv"),
        submission_dir="chair",
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "chkpnt30000.pth").touch()
    (output_dir / "stale_marker.txt").touch()  # simulates leftover state from the old run
    # A fingerprint recorded by a PRIOR real_train_fn call on different
    # scene contents — deliberately does not match what _scene_fingerprint
    # will compute for `scene` below, proving a real mismatch (not just an
    # absent fingerprint) is what triggers the wipe.
    (output_dir / gs_train_fn._FINGERPRINT_FILENAME).write_text("stale-fingerprint-from-old-data")

    calls = []

    def fake_run(argv, cwd, check, env):
        calls.append(argv)
        (output_dir / "chkpnt30000.pth").touch()

    monkeypatch.setattr(gs_train_fn.subprocess, "run", fake_run)

    result = gs_train_fn.real_train_fn(scene, output_dir, iterations=30000)

    # The recorded fingerprint didn't match, so the old checkpoint must be
    # treated as stale/untrusted and training must actually run, not skip.
    assert len(calls) == 1
    assert not (output_dir / "stale_marker.txt").exists(), (
        "output_dir must be wiped before retraining on a fingerprint mismatch"
    )
    assert result == output_dir / "chkpnt30000.pth"


def test_real_train_fn_reuses_checkpoint_when_fingerprint_matches(tmp_path, monkeypatch):
    scene_dir = tmp_path / "scene_images"
    scene_dir.mkdir()
    (scene_dir / "0001.jpg").touch()
    scene = SceneConfig(
        name="chair", root=Path("VAI_NVS_DATA_ROUND2/chair"),
        train_images_dir=scene_dir,
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv"),
        submission_dir="chair",
    )
    output_dir = tmp_path / "output"

    calls = []

    def fake_run(argv, cwd, check, env):
        calls.append(argv)
        (output_dir / "chkpnt30000.pth").touch()

    monkeypatch.setattr(gs_train_fn.subprocess, "run", fake_run)

    # First call: trains from scratch and records a fingerprint.
    gs_train_fn.real_train_fn(scene, output_dir, iterations=30000)
    assert len(calls) == 1

    # Second call, same scene contents: must skip, not retrain.
    result = gs_train_fn.real_train_fn(scene, output_dir, iterations=30000)
    assert len(calls) == 1  # unchanged — no new subprocess call
    assert result == output_dir / "chkpnt30000.pth"


def test_real_train_fn_retrains_when_image_content_changes_under_same_filename(tmp_path, monkeypatch):
    scene_dir = tmp_path / "scene_images"
    scene_dir.mkdir()
    (scene_dir / "0001.jpg").write_bytes(b"original pixel bytes")
    scene = SceneConfig(
        name="chair", root=Path("VAI_NVS_DATA_ROUND2/chair"),
        train_images_dir=scene_dir,
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv"),
        submission_dir="chair",
    )
    output_dir = tmp_path / "output"

    calls = []

    def fake_run(argv, cwd, check, env):
        calls.append(argv)
        (output_dir / "chkpnt30000.pth").touch()

    monkeypatch.setattr(gs_train_fn.subprocess, "run", fake_run)

    # First call: trains from scratch and records a fingerprint over content.
    gs_train_fn.real_train_fn(scene, output_dir, iterations=30000)
    assert len(calls) == 1

    # Re-upload: SAME filename, DIFFERENT pixel content — a filename-only
    # fingerprint would miss this entirely and wrongly skip retraining.
    (scene_dir / "0001.jpg").write_bytes(b"completely different re-uploaded pixel bytes")
    (output_dir / "stale_marker.txt").touch()

    result = gs_train_fn.real_train_fn(scene, output_dir, iterations=30000)

    assert len(calls) == 2, "content changed under the same filename — must retrain, not skip"
    assert not (output_dir / "stale_marker.txt").exists(), (
        "output_dir must be wiped before retraining on a content mismatch"
    )
    assert result == output_dir / "chkpnt30000.pth"


def test_real_train_fn_retrains_when_train_config_version_changes(tmp_path, monkeypatch):
    # Same scene data, unchanged — but a checkpoint recorded under an OLDER
    # _TRAIN_CONFIG_VERSION (e.g. before the --resolution 1 fix) must be
    # treated as stale even though nothing about the scene itself changed.
    scene_dir = tmp_path / "scene_images"
    scene_dir.mkdir()
    (scene_dir / "0001.jpg").write_bytes(b"pixel bytes")
    scene = SceneConfig(
        name="chair", root=Path("VAI_NVS_DATA_ROUND2/chair"),
        train_images_dir=scene_dir,
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv"),
        submission_dir="chair",
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "chkpnt30000.pth").touch()
    (output_dir / "stale_marker.txt").touch()

    old_fingerprint = hashlib.sha256(
        f"old-config-version|{scene_dir.resolve()}|30000".encode()
    ).hexdigest()
    (output_dir / gs_train_fn._FINGERPRINT_FILENAME).write_text(old_fingerprint)

    calls = []

    def fake_run(argv, cwd, check, env):
        calls.append(argv)
        (output_dir / "chkpnt30000.pth").touch()

    monkeypatch.setattr(gs_train_fn.subprocess, "run", fake_run)

    result = gs_train_fn.real_train_fn(scene, output_dir, iterations=30000)

    assert len(calls) == 1, "old config version — must retrain, not skip"
    assert not (output_dir / "stale_marker.txt").exists()
    assert result == output_dir / "chkpnt30000.pth"


def test_checkpoint_schedule_hits_every_interval_and_the_final_iteration():
    assert gs_train_fn._checkpoint_schedule(30000, interval=5000) == [
        5000, 10000, 15000, 20000, 25000, 30000,
    ]


def test_checkpoint_schedule_always_includes_final_iteration_even_if_not_a_multiple():
    assert gs_train_fn._checkpoint_schedule(12000, interval=5000) == [5000, 10000, 12000]


def test_checkpoint_schedule_handles_iterations_smaller_than_interval():
    assert gs_train_fn._checkpoint_schedule(3000, interval=5000) == [3000]
