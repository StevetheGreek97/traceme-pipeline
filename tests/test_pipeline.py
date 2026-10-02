"""Integration tests of the full pipeline against the fake predictor:
outputs, resume, failure handling, retry."""
import json

import cv2
import numpy as np
import pandas as pd
import pytest

import fakes
from traceme.pipeline import PipelineConfig, PipelinePaths, run_pipeline
from traceme.sam2.io import _done_marker, _seed_file


@pytest.fixture
def env(tmp_path):
    frames = tmp_path / "clipA"
    frames.mkdir()
    for i in range(9):
        img = np.full((16, 16, 3), 30 * (i % 8), dtype=np.uint8)
        cv2.imwrite(str(frames / f"{i:05d}.jpg"), img)
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text(
        "prompts:\n"
        "  - {frame_idx: 1, obj_id: 1, points: [[4, 4]], labels: [1]}\n"
        "  - {frame_idx: 2, obj_id: 2, points: [[5, 3]], labels: [1]}\n"
    )
    cfg = PipelineConfig(
        frame_dir=frames,
        output_folder=tmp_path / "out",
        prompt_file=prompt_file,
        chunk_size=5,
        overlap=1,
        fps=5,
    )
    return cfg, PipelinePaths.from_config(cfg)


def _summary(paths):
    return json.loads(paths.run_summary.read_text())


def test_full_run_outputs(env):
    cfg, paths = env
    run_pipeline(cfg)

    s = _summary(paths)
    assert s["status"] == "complete"
    assert s["processed_chunks"] == [0, 1] and s["failed_chunks"] == []
    assert fakes.STATE["builds"] == 1, "predictor must be built once per run"
    assert _seed_file(paths.chunks_dir, 0).exists(), "seeds must persist for resume"
    assert _done_marker(paths.chunks_dir, 0).exists()

    df = pd.read_csv(paths.merged_csv)
    assert sorted(df["global_frame_idx"].unique()) == list(range(9))
    # boundary frame deduped to one row per (frame, obj)
    f4 = df[df["global_frame_idx"] == 4]
    assert len(f4) == len(f4["obj_id"].unique())
    # obj 2 (prompted in chunk 0) must survive into chunk 1 via seeds
    c1 = df[df["chunk_id"] == 1]
    assert set(c1["obj_id"].dropna().astype(int)) == {1, 2}
    assert (df["area_px"] == 24).all()

    # merged video: exactly one frame per input frame (overlap skipped)
    cap = cv2.VideoCapture(str(paths.merged_video))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == 9


def test_resume_skips_completed(env):
    cfg, paths = env
    run_pipeline(cfg)
    builds = fakes.STATE["builds"]

    run_pipeline(cfg)
    assert fakes.STATE["builds"] == builds, "resume must not rebuild the predictor"
    s = _summary(paths)
    assert s["resumed_chunks"] == [0, 1] and s["processed_chunks"] == []


def test_no_resume_reprocesses(env):
    cfg, paths = env
    run_pipeline(cfg)
    builds = fakes.STATE["builds"]

    run_pipeline(PipelineConfig(**{**cfg.__dict__, "resume": False}))
    assert fakes.STATE["builds"] == builds + 1
    assert _summary(paths)["processed_chunks"] == [0, 1]


def test_failure_marks_partial_and_exits_nonzero(env):
    cfg, paths = env
    fakes.STATE["fail_chunk_dirs"] = {str(paths.chunks_dir / "chunk_001")}

    with pytest.raises(SystemExit) as exc:
        run_pipeline(cfg)
    assert exc.value.code == 1

    s = _summary(paths)
    assert s["status"] == "partial" and s["failed_chunks"] == [1]
    assert not _done_marker(paths.chunks_dir, 1).exists()
    # tmp dir must survive even if del_tmp was requested (needed for resume)
    assert paths.tmp_root.exists()


def test_retry_heals_only_failed_chunk(env):
    cfg, paths = env
    fakes.STATE["fail_chunk_dirs"] = {str(paths.chunks_dir / "chunk_001")}
    with pytest.raises(SystemExit):
        run_pipeline(cfg)
    builds = fakes.STATE["builds"]

    fakes.STATE["fail_chunk_dirs"] = set()
    run_pipeline(cfg)
    assert fakes.STATE["builds"] == builds + 1, "only the failed chunk is reprocessed"
    s = _summary(paths)
    assert s["status"] == "complete"
    assert s["resumed_chunks"] == [0] and s["processed_chunks"] == [1]


# ---------------- frame range / contours / resume signature ----------------

def _video_frame_count(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def test_frame_range_keeps_global_indices(env):
    cfg, paths = env
    # frame 1's prompt is outside the range and dropped; frame 2's is kept
    run_pipeline(PipelineConfig(**{**cfg.__dict__, "start_frame": 2, "end_frame": 7}))

    s = _summary(paths)
    assert s["status"] == "complete"
    assert (s["config"]["start_frame"], s["config"]["end_frame"]) == (2, 7)
    df = pd.read_csv(paths.merged_csv)
    assert sorted(df["global_frame_idx"].unique()) == list(range(2, 8))
    assert set(df["obj_id"].dropna().astype(int)) == {2}
    assert _video_frame_count(paths.merged_video) == 6


def test_frame_range_defaults_to_whole_video(env):
    cfg, paths = env
    run_pipeline(PipelineConfig(**{**cfg.__dict__, "start_frame": 2}))  # no end_frame
    df = pd.read_csv(paths.merged_csv)
    assert sorted(df["global_frame_idx"].unique()) == list(range(2, 9))


@pytest.mark.parametrize("start,end", [(0, 9), (5, 3), (-1, 4)])
def test_invalid_frame_range_exits_2(env, start, end):
    cfg, _ = env
    with pytest.raises(SystemExit) as exc:
        run_pipeline(PipelineConfig(**{**cfg.__dict__, "start_frame": start, "end_frame": end}))
    assert exc.value.code == 2


def test_range_without_prompts_exits_2(env):
    cfg, _ = env
    with pytest.raises(SystemExit) as exc:
        run_pipeline(PipelineConfig(**{**cfg.__dict__, "start_frame": 5, "end_frame": 8}))
    assert exc.value.code == 2


def test_contours_output(env):
    cfg, paths = env
    run_pipeline(PipelineConfig(**{**cfg.__dict__, "save_contours": True}))

    lines = [json.loads(l) for l in paths.merged_contours.read_text().splitlines()]
    assert [r["frame"] for r in lines] == list(range(9))  # boundary frame deduped
    by_frame = {r["frame"]: r["objects"] for r in lines}
    # the fake predictor's 4x6 rectangle for obj 1 lives at x 4..9, y 2..5
    (poly,) = by_frame[5]["1"]
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    assert (min(xs), max(xs), min(ys), max(ys)) == (4, 9, 2, 5)
    assert set(by_frame[8]) == {"1", "2"}


def test_contours_with_range_use_global_frames(env):
    cfg, paths = env
    run_pipeline(PipelineConfig(**{**cfg.__dict__, "save_contours": True, "start_frame": 2, "end_frame": 7}))
    frames = [json.loads(l)["frame"] for l in paths.merged_contours.read_text().splitlines()]
    assert frames == list(range(2, 8))


def test_changed_prompts_invalidate_resume(env):
    cfg, paths = env
    run_pipeline(cfg)
    cfg.prompt_file.write_text(
        "prompts:\n  - {frame_idx: 1, obj_id: 3, points: [[4, 4]], labels: [1]}\n"
    )
    run_pipeline(cfg)
    s = _summary(paths)
    assert s["resumed_chunks"] == [] and s["processed_chunks"] == [0, 1]
    df = pd.read_csv(paths.merged_csv)
    assert set(df["obj_id"].dropna().astype(int)) == {3}, "stale obj 1/2 rows must be gone"


def test_changed_range_invalidates_resume(env):
    cfg, paths = env
    run_pipeline(cfg)
    run_pipeline(PipelineConfig(**{**cfg.__dict__, "start_frame": 1}))
    assert _summary(paths)["resumed_chunks"] == []


def test_enabling_contours_reprocesses_chunks(env):
    cfg, paths = env
    run_pipeline(cfg)
    run_pipeline(PipelineConfig(**{**cfg.__dict__, "save_contours": True}))
    s = _summary(paths)
    assert s["resumed_chunks"] == [] and s["processed_chunks"] == [0, 1]
    assert paths.merged_contours.exists()


def test_cli_parses_new_options():
    from traceme.pipeline import parse_args

    cfg = parse_args([
        "-i", "f", "-o", "o", "-p", "p.yaml",
        "--start-frame", "3", "--end-frame", "9", "--device", "cpu", "--save-contours",
    ])
    assert (cfg.start_frame, cfg.end_frame, cfg.device, cfg.save_contours) == (3, 9, "cpu", True)
    cfg = parse_args(["-i", "f", "-o", "o", "-p", "p.yaml"])
    assert (cfg.start_frame, cfg.end_frame, cfg.device, cfg.save_contours) == (None, None, None, False)


def test_no_video_skips_rendering(env):
    cfg, paths = env
    run_pipeline(PipelineConfig(**{**cfg.__dict__, "save_video": False}))
    s = _summary(paths)
    assert s["status"] == "complete" and s["config"]["save_video"] is False
    assert not paths.merged_video.exists()
    assert not list(paths.files_dir.glob("*.mp4"))
    assert sorted(pd.read_csv(paths.merged_csv)["global_frame_idx"].unique()) == list(range(9))


def test_enabling_video_reprocesses_chunks(env):
    cfg, paths = env
    run_pipeline(PipelineConfig(**{**cfg.__dict__, "save_video": False}))
    run_pipeline(cfg)
    s = _summary(paths)
    assert s["resumed_chunks"] == [] and s["processed_chunks"] == [0, 1]
    assert paths.merged_video.exists()


def test_cli_no_video():
    from traceme.pipeline import parse_args

    assert parse_args(["-i", "f", "-o", "o", "-p", "p.yaml", "--no-video"]).save_video is False
    assert parse_args(["-i", "f", "-o", "o", "-p", "p.yaml"]).save_video is True


def test_save_masks_end_to_end(env):
    """--save-masks: per-chunk archives merge into one, reloadable, correct.

    Covers the flag the cluster jobs run with, which had no integration test.
    """
    from dataclasses import replace

    from traceme.sam2.io import _unpack_mask

    cfg, paths = env
    cfg = replace(cfg, save_masks=True)
    run_pipeline(cfg)

    assert _summary(paths)["status"] == "complete"
    assert paths.merged_masks.exists()

    data = np.load(paths.merged_masks, allow_pickle=True)
    frames = data["global_frame_idx"]
    objs = data["obj_id"]

    # every frame of the clip, both objects once the second is prompted
    assert sorted(set(frames.tolist())) == list(range(9))
    assert set(objs.tolist()) == {1, 2}
    # one entry per (frame, object) pair -- no overlap duplicates
    pairs = list(zip(frames.tolist(), objs.tolist()))
    assert len(pairs) == len(set(pairs))

    # masks reload at frame resolution and match the CSV's area/bbox
    df = pd.read_csv(paths.merged_csv)
    checked = 0
    for gidx, oid, packed, shp in zip(frames, objs, data["packed"], data["shape"]):
        mask = _unpack_mask(packed, tuple(shp))
        assert mask.shape == (16, 16), "masks must be stored at frame resolution"
        row = df[(df["global_frame_idx"] == gidx) & (df["obj_id"] == oid)].iloc[0]
        assert int(mask.sum()) == int(row["area_px"])
        ys, xs = np.nonzero(mask)
        assert (int(xs.min()), int(ys.min())) == (int(row["bbox_x"]), int(row["bbox_y"]))
        checked += 1
    assert checked == len(pairs)


def test_save_masks_resume_is_stable(env):
    """A resumed run must leave the mask archive unchanged, not partial."""
    from dataclasses import replace

    cfg, paths = env
    cfg = replace(cfg, save_masks=True)
    run_pipeline(cfg)
    first = paths.merged_masks.read_bytes()

    run_pipeline(cfg)  # fully resumed
    assert _summary(paths)["resumed_chunks"] == [0, 1]
    assert paths.merged_masks.read_bytes() == first


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_chunk_order_and_totals_survive_overlapping_writers(tmp_path, workers):
    """Finalizes may overlap; results must stay ordered and complete."""
    from traceme.sam2.runner import run_sam2
    from traceme.video.chunker import VideoChunker

    frames = tmp_path / "clipB"
    frames.mkdir()
    for i in range(26):
        cv2.imwrite(str(frames / f"{i:05d}.jpg"), np.full((16, 16, 3), 7, np.uint8))

    chunker = VideoChunker(
        frame_dir=frames, output_dir=tmp_path / "tmp", chunk_size=5, overlap=1
    )
    from traceme.prompts.parser import Prompt

    by_chunk = {0: [Prompt(frame_idx=0, obj_id=1, box=None, points=((4, 4),), labels=(1,))]}
    summary = run_sam2(
        chunker=chunker,
        by_chunk=by_chunk,
        output=tmp_path / "out",
        video_fps=5,
        finalize_workers=workers,
    )

    assert summary["failed_chunks"] == []
    assert summary["processed_chunks"] == sorted(summary["processed_chunks"])
    assert summary["processed_chunks"] == list(range(summary["total_chunks"]))
    # every chunk wrote its outputs
    for cid in range(summary["total_chunks"]):
        assert (tmp_path / "out" / f"clipB_chunk_{cid:03d}.csv").exists()
    assert summary["total_frames"] == 26 + (summary["total_chunks"] - 1)  # overlap dupes


def test_failed_chunk_still_attributed_with_overlapping_writers(tmp_path):
    """A chunk that fails must be reported as that chunk, not another."""
    from traceme.prompts.parser import Prompt
    from traceme.sam2.runner import run_sam2
    from traceme.video.chunker import VideoChunker

    frames = tmp_path / "clipC"
    frames.mkdir()
    for i in range(16):
        cv2.imwrite(str(frames / f"{i:05d}.jpg"), np.full((16, 16, 3), 7, np.uint8))

    chunker = VideoChunker(
        frame_dir=frames, output_dir=tmp_path / "tmp", chunk_size=5, overlap=1
    )
    chunker.chunk_frames()
    fakes.STATE["fail_chunk_dirs"].add(str(chunker.get_chunk_dir(1)))

    summary = run_sam2(
        chunker=chunker,
        by_chunk={0: [Prompt(frame_idx=0, obj_id=1, box=None, points=((4, 4),), labels=(1,))]},
        output=tmp_path / "out",
        video_fps=5,
        prepare_chunks=False,
        finalize_workers=2,
    )
    assert summary["failed_chunks"] == [1]
    assert 1 not in summary["processed_chunks"]


def test_importing_pipeline_does_not_resolve_the_model():
    """`traceme.sam2.config` must not be imported by importing the pipeline.

    config.py picks the checkpoint at module level from SAM2_MODEL, and
    run_pipeline only sets that from `--model` well after import. So any
    module-level import chain from traceme.pipeline into traceme.sam2.config
    silently pins the model to the default -- a `--model tiny` run loads
    `large`. A real regression once: video/merge.py imported a helper from
    sam2/io.py, which imports sam2/config.py.

    Runs in a subprocess because conftest stubs config into sys.modules.
    """
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src"
    code = textwrap.dedent(
        """
        import sys, types
        sam2 = types.ModuleType("sam2")
        build = types.ModuleType("sam2.build_sam")
        build.build_sam2_video_predictor = lambda *a, **k: None
        sam2.build_sam = build
        sys.modules["sam2"] = sam2
        sys.modules["sam2.build_sam"] = build

        import traceme.pipeline  # noqa: F401

        leaked = sorted(m for m in sys.modules if m == "traceme.sam2.config")
        if leaked:
            raise SystemExit(
                "traceme.sam2.config was imported at traceme.pipeline import "
                "time; the model would resolve before --model is applied"
            )
        print("clean")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout
