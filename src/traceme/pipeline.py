from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, TypeVar
import argparse
import json
import logging
import shutil

import os

from traceme.core.device import DEVICE_CHOICES
from traceme.core.logging import add_file_handler, get_logger, set_log_context, timer
from traceme.video.merge import merge_contour_chunks, merge_csv_chunks, merge_chunk_videos, merge_mask_chunks
from traceme.prompts.parser import YamlPromptParser
from traceme.video.chunker import VideoChunker

ChunkMode = Literal["auto", "load", "force"]
MODEL_CHOICES = ("tiny", "small", "base_plus", "large", "sam3")
T = TypeVar("T")


@dataclass(frozen=True)
class PipelineConfig:
    frame_dir: Path
    output_folder: Path
    prompt_file: Path
    chunk_size: int = 500
    overlap: int = 1
    fps: int = 60
    del_tmp: bool = False
    chunk_mode: ChunkMode = "auto"
    model: str | None = None
    resume: bool = True
    save_masks: bool = False
    save_contours: bool = False
    save_video: bool = True  # annotated .mp4 per chunk, merged into <clip>.mp4
    # Inclusive range of frames to process (positions in the sorted frame
    # list); None means the start/end of the video. Outputs keep the video's
    # global frame indices.
    start_frame: int | None = None
    end_frame: int | None = None
    device: str | None = None


@dataclass(frozen=True)
class PipelinePaths:
    output_root: Path
    tmp_root: Path
    chunks_dir: Path
    files_dir: Path
    log_file: Path
    merged_csv: Path
    merged_video: Path
    merged_masks: Path
    merged_contours: Path
    run_summary: Path
    resume_signature: Path

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "PipelinePaths":
        tmp_root = cfg.output_folder / f"{cfg.frame_dir.name}_tmp"
        return cls(
            output_root=cfg.output_folder,
            tmp_root=tmp_root,
            chunks_dir=tmp_root / "chunks",
            files_dir=tmp_root / "files",
            log_file=tmp_root / f"{cfg.frame_dir.name}_run.log",
            merged_csv=cfg.output_folder / f"{cfg.frame_dir.name}.csv",
            merged_video=cfg.output_folder / f"{cfg.frame_dir.name}.mp4",
            merged_masks=cfg.output_folder / f"{cfg.frame_dir.name}_masks.npz",
            merged_contours=cfg.output_folder / f"{cfg.frame_dir.name}_contours.jsonl",
            run_summary=cfg.output_folder / f"{cfg.frame_dir.name}_run_summary.json",
            resume_signature=tmp_root / "resume_signature.json",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full video chunking and processing pipeline.")
    parser.add_argument(
        "-i", "--frame_dir", required=True, type=Path,
        help="Path to the input FRAMES directory."
    )
    parser.add_argument(
        "-o", "--output_folder", required=True, type=Path,
        help="Path to the output directory."
    )
    parser.add_argument(
        "-p", "--prompt_file", required=True, type=Path,
        help="YAML prompt file path."
    )
    parser.add_argument(
        "-c", "--chunk_size", type=int, default=500,
        help="Number of frames per chunk (default: 500)."
    )
    parser.add_argument(
        "--overlap", type=int, default=1,
        help="Number of overlapping frames between chunks (default: 1)."
    )
    parser.add_argument(
        "--fps", type=int, default=60,
        help="FPS for output video (default: 60)."
    )
    parser.add_argument(
        "--chunk-mode",
        choices=("auto", "load", "force"),
        default="auto",
        help="Chunking mode (default: auto)."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=None,
        help="Model to use: a SAM2 size (tiny/small/base_plus/large) or 'sam3' "
             "(overrides SAM2_MODEL env var; sam3 requires the sam3 extra)."
    )
    parser.add_argument(
        "-d", "--del_tmp", action="store_true",
        help="If set, deletes temporary frame/chunk folders after processing."
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Reprocess all chunks even if completed outputs from a previous run exist."
    )
    parser.add_argument(
        "--save-masks", action="store_true",
        help="Persist per-frame object masks as a compressed .npz per chunk "
             "(<chunk>_masks.npz, bit-packed) for downstream shape analysis."
    )
    parser.add_argument(
        "--save-contours", action="store_true",
        help="Also write <clip>_contours.jsonl: per frame, each object's mask outline as "
             "polygons (empty list = object lost). Small enough for visual review."
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Don't render the annotated .mp4 (per chunk, merged into <clip>.mp4). "
             "Saves time when only the CSV/masks/contours are needed."
    )
    parser.add_argument(
        "--start-frame", type=int, default=None,
        help="First frame to process, as a 0-based position in the sorted frame folder "
             "(default: first frame). Outputs keep the video's frame numbers."
    )
    parser.add_argument(
        "--end-frame", type=int, default=None,
        help="Last frame to process, inclusive (default: last frame)."
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default=None,
        help="Compute device (default: auto = CUDA, then Apple MPS, then CPU; "
             "overrides TRACEME_DEVICE). An unavailable choice falls back to CPU."
    )
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> PipelineConfig:
    args = build_parser().parse_args(argv)
    return PipelineConfig(
        frame_dir=args.frame_dir,
        output_folder=args.output_folder,
        prompt_file=args.prompt_file,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        fps=args.fps,
        del_tmp=args.del_tmp,
        chunk_mode=args.chunk_mode,
        model=args.model,
        resume=not args.no_resume,
        save_masks=args.save_masks,
        save_contours=args.save_contours,
        save_video=not args.no_video,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        device=args.device,
    )


def _attach_file_logger(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    add_file_handler(log_file)


def _validate_inputs(cfg: PipelineConfig, log: logging.Logger) -> None:
    if not cfg.frame_dir.exists() or not cfg.frame_dir.is_dir():
        log.error(f"frame_dir does not exist or is not a directory: {cfg.frame_dir}")
        raise SystemExit(2)
    if not cfg.prompt_file.exists():
        log.error(f"prompt_file not found: {cfg.prompt_file}")
        raise SystemExit(2)
    if cfg.start_frame is not None and cfg.start_frame < 0:
        log.error(f"--start-frame must be >= 0, got {cfg.start_frame}")
        raise SystemExit(2)
    if cfg.start_frame is not None and cfg.end_frame is not None and cfg.end_frame < cfg.start_frame:
        log.error(f"--end-frame ({cfg.end_frame}) is before --start-frame ({cfg.start_frame})")
        raise SystemExit(2)


def _log_config(cfg: PipelineConfig, log: logging.Logger) -> None:
    log.info(
        "Pipeline config: frame_dir=%s output=%s prompt=%s chunk_size=%s overlap=%s fps=%s "
        "chunk_mode=%s model=%s device=%s frames=%s-%s del_tmp=%s",
        cfg.frame_dir,
        cfg.output_folder,
        cfg.prompt_file,
        cfg.chunk_size,
        cfg.overlap,
        cfg.fps,
        cfg.chunk_mode,
        cfg.model or os.environ.get("SAM2_MODEL", "<env>"),
        cfg.device or os.environ.get("TRACEME_DEVICE", "auto"),
        cfg.start_frame if cfg.start_frame is not None else "first",
        cfg.end_frame if cfg.end_frame is not None else "last",
        cfg.del_tmp,
    )


def _run_step(log: logging.Logger, label: str, fn: Callable[[], T], *, fatal: bool = True) -> T | None:
    with timer(log, label):
        try:
            return fn()
        except Exception as e:
            log.exception(f"{label} failed: {e}")
            if fatal:
                raise
    return None


def run_pipeline(cfg: PipelineConfig) -> None:
    log = get_logger("traceme.pipeline")
    paths = PipelinePaths.from_config(cfg)

    paths.output_root.mkdir(parents=True, exist_ok=True)
    _attach_file_logger(paths.log_file)
    set_log_context(
        run_id=cfg.frame_dir.name,
        job_id=os.getenv("JOBBER_TASK_ID") or os.getenv("SLURM_PROCID") or os.getenv("SLURM_JOB_ID"),
    )
    _validate_inputs(cfg, log)
    _log_config(cfg, log)

    # Both are read when traceme.sam2.config is first imported, below.
    if cfg.model:
        os.environ["SAM2_MODEL"] = cfg.model
    if cfg.device:
        os.environ["TRACEME_DEVICE"] = cfg.device

    try:
        from traceme.sam2 import config as sam2_config
    except Exception as e:
        log.error("Failed to import SAM2 config. Install SAM2 and its dependencies. Error: %s", e)
        raise

    sam2_config.log_runtime_summary(log)
    sam2_config.log_model_summary(log)
    sam2_config.log_precision_summary(log)

    try:
        chunker = VideoChunker(
            frame_dir=cfg.frame_dir,
            output_dir=paths.chunks_dir,
            chunk_size=cfg.chunk_size,
            overlap=cfg.overlap,
            action="symlink",
            remove_org=False,
            start_frame=cfg.start_frame or 0,
            end_frame=cfg.end_frame,
        )
    except ValueError as e:
        log.error(str(e))
        raise SystemExit(2)
    frame_offset = chunker.start_frame
    log.info(
        f"Processing frames {chunker.start_frame}-{chunker.end_frame} "
        f"of {chunker.total_source_frames}"
    )

    if cfg.resume:
        _invalidate_stale_outputs(paths, _resume_signature(cfg, chunker), log, sam2_config.SEED_DIRNAME)

    _run_step(
        log,
        "Chunking frames",
        lambda: chunker.chunk_frames(mode=cfg.chunk_mode),
    )
    log.info(
        f"Chunks prepared: {chunker.count_chunks()} | "
        f"chunk_size={cfg.chunk_size} | overlap={cfg.overlap}"
    )

    prompts = _run_step(
        log,
        "Loading prompts",
        lambda: YamlPromptParser(cfg.prompt_file).load(strict_keys=False),
    ) or []
    prompts = _prompts_in_range(prompts, chunker.start_frame, chunker.end_frame, log)
    by_chunk = YamlPromptParser.prompts_by_chunk(
        prompts,
        chunk_size=cfg.chunk_size,
        overlap=cfg.overlap,
        total_frames=len(chunker.get_all_frames_flat()) or None,
    )
    log.info(
        f"Prompts loaded: {sum(len(v) for v in by_chunk.values())} total "
        f"across {len(by_chunk)} chunks"
    )

    from traceme.sam2.runner import run_sam2

    summary = _run_step(
        log,
        "SAM2 run",
        lambda: run_sam2(
            chunker=chunker,
            by_chunk=by_chunk,
            output=paths.files_dir,
            video_fps=cfg.fps,
            prepare_chunks=False,
            resume=cfg.resume,
            save_masks=cfg.save_masks,
            save_contours=cfg.save_contours,
            save_video=cfg.save_video,
            frame_offset=frame_offset,
        ),
        fatal=True,
    ) or {}
    failed_chunks = summary.get("failed_chunks", [])

    _run_step(
        log,
        "Merging chunk CSVs",
        lambda: merge_csv_chunks(paths.files_dir, paths.merged_csv),
        fatal=False,
    )
    if cfg.save_video:
        _run_step(
            log,
            "Merging chunk videos",
            lambda: merge_chunk_videos(paths.files_dir, paths.merged_video),
            fatal=False,
        )
    if cfg.save_masks:
        _run_step(
            log,
            "Merging chunk masks",
            lambda: merge_mask_chunks(paths.files_dir, paths.merged_masks),
            fatal=False,
        )
    if cfg.save_contours:
        _run_step(
            log,
            "Merging chunk contours",
            lambda: merge_contour_chunks(paths.files_dir, paths.merged_contours),
            fatal=False,
        )

    _run_step(
        log,
        "Writing run summary",
        lambda: _write_run_summary(paths.run_summary, cfg, summary, chunker),
        fatal=False,
    )

    if cfg.del_tmp:
        if failed_chunks:
            log.warning(
                "Keeping temporary files despite --del_tmp: %d chunk(s) failed and "
                "the tmp folder is needed to resume.",
                len(failed_chunks),
            )
        else:
            def _cleanup() -> None:
                shutil.rmtree(paths.tmp_root, ignore_errors=True)
                log.info("Temporary files deleted.")

            _run_step(log, "Cleanup temporary files", _cleanup, fatal=False)

    if failed_chunks:
        log.error(
            "Pipeline finished with %d failed chunk(s): %s. Merged outputs are PARTIAL. "
            "Re-run the same command to retry only the failed chunks.",
            len(failed_chunks),
            failed_chunks,
        )
        raise SystemExit(1)

    log.info("Pipeline finished successfully.")


def _prompts_in_range(prompts: list, start: int, end: int, log: logging.Logger) -> list:
    """Keep prompts inside [start, end] and shift them to range-relative frame
    indices, which is what the chunk math works in."""
    kept = [replace(p, frame_idx=p.frame_idx - start) for p in prompts if start <= p.frame_idx <= end]
    dropped = len(prompts) - len(kept)
    if dropped:
        log.warning(f"Ignoring {dropped} prompt(s) outside frames {start}-{end}")
    if prompts and not kept:
        log.error(f"No prompts fall inside frames {start}-{end}; nothing to track")
        raise SystemExit(2)
    return kept


def _resume_signature(cfg: PipelineConfig, chunker: VideoChunker) -> dict:
    """What a resumed chunk's outputs depend on. If any of it changes between
    runs into the same output folder, completed chunks are stale."""
    return {
        "start_frame": chunker.start_frame,
        "end_frame": chunker.end_frame,
        "chunk_size": cfg.chunk_size,
        "overlap": cfg.overlap,
        "model": cfg.model or os.environ.get("SAM2_MODEL", "large"),
        "prompts_sha256": hashlib.sha256(cfg.prompt_file.read_bytes()).hexdigest(),
    }


def _invalidate_stale_outputs(
    paths: PipelinePaths, signature: dict, log: logging.Logger, seed_dirname: str
) -> None:
    """Drop completed-chunk outputs and seeds from a previous run whose inputs
    differ (prompts, model, frame range, chunking), so resume can't silently
    reuse results computed from other inputs."""
    old = None
    if paths.resume_signature.exists():
        try:
            old = json.loads(paths.resume_signature.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    if old is not None and old != signature:
        changed = sorted(k for k in signature if old.get(k) != signature[k])
        log.warning(
            "Inputs changed since the previous run in this folder (%s); "
            "discarding its chunk outputs instead of resuming.",
            ", ".join(changed) or "unknown",
        )
        shutil.rmtree(paths.files_dir, ignore_errors=True)
        shutil.rmtree(paths.chunks_dir / seed_dirname, ignore_errors=True)
    paths.resume_signature.parent.mkdir(parents=True, exist_ok=True)
    paths.resume_signature.write_text(json.dumps(signature, indent=2), encoding="utf-8")


def _write_run_summary(path: Path, cfg: PipelineConfig, summary: dict, chunker: VideoChunker | None = None) -> None:
    failed = summary.get("failed_chunks", [])
    payload = {
        "status": "partial" if failed else "complete",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        **summary,
        "config": {
            "frame_dir": str(cfg.frame_dir),
            "output_folder": str(cfg.output_folder),
            "prompt_file": str(cfg.prompt_file),
            "chunk_size": cfg.chunk_size,
            "overlap": cfg.overlap,
            "fps": cfg.fps,
            "chunk_mode": cfg.chunk_mode,
            "model": cfg.model or os.environ.get("SAM2_MODEL", "large"),
            "resume": cfg.resume,
            "save_masks": cfg.save_masks,
            "save_contours": cfg.save_contours,
            "save_video": cfg.save_video,
            "start_frame": chunker.start_frame if chunker else cfg.start_frame,
            "end_frame": chunker.end_frame if chunker else cfg.end_frame,
            "device": cfg.device or os.environ.get("TRACEME_DEVICE", "auto"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cli_main(argv: Optional[Iterable[str]] = None) -> None:
    run_pipeline(parse_args(argv))
