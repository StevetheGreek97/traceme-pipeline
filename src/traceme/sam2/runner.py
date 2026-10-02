from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import gc
import numpy as np
import torch
from contextlib import nullcontext
from typing import Literal

from sam2.build_sam import build_sam2_video_predictor
from traceme.sam2.config import SAM2_CHECKPOINT, MODEL_CFG, IS_SAM3, device, SEED_DIRNAME
from traceme.sam2.io import (
    _contours_path,
    _done_marker,
    _mask_archive_path,
    _mask_stats,
    _seed_file,
    _write_contours_for_chunk,
    _write_csv_for_chunk,
    _write_masks_for_chunk,
    global_to_inchunk_idx,
    read_seed_file,
    write_seed_file,
)
from traceme.sam2.masks import MaskRegion
from traceme.core.logging import get_logger, timer
from traceme.video.render import _render_chunk_video

log = get_logger("traceme.sam2.runner")


def _build_predictor():
    if IS_SAM3:
        from traceme.sam2.sam3_backend import build_sam3_tracker

        return build_sam3_tracker(SAM2_CHECKPOINT, device)
    return build_sam2_video_predictor(MODEL_CFG, str(SAM2_CHECKPOINT), device=device)


def _release_state(inf_state):
    if inf_state is not None:
        del inf_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return None


def _collect_regions(obj_ids, mask_logits) -> dict[int, MaskRegion]:
    """Turn one frame of predicted logits into compact per-object masks.

    Each object's bounding box comes from row and column projections reduced
    on the tracker's own device, so only the pixels inside that box cross to
    host memory instead of a frame-sized mask per object. One frame of 12
    objects at 5312x2988 costs ~100 KB here rather than 190 MB, which is what
    lets a chunk of any length be held at once.
    """
    m = mask_logits > 0
    if m.ndim == 4 and m.shape[1] == 1:  # (N, 1, H, W) from both backends
        m = m[:, 0]
    if m.ndim != 3:
        raise ValueError(f"expected (N,1,H,W) or (N,H,W) masks, got {tuple(m.shape)}")

    _, h, w = m.shape
    rows = m.any(dim=2).cpu().numpy()  # (N, H) -- which rows the object spans
    cols = m.any(dim=1).cpu().numpy()  # (N, W) -- which columns
    ids = obj_ids.tolist() if hasattr(obj_ids, "tolist") else list(obj_ids)

    segs: dict[int, MaskRegion] = {}
    for k, oid in enumerate(ids):
        ys = np.flatnonzero(rows[k])
        if ys.size == 0:  # tracked but lost on this frame
            segs[int(oid)] = MaskRegion.empty((h, w))
            continue
        xs = np.flatnonzero(cols[k])
        y0, y1 = int(ys[0]), int(ys[-1])
        x0, x1 = int(xs[0]), int(xs[-1])
        crop = m[k, y0 : y1 + 1, x0 : x1 + 1].cpu().numpy()
        # The projections make this box tight by construction.
        segs[int(oid)] = MaskRegion.from_tight_crop(crop, y0, x0, (h, w))
    return segs


def _finalize_chunk(
    *,
    cid: int,
    out_root: Path,
    csv_path: Path,
    mp4_path: Path,
    frame_files: list,
    video_segments: dict[int, dict[int, MaskRegion]],
    video_fps: int,
    skip_first: int,
    cs: int,
    ov: int,
    save_masks: bool = False,
    save_contours: bool = False,
    save_video: bool = True,
    frame_offset: int = 0,
) -> tuple[int, int]:
    """CPU-side per-chunk output: stats + CSV, annotated video (optional), done marker.

    Runs on the finalize thread so the GPU can start the next chunk. The done
    marker is written last, so a crash anywhere here leaves the chunk
    incomplete and it will be reprocessed on resume.
    """
    with timer(log, f"[chunk {cid:03d}] write_csv"):
        stats_per_frame: dict[int, dict[int, tuple | None]] = {}
        objects_in_chunk = 0
        for idx, segs in video_segments.items():
            if not segs:
                stats_per_frame[idx] = {}
            else:
                spf = {oid: _mask_stats(mask) for oid, mask in segs.items()}
                objects_in_chunk += len(spf)
                stats_per_frame[idx] = spf

        _write_csv_for_chunk(csv_path, stats_per_frame, cid=cid, cs=cs, ov=ov, frame_offset=frame_offset)
        log.info(
            f"[chunk {cid:03d}] wrote {csv_path.name} "
            f"(frames={len(stats_per_frame)}, objs={objects_in_chunk})"
        )

    if save_video:
        with timer(log, f"[chunk {cid:03d}] render_video"):
            _render_chunk_video(
                mp4_path, frame_files, video_segments, fps=video_fps, skip_first=skip_first
            )
            log.info(f"[chunk {cid:03d}] wrote {mp4_path.name}")

    if save_masks:
        with timer(log, f"[chunk {cid:03d}] write_masks"):
            masks_path = _mask_archive_path(csv_path)
            _write_masks_for_chunk(
                masks_path, video_segments, cid=cid, cs=cs, ov=ov, frame_offset=frame_offset
            )
            log.info(f"[chunk {cid:03d}] wrote {masks_path.name}")

    if save_contours:
        with timer(log, f"[chunk {cid:03d}] write_contours"):
            contours_path = _contours_path(csv_path)
            _write_contours_for_chunk(
                contours_path, video_segments, cid=cid, cs=cs, ov=ov, frame_offset=frame_offset
            )
            log.info(f"[chunk {cid:03d}] wrote {contours_path.name}")

    _done_marker(out_root, cid).touch()
    log.info(f"[OK] chunk {cid:03d} complete")
    return len(stats_per_frame), objects_in_chunk


def run_sam2(
    *,
    chunker,
    by_chunk: dict[int, list],
    output: str | Path,
    video_fps: int = 30,
    prepare_chunks: bool = True,
    chunk_mode: Literal["auto", "load", "force"] = "auto",
    resume: bool = True,
    save_masks: bool = False,
    save_contours: bool = False,
    save_video: bool = True,
    frame_offset: int = 0,
    progress_every: int = 50,
    finalize_workers: int = 2,
) -> dict:
    """
    Process all chunks. `frame_offset` is the global index of the first frame
    the chunker holds (non-zero with --start-frame); outputs use global indices.
    Logs "[chunk NNN] propagated k/n frames" every `progress_every` frames.

    `finalize_workers` is how many finished chunks may be written out at once
    while the GPU tracks ahead. Two lets a slow writer -- `--save-masks`
    spends most of its time in zlib, which releases the GIL -- overlap with
    tracking instead of stalling it. Masks are held compactly (see
    traceme.sam2.masks), so each chunk in flight costs KB, and the real
    per-chunk memory is SAM2's own frame cache (~12.6 MB per frame at
    1024x1024 when offload_video_to_cpu is set).

    Returns a summary dict:
    {"total_chunks", "processed_chunks", "resumed_chunks", "failed_chunks",
     "total_frames", "total_objects"} where the chunk entries are lists of ids.
    """
    out_root = chunker.output_dir
    (out_root / SEED_DIRNAME).mkdir(parents=True, exist_ok=True)
    if prepare_chunks:
        with timer(log, "Chunking preparation"):
            chunker.chunk_frames(mode=chunk_mode)

    cs, ov = chunker.chunk_size, chunker.overlap
    total_chunks = chunker.count_chunks()
    all_chunk_ids = list(range(total_chunks))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # Base name for outputs (e.g., 'Control1_A7_100925')
    try:
        base_name = Path(chunker.frame_dir).name
    except Exception:
        base_name = output.name
    log.info(f"Base output name: {base_name}")
    log.info(f"Chunk size: {cs} | overlap: {ov} | total chunks: {total_chunks}")

    # Perf knobs (CUDA only)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        log.info("Enabled TF32 + cuDNN benchmark; using bfloat16 autocast")
    else:
        autocast_ctx = nullcontext()

    total_frames = 0
    total_objects = 0
    processed_chunks: list[int] = []
    resumed_chunks: list[int] = []
    failed_chunks: list[int] = []

    # The predictor is built once (on the first chunk that needs it) and
    # reused for every chunk; only the per-chunk inference state is rebuilt.
    predictor = None
    inf_state = None

    # CSV/render/marker for a finished chunk run on this worker while the GPU
    # tracks the next chunk. At most one finalize is in flight, so memory
    # holds at most two chunks' worth of masks -- compact ones (see
    # traceme.sam2.masks), so this costs KB per frame, not hundreds of MB.
    finalize_workers = max(1, int(finalize_workers))
    finalizer = ThreadPoolExecutor(
        max_workers=finalize_workers, thread_name_prefix="finalize"
    )
    pending: list[tuple[int, Future]] = []

    def _collect_pending(max_inflight: int = 0) -> None:
        """Wait until at most `max_inflight` finalizes are still running.

        Drained oldest-first, so `processed_chunks` stays in chunk order.
        """
        nonlocal total_frames, total_objects
        while len(pending) > max_inflight:
            pcid, fut = pending.pop(0)
            try:
                n_f, n_o = fut.result()
                total_frames += n_f
                total_objects += n_o
                processed_chunks.append(pcid)
            except Exception as e:
                failed_chunks.append(pcid)
                log.error(f"[chunk {pcid:03d}] finalize failed: {e}", exc_info=True)

    try:
        for cid in all_chunk_ids:
            csv_path = output / f"{base_name}_chunk_{cid:03d}.csv"
            mp4_path = output / f"{base_name}_chunk_{cid:03d}.mp4"

            outputs = [csv_path] + ([mp4_path] if save_video else [])
            if save_masks:
                outputs.append(_mask_archive_path(csv_path))
            if save_contours:
                outputs.append(_contours_path(csv_path))
            if resume and _done_marker(out_root, cid).exists() and all(p.exists() for p in outputs):
                resumed_chunks.append(cid)
                log.info(f"[chunk {cid:03d}] already complete; skipping (use --no-resume to reprocess)")
                continue

            chunk_dir = chunker.get_chunk_dir(cid)
            frame_files = chunker.get_frame_paths(cid)
            n_frames = len(frame_files)
            log.info(f"[chunk {cid:03d}] dir={chunk_dir} | frames={n_frames}")
            if n_frames == 0:
                log.warning(f"[chunk {cid:03d}] No frames; skipping")
                _done_marker(out_root, cid).touch()
                continue

            if predictor is None:
                with timer(log, "Build predictor"):
                    predictor = _build_predictor()

            try:
                # init state
                inf_state = predictor.init_state(
                    video_path=str(chunk_dir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=False,
                    async_loading_frames=True,
                )

                # ---- restore overlap seeds ----
                if ov > 0 and cid > 0:
                    prev_seed_path = _seed_file(out_root, cid - 1)
                    if prev_seed_path.exists():
                        # Seeds are kept on disk (not consumed) so an interrupted
                        # run can resume from the first incomplete chunk.
                        with timer(log, f"[chunk {cid:03d}] restore_seeds"):
                            restored = 0
                            for rel, oid, region in read_seed_file(prev_seed_path):
                                if rel >= n_frames:
                                    continue
                                # add_new_mask takes a frame-sized mask, so
                                # this is the one place a region is expanded.
                                predictor.add_new_mask(
                                    inference_state=inf_state,
                                    frame_idx=int(rel),
                                    obj_id=int(oid),
                                    mask=region.dense(),
                                )
                                restored += 1
                            log.info(f"[chunk {cid:03d}] restored {restored} seed masks from overlap")
                    else:
                        log.warning(
                            f"[chunk {cid:03d}] no overlap seeds from chunk {cid - 1:03d}; "
                            "object identity may not carry over"
                        )

                # ---- apply prompts ----
                plist = by_chunk.get(cid, [])
                if plist:
                    with timer(log, f"[chunk {cid:03d}] apply_prompts"):
                        applied = 0
                        per_frame: dict[int, dict[int, dict[str, list]]] = {}
                        for p in plist:
                            i = global_to_inchunk_idx(p.frame_idx, cid, cs, ov)
                            if 0 <= i < n_frames:
                                rec = per_frame.setdefault(i, {}).setdefault(
                                    int(p.obj_id), {"points": [], "labels": [], "boxes": []}
                                )
                                for (x, y), lab in zip(p.points, p.labels):
                                    rec["points"].append([int(x), int(y)])
                                    rec["labels"].append(int(lab))
                                if p.box:
                                    x, y, w, h = map(int, p.box)
                                    rec["boxes"].append([x, y, x + w, y + h])
                        for i in sorted(per_frame.keys()):
                            for obj_id, pack in per_frame[i].items():
                                box_xyxy = pack["boxes"][0] if pack["boxes"] else None
                                predictor.add_new_points_or_box(
                                    inference_state=inf_state,
                                    frame_idx=i,
                                    obj_id=int(obj_id),
                                    points=pack["points"] or None,
                                    labels=pack["labels"] or None,
                                    clear_old_points=True,
                                    normalize_coords=True,
                                    box=box_xyxy,
                                )
                                applied += 1
                        log.info(f"[chunk {cid:03d}] applied {applied} prompts")

                # ---- propagate ----
                video_segments: dict[int, dict[int, MaskRegion]] = {}
                with timer(log, f"[chunk {cid:03d}] propagate_and_collect"):
                    with torch.inference_mode(), autocast_ctx:
                        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inf_state):
                            video_segments[out_frame_idx] = _collect_regions(
                                out_obj_ids, out_mask_logits
                            )
                            done = len(video_segments)
                            if progress_every > 0 and (done % progress_every == 0 or done == n_frames):
                                log.info(f"[chunk {cid:03d}] propagated {done}/{n_frames} frames")

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # ---- overlap seeds (sync: the next chunk restores them) ----
                if ov > 0:
                    with timer(log, f"[chunk {cid:03d}] write_overlap_seeds"):
                        tail_start = max(0, n_frames - ov)
                        entries: list[tuple[int, int, MaskRegion]] = []
                        seeded_frames = 0
                        for rel, abs_idx in enumerate(range(tail_start, n_frames)):
                            segs = video_segments.get(abs_idx, {})
                            if not segs:
                                continue
                            seeded_frames += 1
                            entries.extend((rel, int(oid), reg) for oid, reg in segs.items())

                        if entries:
                            sf = _seed_file(out_root, cid)
                            n_seeds = write_seed_file(sf, entries, entries[0][2].shape)
                            log.info(
                                f"[chunk {cid:03d}] wrote seeds → {sf.name} "
                                f"(frames={seeded_frames}, masks={n_seeds})"
                            )

                # ---- hand off CSV/video/marker to the finalize worker ----
                # Keep a slot free for the chunk about to be submitted.
                _collect_pending(max_inflight=finalize_workers - 1)
                skip = min(ov, cid * cs) if (cid > 0 and ov > 0) else 0
                pending.append((
                    cid,
                    finalizer.submit(
                        _finalize_chunk,
                        cid=cid,
                        out_root=out_root,
                        csv_path=csv_path,
                        mp4_path=mp4_path,
                        frame_files=frame_files,
                        video_segments=video_segments,
                        video_fps=video_fps,
                        skip_first=skip,
                        cs=cs,
                        ov=ov,
                        save_masks=save_masks,
                        save_contours=save_contours,
                        save_video=save_video,
                        frame_offset=frame_offset,
                    ),
                ))

            except Exception as e:
                failed_chunks.append(cid)
                log.exception(f"[chunk {cid:03d}] processing failed: {e}")

            finally:
                inf_state = _release_state(inf_state)

    finally:
        try:
            _collect_pending()
        finally:
            finalizer.shutdown(wait=True)
            if predictor is not None:
                del predictor
                predictor = None
            inf_state = _release_state(inf_state)

    log.info(
        f"Run complete: frames={total_frames}, objects={total_objects}, chunks={total_chunks} "
        f"(processed={len(processed_chunks)}, resumed={len(resumed_chunks)}, failed={len(failed_chunks)})"
    )
    if failed_chunks:
        log.error(f"Failed chunks: {failed_chunks} — their frames are missing from merged outputs")
    return {
        "total_chunks": total_chunks,
        "processed_chunks": processed_chunks,
        "resumed_chunks": resumed_chunks,
        "failed_chunks": failed_chunks,
        "total_frames": total_frames,
        "total_objects": total_objects,
    }
