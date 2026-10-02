from pathlib import Path
import json
import subprocess
import shutil

import numpy as np
import pandas as pd

from traceme.core.logging import get_logger
# From masks (numpy-only), never from sam2.io: io imports sam2.config, whose
# module-level MODEL resolution would then run before the CLI sets SAM2_MODEL.
from traceme.sam2.masks import object_array
from traceme.video.chunker import numeric_sort_key

try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover - optional dependency
    imageio_ffmpeg = None

log = get_logger("traceme.video.merge")


def _handle_single_file(files: list[Path], output: Path, kind: str) -> bool:
    if len(files) != 1:
        return False
    src = files[0]
    if src.resolve() == output.resolve():
        log.info(f"Single {kind} already correctly named → {output.name}")
        return True
    try:
        # Copy (not rename): the per-chunk file must stay in place so a
        # subsequent run can recognize the chunk as complete and resume.
        shutil.copy2(src, output)
        log.info(f"Copied single {kind} → {output.name}")
    except Exception as e:
        log.exception(f"Failed to copy {src.name} to {output.name}: {e}")
    return True


def _resolve_ffmpeg() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path and imageio_ffmpeg is not None:
        try:
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            log.info(f"Using bundled ffmpeg at: {ffmpeg_path}")
        except Exception as e:
            log.exception(f"imageio_ffmpeg.get_ffmpeg_exe() failed: {e}")
            ffmpeg_path = None
    return ffmpeg_path


def merge_csv_chunks(input_dir: Path, output_csv: Path) -> None:
    """
    Merges chunk_*.csv files into one combined CSV named after frame_dir.
    If only one CSV exists, it is simply renamed to match the frame_dir name.
    """
    csv_files = sorted(input_dir.glob("*.csv"), key=numeric_sort_key)
    if not csv_files:
        log.warning(f"No chunk_*.csv files found in {input_dir}")
        return

    if _handle_single_file(csv_files, output_csv, "CSV"):
        return


    # Otherwise, merge multiple CSVs
    log.info(f"Merging {len(csv_files)} CSV files...")

    frames = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            log.exception(f"Failed to read {csv_file}: {e}")
            continue

        if "global_frame_idx" not in df.columns:
            log.error(f"{csv_file} missing 'global_frame_idx'; skipping.")
            continue
        if "obj_id" not in df.columns:
            log.error(f"{csv_file} missing 'obj_id'; skipping.")
            continue

        frames.append(df)

    if not frames:
        log.warning("No valid rows found during merge; not writing merged file.")
        return

    merged_df = pd.concat(frames, ignore_index=True)
    # Dedup per (frame, object): boundary frames are duplicated across adjacent
    # chunks, and each frame can hold multiple tracked objects. `keep="first"`
    # preserves the earlier chunk's row, matching the previous single-row
    # dedup behavior. pandas treats NaN obj_id (frames with no objects) as
    # equal to itself here, so those still dedup correctly too.
    merged_df = merged_df.drop_duplicates(subset=["global_frame_idx", "obj_id"], keep="first")
    merged_df = merged_df.sort_values(["global_frame_idx", "obj_id"]).reset_index(drop=True)

    # A blank obj_id on any row forces that whole column to float64 on read;
    # restore clean integer formatting instead of "500.0" everywhere.
    for col in ("chunk_id", "global_frame_idx", "in_chunk_idx", "area_px"):
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype("int64")
    # Columns blank on empty frames need the nullable integer dtype.
    for col in ("obj_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"):
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype("Int64")

    merged_df.to_csv(output_csv, index=False)

    log.info(
        f"[OK] Merged {len(csv_files)} chunk files → {output_csv.name} "
        f"({len(merged_df)} rows, {merged_df['global_frame_idx'].nunique()} unique frames)"
    )


def merge_mask_chunks(input_dir: Path, output_npz: Path) -> None:
    """
    Merges chunk_*_masks.npz files into one combined mask archive named
    after frame_dir. If only one archive exists, it is copied to match the
    frame_dir name (same rationale as merge_csv_chunks: the per-chunk file
    must stay in place so a subsequent run can recognize the chunk as
    complete and resume).
    """
    mask_files = sorted(input_dir.glob("*_masks.npz"), key=numeric_sort_key)
    if not mask_files:
        log.warning(f"No chunk *_masks.npz files found in {input_dir}")
        return

    if _handle_single_file(mask_files, output_npz, "mask archive"):
        return

    log.info(f"Merging {len(mask_files)} mask archives...")

    # Dedup per (frame, object): boundary frames are duplicated across
    # adjacent chunks. Keep the earlier chunk's mask, matching CSV merge.
    merged: dict[tuple[int, int], tuple] = {}
    for mask_file in mask_files:
        try:
            data = np.load(mask_file, allow_pickle=True)
        except Exception as e:
            log.exception(f"Failed to read {mask_file}: {e}")
            continue

        for gidx, oid, packed, shp in zip(
            data["global_frame_idx"], data["obj_id"], data["packed"], data["shape"]
        ):
            key = (int(gidx), int(oid))
            if key not in merged:
                merged[key] = (packed, shp)

    if not merged:
        log.warning("No mask entries found during merge; not writing merged file.")
        return

    keys_sorted = sorted(merged.keys())
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        global_frame_idx=np.array([k[0] for k in keys_sorted], dtype=np.int32),
        obj_id=np.array([k[1] for k in keys_sorted], dtype=np.int32),
        packed=object_array([merged[k][0] for k in keys_sorted]),
        shape=np.array([merged[k][1] for k in keys_sorted], dtype=object),
    )

    log.info(
        f"[OK] Merged {len(mask_files)} chunk archives → {output_npz.name} "
        f"({len(keys_sorted)} mask entries)"
    )


def merge_contour_chunks(input_dir: Path, output_jsonl: Path) -> None:
    """
    Merges chunk *_contours.jsonl files into one file named after frame_dir,
    one line per frame. Boundary frames appear in adjacent chunks; per
    (frame, object) the earlier chunk's entry is kept, matching the CSV merge.
    """
    contour_files = sorted(input_dir.glob("*_contours.jsonl"), key=numeric_sort_key)
    if not contour_files:
        log.warning(f"No chunk *_contours.jsonl files found in {input_dir}")
        return

    if _handle_single_file(contour_files, output_jsonl, "contours file"):
        return

    log.info(f"Merging {len(contour_files)} contour files...")
    frames: dict[int, dict[str, list]] = {}
    for path in contour_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    objs = frames.setdefault(int(rec["frame"]), {})
                    for oid, polys in rec.get("objects", {}).items():
                        objs.setdefault(oid, polys)
        except Exception as e:
            log.exception(f"Failed to read {path}: {e}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_jsonl.with_name(output_jsonl.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for gidx in sorted(frames):
            objs = dict(sorted(frames[gidx].items(), key=lambda kv: int(kv[0])))
            f.write(json.dumps({"frame": gidx, "objects": objs}, separators=(",", ":")) + "\n")
    tmp.replace(output_jsonl)
    log.info(f"[OK] Merged {len(contour_files)} contour files → {output_jsonl.name} ({len(frames)} frames)")


def merge_chunk_videos(input_dir: Path, output_file: Path) -> None:
    """
    Concatenate all *.mp4 files in `input_dir` into `output_file`.
    Uses system ffmpeg if available; otherwise tries imageio-ffmpeg.
    If only one MP4 exists, it is renamed to `output_file`.
    """
    mp4_files = sorted(input_dir.glob("*.mp4"), key=numeric_sort_key)
    if not mp4_files:
        log.warning(f"No .mp4 files found in {input_dir}")
        return

    if _handle_single_file(mp4_files, output_file, "MP4"):
        return

    # Locate ffmpeg
    ffmpeg_path = _resolve_ffmpeg()

    if not ffmpeg_path:
        log.error(
            "No ffmpeg found in PATH and imageio-ffmpeg not available. "
            "Install with: pip install 'imageio[ffmpeg]'. Skipping merge."
        )
        return

    # Prepare concat list beside the output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    concat_list = output_file.with_suffix(".concat.txt")

    def _ffconcat_line(p: Path) -> str:
        # Use absolute POSIX paths and escape single quotes for ffmpeg concat demuxer
        s = p.resolve().as_posix().replace("'", r"'\''")
        return f"file '{s}'\n"

    try:
        with concat_list.open("w", encoding="utf-8") as f:
            for p in mp4_files:
                f.write(_ffconcat_line(p))

        cmd = [
            str(ffmpeg_path),
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_file),
        ]

        log.info(f"ffmpeg concat → {output_file.name}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log.info(f"[OK] merged video written: {output_file.name}")

    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        log.exception(f"ffmpeg merge failed: {err}")
    finally:
        # Remove the temporary concat list
        try:
            concat_list.unlink(missing_ok=True)
        except Exception:
            pass
