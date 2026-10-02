from __future__ import annotations

from pathlib import Path
import csv
import json

import cv2
import numpy as np

from traceme.sam2.config import SEED_DIRNAME
from traceme.sam2.masks import (
    MaskRegion,
    as_mask_region,
    object_array,  # re-exported: callers have always imported it from here
    squeeze_mask_2d,
)


def _seed_file(out_root: Path, cid: int) -> Path:
    return out_root / SEED_DIRNAME / f"seed_chunk_{cid:03d}.npz"


def _done_marker(out_root: Path, cid: int) -> Path:
    return out_root / SEED_DIRNAME / f"chunk_{cid:03d}.done"


def _pack_mask_bool(m_bool: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Accept masks as (H,W), (1,H,W) or (H,W,1), bool/uint8/0-255.
    Returns (packed_bits, (H, W)).
    """
    m = np.asarray(m_bool)

    # Squeeze common singleton channels
    if m.ndim == 3 and m.shape[0] == 1:   # [1,H,W] -> [H,W]
        m = m[0]
    if m.ndim == 3 and m.shape[2] == 1:   # [H,W,1] -> [H,W]
        m = m[..., 0]

    # Ensure 2D
    if m.ndim != 2:
        raise ValueError(f"_pack_mask_bool expects 2D mask, got shape {m.shape}")

    # Binarize and pack
    if m.dtype != np.bool_:
        m = (m > 0)
    h, w = m.shape
    packed = np.packbits(m.astype(np.uint8), axis=1)  # pack along width
    return packed, (h, w)


def _unpack_mask(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Unpack a mask from packed bits to boolean array."""
    packed = np.asarray(packed, dtype=np.uint8)  # ensure correct dtype
    h, w = shape
    u = np.unpackbits(packed, axis=1)
    return u[:, :w].reshape(h, w).astype(bool)


def global_to_inchunk_idx(global_idx: int, cid: int, chunk_size: int, overlap: int) -> int:
    """In-chunk position of a frame. `global_idx` is relative to the first
    processed frame (i.e. already shifted by any --start-frame offset)."""
    return global_idx - _chunk_first_frame(cid, chunk_size, overlap)


def _chunk_first_frame(cid: int, chunk_size: int, overlap: int) -> int:
    """Index (relative to the processed range) of a chunk's first frame,
    including its leading overlap frames."""
    start = cid * chunk_size
    return max(0, start - overlap) if cid > 0 and overlap > 0 else start


def _mask_stats(mask) -> tuple[int, float, float, int, int, int, int] | None:
    """
    Per-object mask stats: (area_px, centroid_x, centroid_y, bbox_x, bbox_y, bbox_w, bbox_h).
    Returns None for an empty mask. Accepts a MaskRegion, or (H,W), (1,H,W)
    or (H,W,1) arrays.

    A MaskRegion answers from its crop, so the cost follows the object; a
    plain array is still scanned in full.
    """
    if isinstance(mask, MaskRegion):
        return mask.stats()
    m = squeeze_mask_2d(mask)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (
        int(xs.size),
        round(float(xs.mean()), 2),
        round(float(ys.mean()), 2),
        x0,
        y0,
        x1 - x0 + 1,
        y1 - y0 + 1,
    )


def _mask_archive_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_masks.npz")


def _contours_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_contours.jsonl")


def _mask_contours(mask, epsilon: float = 1.0) -> list[list[list[int]]]:
    """Outer contours of every region of a mask, simplified with
    approxPolyDP, as [[[x, y], ...], ...]. Empty list for an empty mask.

    Accepts a MaskRegion or a plain array; a region traces only its bounding
    box instead of the whole frame.
    """
    polys = []
    for c in as_mask_region(mask).cv_contours():
        if epsilon > 0:
            c = cv2.approxPolyDP(c, epsilon, True)
        pts = c.reshape(-1, 2)
        if len(pts) >= 3:
            polys.append(pts.astype(int).tolist())
    return polys


def _write_contours_for_chunk(
    path: Path,
    video_segments: dict[int, dict[int, np.ndarray]],
    *,
    cid: int,
    cs: int,
    ov: int,
    frame_offset: int = 0,
) -> None:
    """One JSON line per frame: {"frame": g, "objects": {"<obj_id>": polygons}}.

    A lightweight stand-in for the full masks, for visual review: an object
    with an empty polygon list was tracked but lost on that frame; a frame
    with no objects has an empty mapping.
    """
    first = _chunk_first_frame(cid, cs, ov) + frame_offset
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for in_idx in sorted(video_segments.keys()):
            objs = {
                str(int(oid)): _mask_contours(mask)
                for oid, mask in sorted(video_segments[in_idx].items())
            }
            f.write(json.dumps({"frame": first + in_idx, "objects": objs}, separators=(",", ":")) + "\n")


def _write_masks_for_chunk(
    path: Path,
    video_segments: dict[int, dict[int, np.ndarray]],
    *,
    cid: int,
    cs: int,
    ov: int,
    frame_offset: int = 0,
) -> None:
    """Persist every non-empty object mask in a chunk as bit-packed arrays.

    One entry per (frame, object) pair. Reload with e.g.:
        data = np.load(path, allow_pickle=True)
        for gidx, oid, packed, shp in zip(
            data["global_frame_idx"], data["obj_id"], data["packed"], data["shape"]
        ):
            mask = _unpack_mask(packed, tuple(shp))
    """
    ovl_start = _chunk_first_frame(cid, cs, ov) + frame_offset

    global_idx_list: list[int] = []
    obj_id_list: list[int] = []
    packed_list: list[np.ndarray] = []
    shape_list: list[tuple[int, int]] = []

    for in_idx in sorted(video_segments.keys()):
        global_idx = ovl_start + in_idx
        for obj_id, mask in video_segments[in_idx].items():
            region = as_mask_region(mask)
            if region.is_empty:
                continue
            packed, shp = region.packed_full(), region.shape
            global_idx_list.append(global_idx)
            obj_id_list.append(int(obj_id))
            packed_list.append(packed)
            shape_list.append(shp)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        global_frame_idx=np.array(global_idx_list, dtype=np.int32),
        obj_id=np.array(obj_id_list, dtype=np.int32),
        packed=object_array(packed_list),
        shape=np.array(shape_list, dtype=object),
    )


# ---------------------------------------------------------------- seed files
#
# A chunk saves the masks of its trailing overlap frames so the next chunk can
# re-seed the same objects and keep their ids. Format 1 stored each mask
# bit-packed over the whole frame in nested object arrays: at 5312x2988 with
# 12 objects and 20 overlap frames that is ~475 MB to write and read back per
# chunk. Format 2 stores each mask's bounding box and the packed crop inside
# it, concatenated into one flat buffer -- the same masks in a few dozen KB,
# with no pickled object arrays.
#
# Readers accept both, so a run can resume from seeds an older version wrote.

SEED_FORMAT = 2


def write_seed_file(
    path: Path,
    entries: list[tuple[int, int, MaskRegion]],
    frame_shape: tuple[int, int],
) -> int:
    """Write overlap seeds as (in-chunk frame index, object id, mask) triples.

    Returns the number of masks written. Empty masks are kept, not dropped:
    re-seeding an object the tracker lost is what keeps its id alive into the
    next chunk, so its "lost" rows keep appearing in the CSV.
    """
    kept = list(entries)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not kept:
        blobs = np.zeros(0, dtype=np.uint8)
        offsets = np.zeros(1, dtype=np.int64)
    else:
        crops = [reg.packed_crop().ravel() for _, _, reg in kept]
        blobs = np.concatenate(crops)
        offsets = np.zeros(len(crops) + 1, dtype=np.int64)
        np.cumsum([c.size for c in crops], out=offsets[1:])

    np.savez_compressed(
        path,
        format=np.array(SEED_FORMAT, dtype=np.int32),
        frame_shape=np.array(frame_shape, dtype=np.int32),
        rel_indices=np.array([r for r, _, _ in kept], dtype=np.int32),
        obj_ids=np.array([o for _, o, _ in kept], dtype=np.int32),
        boxes=np.array(
            [[reg.y0, reg.x0, reg.height, reg.width] for _, _, reg in kept],
            dtype=np.int32,
        ).reshape(-1, 4),
        areas=np.array([reg.area for _, _, reg in kept], dtype=np.int64),
        packed=blobs,
        packed_offsets=offsets,
    )
    return len(kept)


def read_seed_file(path: Path) -> list[tuple[int, int, MaskRegion]]:
    """Read overlap seeds written by either format, newest first in effort.

    Returns (in-chunk frame index, object id, mask) triples.
    """
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        if "boxes" in keys:  # format 2
            rel = data["rel_indices"]
            oids = data["obj_ids"]
            boxes = data["boxes"]
            areas = data["areas"]
            blobs = data["packed"]
            offsets = data["packed_offsets"]
            shape = tuple(int(v) for v in data["frame_shape"])
            out = []
            for i in range(len(rel)):
                y0, x0, h, w = (int(v) for v in boxes[i])
                out.append((
                    int(rel[i]),
                    int(oids[i]),
                    MaskRegion.from_packed_crop(
                        blobs[offsets[i]:offsets[i + 1]], y0, x0, h, w,
                        shape, area=int(areas[i]),
                    ),
                ))
            return out

        # format 1: per-frame nested object arrays of whole-frame packed masks
        rel_indices = data["rel_indices"]
        obj_ids_arr = data["obj_ids"]
        packed_list = data["packed"]
        shapes_list = data["shapes"]

    out = []
    for r, oids, packed_masks, shapes in zip(
        rel_indices, obj_ids_arr, packed_list, shapes_list
    ):
        for oid, packed, shp in zip(
            np.asarray(oids, dtype=np.int32), packed_masks, shapes
        ):
            out.append((
                int(r),
                int(oid),
                MaskRegion.from_packed_full(
                    np.asarray(packed, dtype=np.uint8), tuple(map(int, shp))
                ),
            ))
    return out


CSV_HEADER = [
    "chunk_id", "global_frame_idx", "in_chunk_idx", "obj_id",
    "area_px", "centroid_x", "centroid_y",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
]


def _write_csv_for_chunk(
    csv_path: Path,
    stats_per_frame: dict[int, dict[int, tuple | None]],
    *,
    cid: int,
    cs: int,
    ov: int,
    frame_offset: int = 0,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        ovl_start = _chunk_first_frame(cid, cs, ov) + frame_offset
        for in_idx in sorted(stats_per_frame.keys()):
            per_obj = stats_per_frame[in_idx]
            global_idx = ovl_start + in_idx
            if not per_obj:
                writer.writerow([cid, global_idx, in_idx, "", 0, "", "", "", "", "", ""])
                continue
            for obj_id in sorted(per_obj.keys()):
                stats = per_obj[obj_id]
                if stats is None:
                    # Object is tracked but its mask vanished (lost) in this frame.
                    writer.writerow([cid, global_idx, in_idx, obj_id, -1, -1, -1, -1, -1, -1, -1])
                else:
                    writer.writerow([cid, global_idx, in_idx, obj_id, *stats])
