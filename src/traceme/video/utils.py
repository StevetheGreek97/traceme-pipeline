"""Compatibility exports for legacy utils usage."""

from traceme.sam2.masks import MaskRegion, as_mask_region
from traceme.sam2.io import (
    _seed_file,
    _pack_mask_bool,
    _unpack_mask,
    _write_csv_for_chunk,
    global_to_inchunk_idx,
)
from traceme.video.render import _render_chunk_video
from traceme.video.frames import (
    _annotate_frame,
    extract_frames,
    save_mask_png,
    save_overlay,
    xywh_to_xyxy,
)

__all__ = [
    "MaskRegion",
    "as_mask_region",
    "_seed_file",
    "_pack_mask_bool",
    "_unpack_mask",
    "_write_csv_for_chunk",
    "_render_chunk_video",
    "global_to_inchunk_idx",
    "_annotate_frame",
    "extract_frames",
    "save_mask_png",
    "save_overlay",
    "xywh_to_xyxy",
]
