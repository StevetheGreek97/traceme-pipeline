from __future__ import annotations

import colorsys
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from traceme.sam2.masks import MaskRegion, as_mask_region


def _obj_color(obj_id: int) -> tuple[int, int, int]:
    """Deterministic bright color per obj_id, as BGR for OpenCV."""
    h = (obj_id * 0.1357) % 1.0
    s, v = 0.85, 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def _mask_to_bool(m):
    """Accepts a MaskRegion, torch tensor (1,H,W) or (H,W), or np; returns (H,W) bool.

    Expanding a MaskRegion allocates a frame-sized array, so prefer its own
    crop where the bounding box is enough -- see `_annotate_frame`.
    """
    if isinstance(m, MaskRegion):
        return m.dense()
    if isinstance(m, torch.Tensor):
        m = m.detach().float().cpu().numpy()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    return m > 0


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_CONTOUR_THICKNESS = 2


def _label_rect(txt: str, org: tuple[int, int]) -> tuple[int, int, int, int]:
    """Pixel box `cv2.putText` can touch for `txt` drawn at `org`."""
    (tw, th), baseline = cv2.getTextSize(txt, _FONT, _FONT_SCALE, 2)
    x, y = org
    pad = 2  # stroke thickness and antialiasing spill
    return (y - th - pad, y + baseline + pad, x - pad, x + tw + pad)


def _annotate_frame(frame_bgr: np.ndarray, obj_ids: list[int], masks) -> np.ndarray:
    """
    masks: iterable aligned with obj_ids, each a MaskRegion or an array of
    shape (1,H,W) or (H,W).
    Draws semi-transparent fill + contour + text label.

    Every object is drawn inside its own bounding box, and the final blend
    covers only the pixels that could have changed, so the cost tracks the
    objects rather than the frame. Output is identical either way: outside
    the drawn pixels the blend weights sum to one over identical inputs.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    # Union of every rectangle drawn into, as (y0, y1, x0, x1); None if none.
    touched: tuple[int, int, int, int] | None = None

    def _touch(rect: tuple[int, int, int, int]) -> None:
        nonlocal touched
        if touched is None:
            touched = rect
        else:
            touched = (
                min(touched[0], rect[0]), max(touched[1], rect[1]),
                min(touched[2], rect[2]), max(touched[3], rect[3]),
            )

    for k, oid in enumerate(obj_ids):
        region = as_mask_region(masks[k])
        if region.shape != (h, w):
            resized = cv2.resize(
                region.dense().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
            region = MaskRegion.from_dense(resized, (h, w))
        if region.is_empty:
            continue

        color = _obj_color(int(oid))
        y0, x0 = region.y0, region.x0
        bh, bw = region.height, region.width

        # fill, inside the bounding box only
        crop = region.crop_bool()
        sub = overlay[y0 : y0 + bh, x0 : x0 + bw]
        sub[crop] = (0.6 * np.array(color) + 0.4 * sub[crop]).astype(np.uint8)
        _touch((y0, y0 + bh, x0, x0 + bw))

        # contour, traced on the crop and offset back into frame coordinates
        cnts = region.cv_contours()
        cv2.drawContours(overlay, cnts, -1, color, thickness=_CONTOUR_THICKNESS)
        t = _CONTOUR_THICKNESS
        _touch((y0 - t, y0 + bh + t, x0 - t, x0 + bw + t))

        # label: place near the largest contour if available
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            moments = cv2.moments(c)
            cx = int(moments["m10"] / (moments["m00"] + 1e-6))
            cy = int(moments["m01"] / (moments["m00"] + 1e-6))
        else:
            # fall back to the mask's own centroid
            ys, xs = np.nonzero(crop)
            cy = int((ys + y0).mean()) if ys.size else 20
            cx = int((xs + x0).mean()) if xs.size else 20

        txt = f"id:{oid}"
        org = (max(2, cx - 20), max(15, cy - 8))
        cv2.putText(overlay, txt, org, _FONT, _FONT_SCALE, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, txt, org, _FONT, _FONT_SCALE, color, 1, cv2.LINE_AA)
        _touch(_label_rect(txt, org))

    if touched is None:  # nothing drawn: the blend would be a no-op
        return out

    y0 = max(0, touched[0]); y1 = min(h, touched[1])
    x0 = max(0, touched[2]); x1 = min(w, touched[3])
    out[y0:y1, x0:x1] = cv2.addWeighted(
        overlay[y0:y1, x0:x1], 0.9, out[y0:y1, x0:x1], 0.1, 0.0
    )
    return out


def xywh_to_xyxy(box_xywh):
    x, y, w, h = map(int, box_xywh)
    return [x, y, x + w, y + h]


def extract_frames(input_file, output_dir, quality: int = 1, start_number: int = 0, threads: int = 10):
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_pattern = str(output_path / "%05d.jpg")

    command = [
        "ffmpeg",
        "-i", str(input_path),
        "-q:v", str(quality),
        "-start_number", str(start_number),
        output_pattern,
        "-threads", str(threads),
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Frames extracted to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        raise


def save_mask_png(mask: torch.Tensor | np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = mask.detach().float().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    if m.ndim == 3 and m.shape[0] == 1:  # (1,H,W) -> (H,W)
        m = m[0]
    cv2.imwrite(str(out_path), ((m > 0).astype(np.uint8) * 255))


def save_overlay(
    img_path: Path,
    mask: torch.Tensor | np.ndarray,
    out_path: Path,
    alpha: float = 0.5,
):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    m = mask.detach().float().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    m = m > 0
    overlay = img.copy()
    color = np.zeros_like(img)
    color[..., 2] = 255
    overlay[m] = (alpha * color[m] + (1 - alpha) * img[m]).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
