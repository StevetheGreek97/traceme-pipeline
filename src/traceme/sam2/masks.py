"""Compact in-memory representation for tracked object masks.

The tracker hands back one mask per object per frame, sized like the whole
frame. Keeping those dense costs ``H * W`` bytes each, which is what made long
chunks of high-resolution video exhaust host memory: a 1000-frame chunk with
12 objects at 5312x2988 needs ~190 GB, and the runner holds two chunks at once.

Yet a tracked object covers almost none of the frame. On a GoPro clip of
spiders at 5312x2988 the mean mask is ~800 of 15.9M pixels, so better than
99.99% of every dense mask is zeros.

:class:`MaskRegion` stores a mask as its bounding box plus the bit-packed crop
inside it, so the cost follows the object instead of the frame: ~250 bytes for
that spider, and never worse than ``H * W / 8`` bytes (a bit-packed whole
frame) for an object that fills the view. Consumers ask for what they need --
:meth:`~MaskRegion.stats`, :meth:`~MaskRegion.cv_contours`,
:meth:`~MaskRegion.crop_bool` -- and only :meth:`~MaskRegion.dense` and
:meth:`~MaskRegion.packed_full`, which exist to feed the tracker and the
on-disk archives, materialise anything frame-sized.

Every region keeps a *tight* box: with a non-empty mask, its first and last
row and column each hold at least one set pixel. That invariant is what lets
:meth:`~MaskRegion.stats` report the bounding box without looking at pixels.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MaskRegion", "as_mask_region", "object_array", "squeeze_mask_2d"]


def object_array(items: list) -> np.ndarray:
    """A 1-D object array holding `items` by reference.

    `np.array(items, dtype=object)` looks equivalent but is not: given a list
    of equally-shaped arrays it *broadcasts* them into one N-dimensional
    object array. For 3600 whole-frame packed masks that is a
    (3600, 2988, 664) array -- seven billion Python pointers, tens of GB --
    where the intent was 3600 references. Iterating either one yields the same
    per-mask arrays, so files written the old way still read correctly.
    """
    out = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        out[i] = item
    return out


def squeeze_mask_2d(mask) -> np.ndarray:
    """Normalise a mask to a 2-D array.

    Accepts ``(H, W)``, ``(1, H, W)`` or ``(H, W, 1)``, numpy or torch.
    """
    if hasattr(mask, "detach"):  # torch tensor, without importing torch
        mask = mask.detach().cpu().numpy()
    m = np.asarray(mask)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {m.shape}")
    return m


def _to_bool(m: np.ndarray) -> np.ndarray:
    """Binarise like the rest of the pipeline does: strictly positive is set.

    Not ``astype(bool)``, which would also set negative values -- masks
    arriving as raw logits must threshold the same way ``_pack_mask_bool``
    always has.
    """
    return m if m.dtype == np.bool_ else (m > 0)


def _bounds(flags: np.ndarray) -> tuple[int, int] | None:
    """First and last index of a True in a 1-D boolean array, or None."""
    hits = np.flatnonzero(flags)
    if hits.size == 0:
        return None
    return int(hits[0]), int(hits[-1])


class MaskRegion:
    """A binary mask held as a bounding box plus a bit-packed crop.

    Construct with :meth:`from_dense` (any full-frame mask),
    :meth:`from_crop` (a sub-rectangle, tightened for you),
    :meth:`from_tight_crop` (a sub-rectangle already tight) or
    :meth:`from_packed_full` (the on-disk form).
    """

    __slots__ = ("shape", "y0", "x0", "height", "width", "area", "_packed")

    def __init__(
        self,
        shape: tuple[int, int],
        y0: int,
        x0: int,
        height: int,
        width: int,
        area: int,
        packed: np.ndarray | None,
    ) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.y0 = int(y0)
        self.x0 = int(x0)
        self.height = int(height)
        self.width = int(width)
        self.area = int(area)
        self._packed = packed

    # ---------------- constructors ----------------

    @classmethod
    def empty(cls, shape: tuple[int, int]) -> "MaskRegion":
        """A region holding no set pixels."""
        return cls(shape, 0, 0, 0, 0, 0, None)

    @classmethod
    def from_tight_crop(
        cls, crop, y0: int, x0: int, shape: tuple[int, int]
    ) -> "MaskRegion":
        """Build from a crop whose bounding box is already tight.

        The caller promises every edge row and column of `crop` holds a set
        pixel -- true when the box came from row/column projections of the
        mask, as in the propagate loop. Use :meth:`from_crop` when unsure;
        a wrong promise makes :meth:`stats` report the wrong bounding box.
        """
        cb = _to_bool(squeeze_mask_2d(crop))
        if cb.size == 0 or not cb.any():
            return cls.empty(shape)
        h, w = cb.shape
        return cls(
            shape,
            y0,
            x0,
            h,
            w,
            int(cb.sum()),
            np.packbits(cb, axis=1),
        )

    @classmethod
    def from_crop(cls, crop, y0: int, x0: int, shape: tuple[int, int]) -> "MaskRegion":
        """Build from an arbitrary sub-rectangle, tightening its box."""
        cb = _to_bool(squeeze_mask_2d(crop))
        if cb.size == 0:
            return cls.empty(shape)
        rows = _bounds(cb.any(axis=1))
        cols = _bounds(cb.any(axis=0))
        if rows is None or cols is None:
            return cls.empty(shape)
        r0, r1 = rows
        c0, c1 = cols
        return cls.from_tight_crop(
            cb[r0 : r1 + 1, c0 : c1 + 1], y0 + r0, x0 + c0, shape
        )

    @classmethod
    def from_dense(cls, mask, shape: tuple[int, int] | None = None) -> "MaskRegion":
        """Build from a full-frame mask of any dtype."""
        m = squeeze_mask_2d(mask)
        full = (int(m.shape[0]), int(m.shape[1])) if shape is None else shape
        return cls.from_crop(m, 0, 0, full)

    @classmethod
    def from_packed_crop(
        cls,
        packed,
        y0: int,
        x0: int,
        height: int,
        width: int,
        shape: tuple[int, int],
        area: int | None = None,
    ) -> "MaskRegion":
        """Rebuild a region from :meth:`packed_crop` and its box.

        `area` is recomputed when not supplied. The box is trusted, as it
        came from a region that already held a tight one.
        """
        if area == 0 or height == 0 or width == 0:
            return cls.empty(shape)
        p = np.asarray(packed, dtype=np.uint8).reshape(height, (width + 7) // 8)
        if area is None:
            area = int(np.unpackbits(p, axis=1)[:, :width].sum())
            if area == 0:
                return cls.empty(shape)
        return cls(shape, y0, x0, height, width, area, p)

    @classmethod
    def from_packed_full(cls, packed, shape: tuple[int, int]) -> "MaskRegion":
        """Build from a whole-frame bit-packed mask (the archive form)."""
        h, w = int(shape[0]), int(shape[1])
        p = np.asarray(packed, dtype=np.uint8)
        dense = np.unpackbits(p, axis=1)[:, :w].astype(bool)
        return cls.from_crop(dense, 0, 0, (h, w))

    # ---------------- queries ----------------

    @property
    def is_empty(self) -> bool:
        return self.area == 0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """``(x, y, width, height)``, or all zeros when empty."""
        return (self.x0, self.y0, self.width, self.height)

    @property
    def nbytes(self) -> int:
        """Bytes of pixel payload held, for accounting and tests."""
        return 0 if self._packed is None else int(self._packed.nbytes)

    def crop_bool(self) -> np.ndarray:
        """The mask inside its bounding box, as ``(height, width)`` bool."""
        if self._packed is None:
            return np.zeros((0, 0), dtype=bool)
        return np.unpackbits(self._packed, axis=1)[:, : self.width].astype(bool)

    def packed_crop(self) -> np.ndarray:
        """The bit-packed crop itself, shape ``(height, ceil(width/8))``.

        The region's whole payload, for callers that persist it verbatim.
        """
        if self._packed is None:
            return np.zeros((0, 0), dtype=np.uint8)
        return self._packed

    def dense(self) -> np.ndarray:
        """The mask over the whole frame, as ``(H, W)`` bool.

        Allocates ``H * W`` bytes. Only the tracker (which takes full-frame
        masks) and legacy callers need this.
        """
        out = np.zeros(self.shape, dtype=bool)
        if self.area:
            out[self.y0 : self.y0 + self.height, self.x0 : self.x0 + self.width] = (
                self.crop_bool()
            )
        return out

    def packed_full(self) -> np.ndarray:
        """Whole-frame bit-packed mask, matching ``_pack_mask_bool``'s output.

        Builds the ``(H, ceil(W/8))`` result a box-height strip at a time, so
        no frame-sized boolean array is ever allocated.
        """
        h, w = self.shape
        out = np.zeros((h, (w + 7) // 8), dtype=np.uint8)
        if self.area:
            strip = np.zeros((self.height, w), dtype=np.uint8)
            strip[:, self.x0 : self.x0 + self.width] = self.crop_bool()
            out[self.y0 : self.y0 + self.height] = np.packbits(strip, axis=1)
        return out

    def stats(self) -> tuple[int, float, float, int, int, int, int] | None:
        """``(area_px, centroid_x, centroid_y, bbox_x, bbox_y, bbox_w, bbox_h)``.

        ``None`` for an empty mask, matching ``_mask_stats``.
        """
        if self.area == 0:
            return None
        ys, xs = np.nonzero(self.crop_bool())
        return (
            int(xs.size),
            round(float((xs + self.x0).mean()), 2),
            round(float((ys + self.y0).mean()), 2),
            self.x0,
            self.y0,
            self.width,
            self.height,
        )

    def cv_contours(self) -> list[np.ndarray]:
        """Outer contours in frame coordinates, as OpenCV point arrays.

        Found on the crop padded by one background pixel, so a mask touching
        its box edge is bounded the same way it would be inside a full frame,
        then offset back into frame coordinates by OpenCV itself.
        """
        if self.area == 0:
            return []
        import cv2  # local: keeps this module importable without OpenCV

        padded = np.zeros((self.height + 2, self.width + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = self.crop_bool()
        cnts, _ = cv2.findContours(
            padded,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
            offset=(self.x0 - 1, self.y0 - 1),
        )
        return list(cnts)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.area == 0:
            return f"MaskRegion(empty, shape={self.shape})"
        return (
            f"MaskRegion(shape={self.shape}, bbox=(x={self.x0}, y={self.y0}, "
            f"w={self.width}, h={self.height}), area={self.area}, "
            f"payload={self.nbytes}B)"
        )


def as_mask_region(mask, shape: tuple[int, int] | None = None) -> MaskRegion:
    """Coerce a mask to a :class:`MaskRegion`, passing one through unchanged.

    Lets every consumer take either a region or a plain array, so callers
    outside the pipeline keep working.
    """
    if isinstance(mask, MaskRegion):
        return mask
    return MaskRegion.from_dense(mask, shape)
