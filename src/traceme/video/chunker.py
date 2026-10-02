from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Dict, Any, Optional
import shutil
import re
import json
from datetime import datetime, timezone
from traceme.core.logging import get_logger, timer

log = get_logger("traceme.video.chunker")

_IMG_EXTS = {".jpg", ".jpeg", ".png"}
_MANIFEST = ".chunks_manifest.json"


def numeric_sort_key(p: Path) -> int:
    """
    Sort key based on the trailing integer in a path's stem (e.g. "chunk_007"
    -> 7, "vid_chunk_1000.csv" -> 1000). Plain string sort breaks once
    zero-padded indices roll past their padding width (e.g. "chunk_1000"
    sorts before "chunk_099"), so anything ordered by chunk/frame index
    should use this instead of `sorted()`'s default string comparison.
    """
    stem = p.stem
    if stem.isdigit():
        return int(stem)
    m = re.findall(r"\d+", stem)
    if not m:
        raise ValueError(f"Filename has no digits for numeric sort: {p.name}")
    return int(m[-1])



@dataclass
class VideoChunker:
    frame_dir: Path
    output_dir: Path
    chunk_size: int = 1000
    overlap: int = 2
    action: str = "symlink"  # "copy", "move" or "symlink"
    remove_org: bool = False
    # Inclusive range of source frames to chunk (by position in the sorted
    # frame list). Defaults to the whole video; end_frame is resolved to the
    # last frame's index once frames are listed.
    start_frame: int = 0
    end_frame: Optional[int] = None

    def __post_init__(self) -> None:
        self.frame_dir = Path(self.frame_dir)
        self.output_dir = Path(self.output_dir)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.overlap < 0:
            raise ValueError("overlap must be >= 0")
        if self.action not in {"copy", "move", "symlink"}:
            raise ValueError('action must be "copy", "move" or "symlink"')

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Frames may not exist if user previously moved them.
        if self.frame_dir.exists():
            all_frames: List[Path] = sorted(
                (p for p in self.frame_dir.iterdir() if p.suffix.lower() in _IMG_EXTS),
                key=numeric_sort_key
            )
        else:
            all_frames = []
        self.total_source_frames = len(all_frames)
        self._frame_paths = self._select_range(all_frames)

    def _select_range(self, frames: List[Path]) -> List[Path]:
        if self.start_frame < 0:
            raise ValueError(f"start_frame must be >= 0, got {self.start_frame}")
        if not frames:
            return []
        n = len(frames)
        if self.end_frame is None:
            self.end_frame = n - 1
        if not (self.start_frame <= self.end_frame < n):
            raise ValueError(
                f"Frame range {self.start_frame}-{self.end_frame} is outside the video "
                f"(frames 0-{n - 1} in {self.frame_dir})."
            )
        return frames[self.start_frame:self.end_frame + 1]

    # -------------------- Public API --------------------

    @property
    def frame_paths(self) -> List[Path]:
        return self._frame_paths

    # --- update total_chunks property to respect param changes ---
    @property
    def total_chunks(self) -> int:
        n = len(self._frame_paths)
        if n:
            return (n + self.chunk_size - 1) // self.chunk_size
        # If no frames present, only trust manifest when fully valid.
        reason = self._manifest_reason()
        if reason and reason[0] == "valid":
            return self._manifest_data().get("num_chunks", 0)
        return 0

    # --- replace the start of chunk_frames(...) with this logging-aware version ---
    def chunk_frames(self, mode: Literal["auto", "load", "force"] = "auto") -> List[Path]:
        """
        Create or load chunk folders based on `mode`.
        Returns a list of chunk directories.
        """
        log.info(
            f"chunk_frames(mode={mode}) | "
            f"chunk_size={self.chunk_size} | overlap={self.overlap} | action={self.action} | "
            f"frames_detected={len(self._frame_paths)}"
        )

        if mode == "force":
            log.info("Mode=force → clearing existing chunk folders and manifest")
            self._clear_existing_chunks()

        if mode in {"auto", "load"}:
            reason = self._manifest_reason()
            log.debug(f"Manifest reason: {reason}")

            if reason[0] == "valid":
                log.info("Existing chunks + manifest are valid → reusing chunk folders")
                return self.get_chunk_dirs()

            if mode == "load":
                why = {
                    "missing": "manifest missing",
                    "param_mismatch": f"parameter mismatch ({reason[1]}: was {reason[2]}, now {reason[3]})",
                    "chunks_mismatch": f"chunk folders differ (expected {reason[1]}, found {reason[2]})",
                    "frames_changed": "frame list changed",
                }.get(reason[0], "unknown reason")
                log.error(f"Mode=load but chunks not reusable ({why}).")
                raise FileNotFoundError(
                    f"Chunks not reusable ({why}). Use mode='auto' or 'force' to (re)create."
                )

            # mode == "auto" and manifest is not valid:
            if reason[0] in {"param_mismatch", "frames_changed"} and self._frame_paths:
                log.info(f"Parameters/frames changed ({reason}); rebuilding chunks from source frames")
                self._clear_existing_chunks()
            elif reason[0] in {"param_mismatch", "frames_changed"} and not self._frame_paths:
                key = reason[1] if reason[0] == "param_mismatch" else "frames"
                details = (f"{key}: was {reason[2]}, now {reason[3]}"
                        if reason[0] == "param_mismatch" else "frame list differs")
                log.error(
                    "Cannot rebuild chunks because source frames are not available. "
                    f"Reason: {details}"
                )
                raise FileNotFoundError(
                    "Cannot rebuild chunks because source frames are not available.\n"
                    f"- Reason: {details}\n"
                    f"- Your options:\n"
                    f"  * Re-run with mode='load' to reuse existing chunks (will keep old chunking), or\n"
                    f"  * Restore frames to {self.frame_dir} and re-run (auto/force) to rebuild."
                )
            elif reason[0] == "chunks_mismatch":
                if self._frame_paths:
                    log.info(
                        f"Chunk folders incomplete (expected {reason[1]}, found {reason[2]}). "
                        "Rebuilding from source frames."
                    )
                    self._clear_existing_chunks()
                else:
                    log.error(
                        "Chunk folders are incomplete and source frames are missing; cannot rebuild."
                    )
                    raise FileNotFoundError(
                        "Chunk folders are incomplete, and source frames are not available to rebuild.\n"
                        f"Expected {reason[1]} folders but found {reason[2]}.\n"
                        f"Use mode='load' only if all chunk folders exist, or restore frames and re-run."
                    )
            # if reason[0] == "missing": fall through to build

        # --- (Re)build chunks from frames ---
        if not self._frame_paths:
            log.error(f"No source frames found in {self.frame_dir}")
            raise FileNotFoundError(
                f"No source frames found in {self.frame_dir}. "
                f"If you previously moved them, use mode='load' to reuse existing chunks."
            )

        chunk_dirs: List[Path] = []
        n = len(self._frame_paths)
        log.info(f"Building chunks from {n} frames (chunk_size={self.chunk_size}, overlap={self.overlap})")

        with timer(log, "build_chunks"):
            start = 0
            for i in range((n + self.chunk_size - 1) // self.chunk_size):
                end = min(start + self.chunk_size, n)
                ovl_start = max(0, start - self.overlap) if i > 0 and self.overlap > 0 else start
                frames = self._frame_paths[ovl_start:end]

                chunk_dir = self.output_dir / f"chunk_{i:03d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                self._transfer(frames, chunk_dir)
                chunk_dirs.append(chunk_dir)

                log.debug(
                    f"Created {chunk_dir.name}: frames[{ovl_start}:{end}) → "
                    f"count={len(frames)} (overlap applied={i>0 and self.overlap>0})"
                )
                start = end

        if self.action == "symlink" and self.remove_org:
            log.warning("remove_org ignored: chunks are symlinks into the original folder")
        elif self.action == "copy" and self.remove_org:
            log.info(f"remove_org=True → deleting original folder {self.frame_dir}")
            self._safe_rmtree(self.frame_dir)

        manifest_payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "action": self.action,
            "num_chunks": len(chunk_dirs),
            "source_dir": str(self.frame_dir),
            "frame_range": [self.start_frame, self.end_frame],
            "frame_names": [p.name for p in self._frame_paths],
        }
        self._write_manifest(manifest_payload)
        log.info(
            f"Wrote manifest to {self._manifest_path().name} "
            f"(num_chunks={len(chunk_dirs)}, first_chunk={chunk_dirs[0].name if chunk_dirs else 'n/a'})"
        )

        return chunk_dirs

    def _manifest_reason(self) -> tuple:
        """
        Returns a tuple describing manifest state:
        ('valid',) |
        ('missing',) |
        ('param_mismatch', key, old, new) |
        ('chunks_mismatch', expected, found) |
        ('frames_changed',)   # frame list differs (when source frames are present)
        """
        data = self._manifest_data()
        if not data:
            return ("missing",)

        # Parameter checks
        for key in ("chunk_size", "overlap", "action"):
            old = data.get(key, None)
            new = getattr(self, key)
            if old != new:
                return ("param_mismatch", key, old, new)

        # Folder count check
        expected = data.get("num_chunks", 0)
        found = self.count_chunks()
        if expected <= 0 or expected != found:
            return ("chunks_mismatch", expected, found)

        # If we still have source frames, ensure the ordered list matches
        if self._frame_paths:
            old_names = data.get("frame_names", [])
            new_names = [p.name for p in self._frame_paths]
            if old_names != new_names:
                return ("frames_changed",)

        return ("valid",)


    def get_chunk_dir(self, chunk_index: int) -> Path:
        chunk_dir = self.output_dir / f"chunk_{chunk_index:03d}"
        if not chunk_dir.exists():
            raise FileNotFoundError(f"Chunk folder does not exist: {chunk_dir}")
        return chunk_dir

    def get_frame_paths(self, chunk_index: int) -> List[Path]:
        chunk_dir = self.get_chunk_dir(chunk_index)
        return sorted(
            (p for p in chunk_dir.iterdir() if p.suffix.lower() in _IMG_EXTS),
            key=numeric_sort_key
        )

    def get_frame_names(self, chunk_index: int) -> List[str]:
        return [p.name for p in self.get_frame_paths(chunk_index)]

    def get_chunk_dirs(self) -> List[Path]:
        return sorted(
            (d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")),
            key=numeric_sort_key,
        )

    def count_chunks(self) -> int:
        return len(self.get_chunk_dirs())

    def get_all_frames_flat(self) -> List[Path]:
        # If frames no longer in `frame_dir` (e.g., after move), fall back to manifest names
        if self._frame_paths:
            return list(self._frame_paths)
        data = self._manifest_data()
        names = data.get("frame_names", [])
        # Rebuild as paths under original dir (may not exist) just to keep API compatible
        return [self.frame_dir / n for n in names]

    # -------------------- Internals --------------------

    def _transfer(self, frames: Iterable[Path], dst_dir: Path) -> None:
        warned_symlink = False
        for src in frames:
            dst = dst_dir / src.name
            if dst.exists():  # idempotent on re-run
                continue
            if self.action == "symlink":
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    # Filesystem without symlink support (or no permission):
                    # fall back to copying, once per chunk with a warning.
                    if not warned_symlink:
                        log.warning(
                            f"Symlinks not supported for {dst_dir}; falling back to copy."
                        )
                        warned_symlink = True
                    shutil.copy2(src, dst)
            elif self.action == "copy":
                shutil.copy2(src, dst)
            else:  # move
                shutil.move(src, dst)

    @staticmethod
    def _safe_rmtree(path: Path) -> None:
        try:
            shutil.rmtree(path)
            log.info(f"Deleted original folder: {path}")
        except Exception as e:
            log.warning(f"Could not delete original folder '{path}': {e}")

    # ----- manifest helpers -----

    def _manifest_path(self) -> Path:
        return self.output_dir / _MANIFEST

    def _write_manifest(self, data: Dict[str, Any]) -> None:
        with self._manifest_path().open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _manifest_data(self) -> Dict[str, Any]:
        p = self._manifest_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _clear_existing_chunks(self) -> None:
        for d in self.get_chunk_dirs():
            shutil.rmtree(d, ignore_errors=True)
        # keep manifest until we rewrite it (or remove to be clean)
        mp = self._manifest_path()
        if mp.exists():
            mp.unlink()
