import numpy as np
import pandas as pd
import pytest

from traceme.sam2.io import (
    _mask_stats,
    _pack_mask_bool,
    _unpack_mask,
    _write_csv_for_chunk,
    global_to_inchunk_idx,
    CSV_HEADER,
)
from traceme.prompts.parser import Prompt, YamlPromptParser
from traceme.video.chunker import VideoChunker, numeric_sort_key
from traceme.video.merge import merge_csv_chunks


# ---------------- mask stats & packing ----------------

def test_mask_stats_rectangle():
    m = np.zeros((16, 16), dtype=np.uint8)
    m[2:6, 3:9] = 1
    area, cx, cy, bx, by, bw, bh = _mask_stats(m[None, :, :])
    assert area == 24
    assert (bx, by, bw, bh) == (3, 2, 6, 4)
    assert (cx, cy) == (5.5, 3.5)


def test_mask_stats_empty():
    assert _mask_stats(np.zeros((8, 8))) is None


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(42)
    for shape in [(16, 16), (17, 31), (1, 64, 65)]:
        m = rng.random(shape) > 0.5
        packed, shp = _pack_mask_bool(m)
        out = _unpack_mask(packed, shp)
        assert out.shape == m.squeeze().shape
        assert np.array_equal(out, m.squeeze().astype(bool))


# ---------------- chunk index math consistency ----------------

@pytest.mark.parametrize("cs,ov", [(5, 1), (500, 1), (100, 5), (10, 0)])
def test_chunk_math_agrees(cs, ov):
    for fi in list(range(0, 3 * cs)) + [7 * cs - 1, 7 * cs]:
        chunks = YamlPromptParser.frame_to_chunks(fi, chunk_size=cs, overlap=ov)
        assert chunks, fi
        for c in chunks:
            start, end = YamlPromptParser.chunk_span(c, chunk_size=cs, overlap=ov)
            assert start <= fi <= end, (fi, c, start, end)
            in_idx = global_to_inchunk_idx(fi, c, cs, ov)
            assert 0 <= in_idx <= end - start, (fi, c, in_idx)
            assert start + in_idx == fi


def test_numeric_sort_key_padding_rollover(tmp_path):
    names = [f"chunk_{i}" for i in (99, 100, 1000, 2)]
    paths = [tmp_path / f"{n}.csv" for n in names]
    ordered = sorted(paths, key=numeric_sort_key)
    assert [p.stem for p in ordered] == ["chunk_2", "chunk_99", "chunk_100", "chunk_1000"]


# ---------------- prompt parser ----------------

def _write_yaml(tmp_path, text):
    p = tmp_path / "prompts.yaml"
    p.write_text(text)
    return p


def test_parser_valid(tmp_path):
    p = _write_yaml(
        tmp_path,
        "prompts:\n"
        "  - frame_idx: 3\n"
        "    obj_id: 1\n"
        "    box: [10, 20, 30, 40]\n"
        "    points: [[1, 2], [3, 4]]\n"
        "    labels: [0, 1]\n",
    )
    prompts = YamlPromptParser(p).load()
    assert prompts == [
        Prompt(frame_idx=3, obj_id=1, box=(10, 20, 30, 40), points=((1, 2), (3, 4)), labels=(0, 1))
    ]


def test_parser_zero_size_box_dropped(tmp_path):
    p = _write_yaml(
        tmp_path,
        "prompts:\n"
        "  - {frame_idx: 0, obj_id: 1, box: [5, 5, 0, 10], points: [[1, 1]], labels: [1]}\n",
    )
    assert YamlPromptParser(p).load()[0].box is None


def test_parser_missing_key(tmp_path):
    p = _write_yaml(tmp_path, "prompts:\n  - {frame_idx: 0, obj_id: 1, points: []}\n")
    with pytest.raises(ValueError, match="labels"):
        YamlPromptParser(p).load()


def test_parser_strict_unknown_key(tmp_path):
    p = _write_yaml(
        tmp_path,
        "prompts:\n  - {frame_idx: 0, obj_id: 1, points: [], labels: [], bogus: 1}\n",
    )
    with pytest.raises(ValueError, match="bogus"):
        YamlPromptParser(p).load(strict_keys=True)


def test_prompts_by_chunk_boundary_duplication():
    prompts = [Prompt(frame_idx=499, obj_id=1, box=None, points=((1, 1),), labels=(1,))]
    # with the real frame count, the boundary prompt lands in both chunks
    buckets = YamlPromptParser.prompts_by_chunk(
        prompts, chunk_size=500, overlap=1, total_frames=600
    )
    assert set(buckets) == {0, 1}
    # without it, the chunk count is bounded by the highest prompted frame
    buckets = YamlPromptParser.prompts_by_chunk(prompts, chunk_size=500, overlap=1)
    assert set(buckets) == {0}


# ---------------- CSV writer + merge ----------------

def test_csv_write_and_merge(tmp_path):
    files_dir = tmp_path / "files"
    stats0 = {
        0: {1: (24, 5.5, 3.5, 3, 2, 6, 4)},
        1: {},                     # empty frame
        4: {1: (24, 5.5, 3.5, 3, 2, 6, 4), 2: None},  # vanished object
    }
    stats1 = {0: {1: (24, 6.5, 3.5, 4, 2, 6, 4)}}  # in-chunk 0 == global 4 (overlap dup)
    _write_csv_for_chunk(files_dir / "clip_chunk_000.csv", stats0, cid=0, cs=5, ov=1)
    _write_csv_for_chunk(files_dir / "clip_chunk_001.csv", stats1, cid=1, cs=5, ov=1)

    merged = tmp_path / "clip.csv"
    merge_csv_chunks(files_dir, merged)
    df = pd.read_csv(merged)

    assert list(df.columns) == CSV_HEADER
    # boundary frame 4 deduped: obj 1 kept once (first chunk's row wins)
    f4_obj1 = df[(df["global_frame_idx"] == 4) & (df["obj_id"] == 1)]
    assert len(f4_obj1) == 1
    assert f4_obj1.iloc[0]["chunk_id"] == 0
    # empty frame kept with area 0 and blank obj_id
    f1 = df[df["global_frame_idx"] == 1]
    assert len(f1) == 1 and f1.iloc[0]["area_px"] == 0 and pd.isna(f1.iloc[0]["obj_id"])
    # vanished (lost) object row: sentinel -1 across all stat columns
    van = df[(df["global_frame_idx"] == 4) & (df["obj_id"] == 2)]
    assert van.iloc[0]["area_px"] == -1 and van.iloc[0]["bbox_x"] == -1


# ---------------- chunker ----------------

def _make_frames(d, n):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"{i:05d}.jpg").write_bytes(b"\xff\xd8fake")
    return d


def test_chunker_symlink_build_and_reuse(tmp_path):
    frames = _make_frames(tmp_path / "frames", 9)
    out = tmp_path / "chunks"
    ch = VideoChunker(frame_dir=frames, output_dir=out, chunk_size=5, overlap=1, action="symlink")
    dirs = ch.chunk_frames(mode="auto")
    assert [d.name for d in dirs] == ["chunk_000", "chunk_001"]
    assert len(ch.get_frame_paths(0)) == 5
    assert len(ch.get_frame_paths(1)) == 5  # 1 overlap + 4
    assert (out / "chunk_001" / "00004.jpg").is_symlink()

    # second run with identical params reuses the manifest
    ch2 = VideoChunker(frame_dir=frames, output_dir=out, chunk_size=5, overlap=1, action="symlink")
    assert ch2._manifest_reason()[0] == "valid"

    # param change invalidates
    ch3 = VideoChunker(frame_dir=frames, output_dir=out, chunk_size=4, overlap=1, action="symlink")
    assert ch3._manifest_reason()[0] == "param_mismatch"


# ---------------- contours / device ----------------

def test_mask_contours_all_regions():
    from traceme.sam2.io import _mask_contours

    m = np.zeros((20, 30), dtype=np.uint8)
    m[2:6, 3:9] = 1
    m[10:15, 20:25] = 1
    polys = _mask_contours(m[None])
    assert len(polys) == 2  # both parts kept, not just the largest
    assert _mask_contours(np.zeros((5, 5))) == []


def test_chunker_frame_range(tmp_path):
    frames = _make_frames(tmp_path / "frames", 9)
    ch = VideoChunker(frame_dir=frames, output_dir=tmp_path / "c", chunk_size=4, overlap=1,
                      start_frame=2, end_frame=7)
    assert [p.name for p in ch.frame_paths] == [f"{i:05d}.jpg" for i in range(2, 8)]
    assert ch.total_source_frames == 9
    ch.chunk_frames()
    assert ch.count_chunks() == 2

    whole = VideoChunker(frame_dir=frames, output_dir=tmp_path / "w", chunk_size=4, overlap=1)
    assert (whole.start_frame, whole.end_frame) == (0, 8)
    with pytest.raises(ValueError, match="outside the video"):
        VideoChunker(frame_dir=frames, output_dir=tmp_path / "x", start_frame=3, end_frame=9)


@pytest.mark.parametrize(
    "cuda, mps, pref, expected",
    [
        (True, True, "auto", "cuda"),
        (False, True, "auto", "mps"),
        (False, False, "auto", "cpu"),
        (True, False, "cpu", "cpu"),
        (False, False, "cuda", "cpu"),
        (False, True, "mps", "mps"),
    ],
)
def test_pick_device(monkeypatch, cuda, mps, pref, expected):
    import torch
    from traceme.core import device as dev

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(dev, "_mps_available", lambda: mps)
    assert dev.pick_device(pref).type == expected
    with pytest.raises(ValueError):
        dev.pick_device("tpu")


# ---------------- compact mask regions ----------------

def _dense_cases():
    rng = np.random.default_rng(11)
    m = np.zeros((40, 60), bool); m[5:12, 7:20] = True
    yield "rect", m
    m = np.zeros((40, 60), bool); m[2:8, 3:9] = True; m[25:35, 40:55] = True
    yield "two_blobs", m
    m = np.zeros((40, 60), bool); m[13, 21] = True
    yield "one_px", m
    yield "empty", np.zeros((40, 60), bool)
    yield "full", np.ones((40, 60), bool)
    m = np.zeros((40, 60), bool)
    m[0, :] = True; m[:, 0] = True; m[-1, :] = True; m[:, -1] = True
    yield "border_ring", m
    m = np.zeros((50, 50), bool); m[10:40, 10:40] = True; m[20:30, 20:30] = False
    yield "donut", m
    for w in (1, 7, 8, 9, 31, 33):  # non byte-aligned widths
        yield f"rand_w{w}", rng.random((23, w)) > 0.5
    yield "sparse_rand", rng.random((97, 131)) > 0.93


@pytest.mark.parametrize("name,dense", list(_dense_cases()), ids=lambda v: v if isinstance(v, str) else "")
def test_mask_region_matches_dense_helpers(name, dense):
    """A region must answer exactly as the whole-frame helpers do."""
    from traceme.sam2.io import _mask_contours
    from traceme.sam2.masks import MaskRegion

    region = MaskRegion.from_dense(dense)
    assert region.stats() == _mask_stats(dense)
    assert np.array_equal(region.dense(), dense)
    # the archive format must stay byte-identical to _pack_mask_bool
    packed, shape = _pack_mask_bool(dense)
    assert region.shape == shape
    assert np.array_equal(region.packed_full(), packed)
    # contours agree with tracing the full frame
    assert _mask_contours(region) == _mask_contours(dense)
    # (1,H,W) inputs behave the same
    assert MaskRegion.from_dense(dense[None]).stats() == region.stats()
    # and the on-disk packed form round-trips back
    assert np.array_equal(MaskRegion.from_packed_full(packed, shape).dense(), dense)
    # as does the crop form
    back = MaskRegion.from_packed_crop(
        region.packed_crop(), region.y0, region.x0,
        region.height, region.width, region.shape,
    )
    assert np.array_equal(back.dense(), dense)


def test_mask_region_payload_follows_object_not_frame():
    """The point of the class: cost scales with the object, not the frame."""
    from traceme.sam2.masks import MaskRegion

    frame = (2988, 5312)  # the resolution that exhausted host memory
    dense_bytes = frame[0] * frame[1]
    m = np.zeros(frame, bool)
    m[1000:1030, 2000:2040] = True  # a 30x40 animal, 1200 px
    region = MaskRegion.from_dense(m)
    assert region.area == 1200
    assert region.bbox == (2000, 1000, 40, 30)
    assert region.nbytes < 1024                    # ~150 bytes, not 15.9 MB
    assert region.nbytes < dense_bytes / 10_000

    # worst case (whole frame set) is still no worse than bit-packing it
    full = MaskRegion.from_dense(np.ones(frame, bool))
    assert full.nbytes <= dense_bytes // 8 + frame[0]


def test_mask_region_thresholds_strictly_positive():
    """A region binarises with `> 0`, the same rule `_pack_mask_bool` uses.

    `_mask_stats`' array path instead calls `np.nonzero`, which counts
    negative values as set. The two have always disagreed on raw logits; it
    never showed because the runner stores `(logits > 0)`, already 0/1, and
    on such masks every path agrees. This test pins both behaviours so the
    difference stays deliberate.
    """
    from traceme.sam2.masks import MaskRegion

    logits = np.array([[-2.0, 0.0, 3.0], [-1.0, 0.5, -0.1]])
    assert MaskRegion.from_dense(logits).area == 2   # positives only
    assert _mask_stats(logits)[0] == 5               # legacy: any nonzero

    # on what the pipeline really produces, they agree exactly
    thresholded = logits > 0
    assert MaskRegion.from_dense(thresholded).stats() == _mask_stats(thresholded)


def test_mask_archive_same_for_regions_and_arrays(tmp_path):
    """--save-masks output must not depend on how masks were held in memory."""
    from traceme.sam2.io import _write_masks_for_chunk
    from traceme.sam2.masks import MaskRegion

    rng = np.random.default_rng(3)
    dense = {
        0: {1: rng.random((30, 45)) > 0.8, 2: np.zeros((30, 45), bool)},  # obj 2 lost
        1: {1: rng.random((30, 45)) > 0.8},
    }
    regions = {
        f: {o: MaskRegion.from_dense(m) for o, m in objs.items()}
        for f, objs in dense.items()
    }

    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_masks_for_chunk(a, dense, cid=0, cs=5, ov=0)
    _write_masks_for_chunk(b, regions, cid=0, cs=5, ov=0)

    da, db = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    assert set(da.files) == set(db.files)
    assert np.array_equal(da["global_frame_idx"], db["global_frame_idx"])
    assert np.array_equal(da["obj_id"], db["obj_id"])
    # the lost object contributes no entry, in either representation
    assert list(da["obj_id"]) == [1, 1]
    for pa, pb in zip(da["packed"], db["packed"]):
        assert np.array_equal(pa, pb)
    for sa, sb in zip(da["shape"], db["shape"]):
        assert tuple(sa) == tuple(sb)


# ---------------- overlap seed files ----------------

def test_seed_file_roundtrip(tmp_path):
    from traceme.sam2.io import read_seed_file, write_seed_file
    from traceme.sam2.masks import MaskRegion

    shape = (40, 60)
    m1 = np.zeros(shape, bool); m1[5:9, 11:17] = True
    m2 = np.zeros(shape, bool); m2[30:36, 50:58] = True
    entries = [
        (0, 1, MaskRegion.from_dense(m1)),
        (0, 2, MaskRegion.empty(shape)),   # tracked but lost
        (1, 1, MaskRegion.from_dense(m2)),
    ]
    path = tmp_path / "seed.npz"
    assert write_seed_file(path, entries, shape) == 3

    got = read_seed_file(path)
    assert [(r, o) for r, o, _ in got] == [(0, 1), (0, 2), (1, 1)]
    assert np.array_equal(got[0][2].dense(), m1)
    assert got[1][2].is_empty          # a lost object survives the round trip
    assert np.array_equal(got[2][2].dense(), m2)
    # compact: three masks over a 40x60 frame in well under one dense mask
    assert path.stat().st_size < shape[0] * shape[1]


def test_seed_reader_accepts_legacy_format(tmp_path):
    """Seeds written by an older version must still resume a run."""
    from traceme.sam2.io import read_seed_file

    shape = (40, 60)
    m1 = np.zeros(shape, bool); m1[5:9, 11:17] = True
    m2 = np.zeros(shape, bool); m2[30:36, 50:58] = True
    p1, s1 = _pack_mask_bool(m1)
    p2, s2 = _pack_mask_bool(m2)

    path = tmp_path / "legacy.npz"
    np.savez_compressed(                     # exactly the format 1 layout
        path,
        rel_indices=np.array([0, 1], dtype=np.int32),
        obj_ids=np.array([np.array([1, 2], dtype=np.int32),
                          np.array([1], dtype=np.int32)], dtype=object),
        packed=np.array([np.array([p1, p2], dtype=object),
                         np.array([p2], dtype=object)], dtype=object),
        shapes=np.array([np.array([s1, s2], dtype=object),
                         np.array([s2], dtype=object)], dtype=object),
    )

    got = read_seed_file(path)
    assert [(r, o) for r, o, _ in got] == [(0, 1), (0, 2), (1, 1)]
    assert np.array_equal(got[0][2].dense(), m1)
    assert np.array_equal(got[1][2].dense(), m2)


# ---------------- the propagate-loop collector ----------------

def test_collect_regions_from_logits():
    """Boxes come from device-side projections; only the crop is copied."""
    import torch

    from traceme.sam2.runner import _collect_regions

    logits = torch.full((3, 1, 24, 32), -1.0)
    logits[0, 0, 4:9, 6:14] = 1.0      # object 5
    logits[1, 0, 20:24, 28:32] = 1.0   # object 6, against the corner
    #  object 7 stays negative -> lost on this frame
    segs = _collect_regions([5, 6, 7], logits)

    assert set(segs) == {5, 6, 7}
    assert segs[5].bbox == (6, 4, 8, 5) and segs[5].area == 40
    assert segs[6].bbox == (28, 20, 4, 4) and segs[6].area == 16
    assert segs[7].is_empty
    assert segs[7].shape == (24, 32)
    # equivalent to thresholding the whole frame
    assert segs[5].stats() == _mask_stats((logits[0] > 0).numpy())
    # (N,H,W) logits are accepted too
    assert _collect_regions([5], (logits[0] > 0))[5].bbox == (6, 4, 8, 5)


def test_annotate_frame_region_and_array_agree():
    """Rendering must not depend on how the mask was held."""
    from traceme.sam2.masks import MaskRegion
    from traceme.video.frames import _annotate_frame

    rng = np.random.default_rng(5)
    frame = rng.integers(0, 256, (90, 120, 3), dtype=np.uint8)
    masks = [np.zeros((90, 120), bool) for _ in range(3)]
    masks[0][10:25, 15:40] = True
    masks[1][0:6, 0:6] = True              # against the frame corner
    masks[2][80:90, 110:120] = True        # against the far corner
    ids = [1, 4, 9]

    from_arrays = _annotate_frame(frame.copy(), ids, [m[None] for m in masks])
    from_regions = _annotate_frame(
        frame.copy(), ids, [MaskRegion.from_dense(m) for m in masks]
    )
    assert np.array_equal(from_arrays, from_regions)
    # an all-empty frame is returned untouched
    empty = [MaskRegion.empty((90, 120))] * 3
    assert np.array_equal(_annotate_frame(frame.copy(), ids, empty), frame)


def test_object_array_keeps_one_dimension():
    """Guard the numpy trap that made --save-masks cost ~8x its own data.

    `np.array(list_of_equal_shaped_arrays, dtype=object)` broadcasts into an
    N-dimensional object array of scalars. The archive writer needs a flat
    array of references.
    """
    from traceme.sam2.io import object_array

    masks = [np.zeros((4, 6), dtype=np.uint8) for _ in range(3)]

    trap = np.array(masks, dtype=object)
    assert trap.shape == (3, 4, 6), "if numpy stops doing this, the guard can go"

    arr = object_array(masks)
    assert arr.shape == (3,)
    assert arr[0] is masks[0], "elements must be the arrays themselves"
    # and both still iterate to the same per-mask content, so old files read
    for a, b in zip(trap, arr):
        assert np.array_equal(np.asarray(a, dtype=np.uint8), b)


def test_object_array_handles_empty_and_ragged():
    from traceme.sam2.io import object_array

    assert object_array([]).shape == (0,)
    ragged = [np.zeros(2, np.uint8), np.zeros(5, np.uint8)]
    arr = object_array(ragged)
    assert arr.shape == (2,) and arr[1].shape == (5,)
