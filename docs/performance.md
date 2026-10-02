# Memory and speed

How TraceME holds masks, why it used to run out of memory on high-resolution
video, and how to size a run now. Written against v0.6.0.

## The problem this solves

The tracker returns one mask per object per frame, sized like the whole frame.
Held as a dense byte-per-pixel array, one mask of a 5312x2988 GoPro frame is
15.87 MB. The runner buffers a whole chunk before writing anything, and the
finalize thread keeps the previous chunk alive while the GPU tracks the next:

```
15.87 MB  x  12 objects  x  1000 frames  x  2 chunks  =  381 GB
```

On a node that grants 140.7 GiB that is an OOM kill partway through the first
chunk, which is exactly what happened on jobs 2934542 (`-c 1000`) and 2934597
(`-c 300`). Shrinking the chunk traded the crash for a much slower run: the
per-chunk overheads (building the inference state, re-seeding the overlap) do
not shrink with it, so 768 chunks of 100 frames cost more than they save.

What made it wasteful is that a tracked animal covers almost none of the frame.
Measured over a full run (`11_28_12b1M.csv`, 31,618 object-frames): **mean mask
area 804 px** of 15.9M, so better than 99.99% of every dense mask was zeros.
The masks are also upsampled — SAM2 decodes 256x256 logits and
`_get_orig_video_res_output` interpolates them up to the frame — so the dense
form carries no detail the small form lacks.

## How masks are held now

`traceme.sam2.masks.MaskRegion` stores a mask as its **bounding box plus the
bit-packed crop inside it**. Cost follows the object, not the frame:

| | one mask, 5312x2988 |
|---|---|
| dense `uint8` (before) | 15.87 MB |
| bit-packed whole frame | 1.98 MB |
| `MaskRegion` of an 800 px animal | **~180 bytes** |

Every region keeps a *tight* box — with a non-empty mask, its first and last
row and column each hold a set pixel — which is what lets `stats()` report the
bounding box without touching pixels.

The box comes from row and column projections reduced **on the tracker's own
device**, so only the pixels inside each box cross to host memory instead of a
frame-sized mask per object (`_collect_regions` in `sam2/runner.py`).

Consumers ask for what they need. Only two methods materialise anything
frame-sized, and both exist to meet a fixed external contract:
`dense()` feeds `add_new_mask`, which takes full-frame masks, and
`packed_full()` writes the `_masks.npz` archive.

## Measured effect

One frame, 5312x2988, 12 objects, ~800 px each — the GX060026 workload:

| | before | after | |
|---|---|---|---|
| masks held per frame | 190.47 MB | 2.10 KB | 90,000x smaller |
| a 1000-frame chunk, x2 in flight | 381 GB | 4.2 MB | |
| `_mask_stats` per frame | 256.8 ms | 0.24 ms | 1,000x faster |
| `_annotate_frame` per frame | 848 ms | 33 ms | 26x faster |
| overlap seeds, 20 frames (write / read) | 8.9 s / 15.2 s | 1 ms / 2 ms | |

`_mask_stats` and `_annotate_frame` got faster for the same reason the memory
shrank: both used to scan or index across the whole frame per object, and now
work inside the bounding box. Annotated video output is **pixel-identical** —
verified against the previous implementation over 66 renders covering
edge-touching, overlapping, full-frame, multi-region and sparse random masks.

## Sizing a run

Masks are no longer the binding constraint. **SAM2's own frame cache is.** With
`offload_video_to_cpu=True` the predictor keeps every frame of the chunk at the
model's input resolution as float32 RGB:

```
1024 x 1024 x 3 x 4 bytes  =  12.6 MB per frame
```

So host memory is now roughly `chunk_size x 12.6 MB`, independent of
resolution and object count:

With `--save-masks` a second, larger term appears, because the archive format
stores each mask packed over the *whole* frame:

```
archive peak = chunk_size x objects x H x ceil(W/8)
```

For 12 objects at 5312x2988 that is 23.8 MB per frame of chunk. Measured peak
RSS of `run_sam2` at that resolution, chunk held entirely in memory:

| `-c` | frame cache | tracked masks | `--save-masks` | measured peak |
|---|---|---|---|---|
| 240 | 3.0 GB | 0.5 MB | 5.7 GB | 6.95 GB |
| 300 | 3.8 GB | 0.6 MB | 7.1 GB | ~8.5 GB |
| 1000 | 12.6 GB | 2.1 MB | 23.8 GB | ~36 GB |
| 2000 | 25.2 GB | 4.2 MB | 47.6 GB | ~73 GB |
| 3000 | 37.7 GB | 6.3 MB | 71.4 GB | ~109 GB |

Without `--save-masks`, peak RSS measured **flat at 2.02 GB** across chunks of
60, 120 and 240 frames, where the dense representation would have needed
11.4, 22.9 and 45.7 GB. Chunk size is then effectively unconstrained.

Prefer large chunks within that budget. Total tracking time is fixed by the frame count, but every
chunk pays to build an inference state and re-seed its overlap, and every
overlap frame is tracked twice. Fewer, bigger chunks means less of both.

Keep `--overlap` small — it exists to carry object identity across the seam, and
a handful of frames does that. Cost is linear in it: re-seeding runs
`add_new_mask` once per object per overlap frame, each a forward pass through
the memory encoder.

## What deliberately did not change

- **`_masks.npz` layout.** Still one whole-frame bit-packed mask per
  (frame, object), because `traceme-app`
  (`src/services/tracking_results.py`) and downstream scripts read it. It stays
  the most expensive part of a run: the format obliges ~24 MB of
  mostly-zeros per frame through zlib. A crop-based archive would be ~100x
  cheaper but has to land together with its readers.

  One thing here *was* fixed, without changing what the file means. Both the
  chunk writer and the merge step built their object arrays with
  `np.array(masks, dtype=object)`, which broadcasts a list of equally-shaped
  arrays into one N-dimensional object array of scalars -- billions of Python
  pointers. Writing 240 masks cost 9.66 GB and 13.4 s; it now costs 2.02 GB
  and 4.2 s. Iterating either form yields the same per-mask arrays, so files
  written before the fix still read correctly. `object_array` in
  `sam2/io.py` carries the explanation, and
  `test_object_array_keeps_one_dimension` guards it.
- **CSV, contours JSONL, and the run summary.** Unchanged.
- **`_mask_stats`' array path** still binarises with `np.nonzero` (any nonzero)
  while everything else uses `> 0`. They disagree only on raw logits, which the
  pipeline never stores — it keeps `(logits > 0)`. Both behaviours are pinned by
  `test_mask_region_thresholds_strictly_positive`.
- **`_annotate_frame` still copies the frame twice** (`out`, then `overlay`).
  Removing the second copy means drawing into a shifted sub-window; the win is
  small next to the JPEG decode that dominates rendering at this resolution.

## Overlap seed files

A chunk saves the masks of its trailing overlap frames so the next chunk can
re-seed the same objects and keep their ids.

- **Format 1** stored each mask bit-packed over the whole frame, in nested
  pickled object arrays: ~2 MB and, worse, 8.9 s to write and 15.2 s to read
  per chunk at 5312x2988 with 12 objects and 20 overlap frames.
- **Format 2** (written since v0.6.0) stores each mask's box and packed crop,
  concatenated into one flat buffer with no pickled objects: 4.8 KB, ~1 ms.

`read_seed_file` accepts both, so a run started under an older version resumes
without reprocessing. Empty masks are kept, not dropped: re-seeding an object
the tracker lost is what keeps its id alive into the next chunk, and therefore
what keeps its `-1` rows in the CSV.

## Writing chunks out

`run_sam2(finalize_workers=...)` sets how many finished chunks may be written
while the GPU tracks ahead; it defaults to 2. One writer was the old bound
because each in-flight chunk cost a chunk of dense masks; now it costs
kilobytes, so a slow writer can overlap tracking instead of stalling it. This
matters most with `--save-masks`, which spends its time in zlib and releases
the GIL. Chunks are drained oldest-first, so `processed_chunks` stays in order.
