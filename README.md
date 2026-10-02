# TraceME Pipeline

TraceME is a video tracking pipeline that runs SAM2 on frame sequences, produces per-frame CSV summaries, and exports annotated videos.

## What You Get
- Annotated MP4 outputs per chunk (and a merged MP4).
- Per-frame CSVs per chunk (and a merged CSV).
- A CLI for running the pipeline over frame folders and prompt YAMLs.

## Requirements
- Python 3.12+
- SAM2 (installed automatically via pip)
- CUDA GPU recommended for speed (CPU is supported but slow)

## Install (PyPI)
```
pip install traceme-pipeline
```

### Checkpoints (Auto-Download)
TraceME automatically downloads the required SAM2 checkpoint on first run if it is missing.
By default, checkpoints are stored in:
```
~/.cache/traceme/sam2/checkpoints
```

You can also pre-download all checkpoints:
```
traceme-download-checkpoints
```

To change the checkpoint location:
- `SAM2_CHECKPOINT_DIR=/path/to/checkpoints`

To disable auto-download:
- `TRACEME_AUTO_DOWNLOAD=0`

### Dev Helper (SAM2 + Checkpoints)
If you prefer a local clone of SAM2 (for development), you can still use:
```
make install
```
This creates a venv, installs TraceME in editable mode, clones SAM2 into `third_party/sam2`, and downloads checkpoints.

### Configure SAM2 (optional)
If SAM2 lives somewhere else, set:
- `SAM2_ROOT=/path/to/sam2` (should contain `sam2/` and `checkpoints/`)
- `SAM2_MODEL=tiny|small|base_plus|large|sam3` (default: `large`)

### SAM3 (optional)
The pipeline can also run Meta's SAM3 tracker with the same points/boxes prompt
workflow:
```
pip install 'traceme-pipeline[sam3]'   # requires Python >= 3.12
traceme -i frames/ -o out/ -p prompts.yaml --model sam3
```
The `sam3.pt` checkpoint is resolved through the same logic as the SAM2
checkpoints (`SAM2_CHECKPOINT`, `SAM2_CHECKPOINT_DIR`, or the default cache
dir), but it is **not** auto-downloaded: it is gated on Hugging Face, so log
in with `hf auth login`, download it from the `facebook/sam3` repo, and place
it in your checkpoint directory as `sam3.pt`.

## Usage
Run the pipeline:
```
traceme -i /path/to/frames -o /path/to/output -p /path/to/prompts.yaml
```

Process only part of a video (frame numbers are 0-based positions in the sorted
frame folder; both ends inclusive, and either can be omitted):
```
traceme -i /path/to/frames -o /path/to/output -p prompts.yaml --start-frame 1200 --end-frame 2400
```
Prompts outside the range are ignored, and outputs keep the video's frame numbers
(`global_frame_idx` 1200 is frame 1200 of the video, not of the range).

Choose the compute device (default `auto`: CUDA, then Apple MPS, then CPU; an
unavailable choice falls back to CPU). Same as the `TRACEME_DEVICE` env var:
```
traceme ... --device cpu
```

Generate a tasks file for batch runs:
```
traceme-gen-tasks /path/to/root -o tasks.tsv
```

## Prompt YAML Format
Prompt files must contain a top-level `prompts` list. See `src/traceme/prompts/parser.py` for the exact schema and examples.

## Outputs
Given `frame_dir=/data/frames/clipA`, outputs are:
- `/output/clipA.csv` (merged CSV)
- `/output/clipA.mp4` (merged annotated video; one color and `id:` label per tracked object; exactly one video frame per input frame, so video frame N corresponds to `global_frame_idx` N)
- `/output/clipA_run_summary.json` (run status, processed/resumed/failed chunk ids, totals)
- `/output/clipA_tmp/` (intermediate chunk files; removed if `--del_tmp` is set). Chunk folders contain symlinks to the original frames (falling back to copies on filesystems without symlink support), so chunking costs almost no disk space.

CSV columns: `chunk_id, global_frame_idx, in_chunk_idx, obj_id, area_px, centroid_x, centroid_y, bbox_x, bbox_y, bbox_w, bbox_h`. Frames with no tracked objects produce a single row with an empty `obj_id` and `area_px=0`. If an object is tracked but the model loses its mask for a given frame (empty prediction), every stat column for that row is `-1`.

## Saving Masks (for shape analysis)
Pass `--save-masks` to persist every object's binary mask, bit-packed, into
`/output/clipA_masks.npz` — one merged archive per video (per-chunk masks are
written alongside each chunk's CSV/video during the run, then combined the
same way the merged CSV/video are, so `--del_tmp` does **not** remove them).
Only non-empty masks are stored. The archive layout is unchanged from
earlier versions -- one whole-frame bit-packed mask per (frame, object) --
so existing readers keep working. It is also the most expensive part of a
run: see [docs/performance.md](docs/performance.md).

Reload the mask archive:
```python
import numpy as np
from traceme.sam2.io import _unpack_mask

data = np.load("clipA_masks.npz", allow_pickle=True)
for gidx, oid, packed, shp in zip(
    data["global_frame_idx"], data["obj_id"], data["packed"], data["shape"]
):
    mask = _unpack_mask(packed, tuple(shp))  # bool array, shape (H, W)
    # e.g. shape descriptors via skimage:
    # from skimage.measure import regionprops, label
    # props = regionprops(label(mask))[0]
    # props.eccentricity, props.perimeter, props.solidity, ...
```

## Saving Contours (for visual review)
Pass `--save-contours` to write `/output/clipA_contours.jsonl`: one JSON line per
frame with each object's mask outline as polygons (all regions, simplified):
```
{"frame": 12, "objects": {"1": [[[x, y], ...]], "2": []}}
```
An empty list means the object was tracked but lost on that frame. The TraceME
app uses this file to overlay tracking results without loading full masks.

## Skipping the Video
By default every chunk is rendered to an annotated `.mp4` and merged into
`/output/clipA.mp4`. Pass `--no-video` to skip that when you only need the
CSV, masks or contours; on long videos or small chunks it saves a noticeable
amount of time. Turning the video back on for the same output folder
re-processes chunks that have no video yet.

## Chunk Size & Memory
Masks are held as a bounding box plus a bit-packed crop
(`traceme.sam2.masks.MaskRegion`), so tracking itself no longer scales with
resolution or object count. Two things still do:

```
SAM2's frame cache   ~= chunk_size x 12.6 MB        (1024x1024x3 float32 per frame)
--save-masks archive ~= chunk_size x objects x H x ceil(W/8)
```

The archive term dominates, because its file format stores each mask packed
over the *whole* frame. For 12 objects at 5312x2988 that is 23.8 MB per frame
of chunk, on top of 12.6 MB for the cache:

| `-c` | frame cache | `--save-masks` | total |
|---|---|---|---|
| 300 | 3.8 GB | 7.1 GB | ~11 GB |
| 1000 | 12.6 GB | 23.8 GB | ~36 GB |
| 2000 | 25.2 GB | 47.6 GB | ~73 GB |
| 3000 | 37.7 GB | 71.4 GB | ~109 GB |

Without `--save-masks`, only the first column applies and chunk size is
effectively unconstrained.

Prefer large chunks within that budget: total tracking time is set by the frame
count, but every chunk pays to build an inference state and re-seed its
overlap, and every overlap frame is tracked twice. Keep `--overlap` small -- a
handful of frames carries object identity across the seam, and re-seeding costs
a forward pass per object per overlap frame.

On SLURM, note that a job's memory limit is proportional to the cores it asks
for unless `--mem` is set, so `--cpus-per-task=8` of a 96-core node grants
1/12th of it.

See [docs/performance.md](docs/performance.md) for measurements and how masks
are stored.

## Resume & Failures
- Completed chunks are marked in the tmp folder; re-running the same command skips them and continues from the first incomplete chunk. Use `--no-resume` to reprocess everything.
- If the prompts, model, frame range or chunking changed since the previous run into the same output folder, its chunk outputs are discarded instead of resumed, so results never mix inputs.
- If any chunk fails, the pipeline still merges what it has, marks the run `"partial"` in the run summary, keeps the tmp folder (even with `--del_tmp`), and exits with code 1. Re-run the same command to retry only the failed chunks.
- Overlap seed files changed layout in v0.6.0 (bounding box plus packed crop, far smaller and faster). Seeds written by older versions are still read, so a run in progress resumes without reprocessing.

## Troubleshooting
- If SAM2 configs or checkpoints are missing, TraceME will raise a clear error with the expected paths.
- Set `HYDRA_FULL_ERROR=1` for detailed SAM2 errors.

## License
See `LICENSE`.
