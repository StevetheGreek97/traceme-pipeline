import os
from pathlib import Path
from contextlib import nullcontext

import torch
import sam2

from traceme.core.logging import get_logger
from traceme.sam2.checkpoints import resolve_checkpoint, default_checkpoint_dir
from traceme.core.device import DEVICE_CHOICES, pick_device
log = get_logger("traceme.sam2.config")

# Optional NVML import (unchanged)
try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlSystemGetDriverVersion,
        nvmlDeviceGetCudaComputeCapability
    )
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

# ---------------------------
# Device + env summary helpers (unchanged)
# ---------------------------
def gpu_inventory_nvml():
    if not _HAS_NVML:
        return None
    try:
        nvmlInit()
        cnt = nvmlDeviceGetCount()
        items = []
        for i in range(cnt):
            h = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(h).decode()
            mem = nvmlDeviceGetMemoryInfo(h)
            cc_major, cc_minor = nvmlDeviceGetCudaComputeCapability(h)
            items.append({
                "index": i,
                "name": name,
                "total_mem_gb": round(mem.total / 1024**3, 2),
                "used_mem_gb": round(mem.used / 1024**3, 2),
                "free_mem_gb": round(mem.free / 1024**3, 2),
                "cc": f"{cc_major}.{cc_minor}",
            })
        drv = nvmlSystemGetDriverVersion().decode()
        return {"driver": drv, "gpus": items}
    except Exception as e:
        log.debug(f"NVML inventory failed: {e}")
        return None
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass

def gpu_inventory_torch():
    if not torch.cuda.is_available():
        return None
    out = {"driver": None, "gpus": []}
    try:
        out["driver"] = torch._C._cuda_getDriverVersion() / 1000.0
    except Exception:
        pass
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        out["gpus"].append({
            "index": i,
            "name": props.name,
            "total_mem_gb": round(props.total_memory / 1024**3, 2),
            "cc": f"{props.major}.{props.minor}",
        })
    return out

def log_runtime_summary(logger=log) -> None:
    cudnn_ver = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    logger.info(
        "Runtime: torch=%s cuda=%s cudnn=%s visible_gpus=%s",
        torch.__version__,
        torch.version.cuda,
        cudnn_ver,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )
    inv = gpu_inventory_nvml() or gpu_inventory_torch()
    if inv is None:
        logger.info("Runtime: no CUDA GPUs visible")
    else:
        drv = inv.get("driver")
        if drv is not None:
            logger.info("Runtime: nvidia_driver=%s", drv)
        for g in inv["gpus"]:
            mem = f"{g.get('total_mem_gb','?')} GB"
            cc = g.get("cc", "?")
            name = g.get("name", "?")
            idx = g.get("index", "?")
            extra = ""
            if "used_mem_gb" in g and "free_mem_gb" in g:
                extra = f" | used {g['used_mem_gb']} GB, free {g['free_mem_gb']} GB"
            logger.info("GPU %s: %s | CC %s | VRAM %s%s", idx, name, cc, mem, extra)

# ---------------------------
# Device + precision setup (unchanged)
# ---------------------------
DEVICE_PREFERENCE = os.environ.get("TRACEME_DEVICE", "auto")
device = pick_device(DEVICE_PREFERENCE)
if device.type == "cuda":
    current_idx = torch.cuda.current_device()
    log.debug("Using CUDA device %s: %s", current_idx, torch.cuda.get_device_name(current_idx))
elif device.type == "mps":
    log.debug("Using Apple Metal (MPS) device")
else:
    log.debug("Using CPU device")

AUTOCAST = nullcontext()
use_bf16 = False
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    if props.major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log.debug("Enabled TF32 for matmul and cuDNN on Ampere+")
    if torch.cuda.is_bf16_supported():
        use_bf16 = True
        AUTOCAST = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        log.debug("bfloat16 autocast is supported and will be used in AUTOCAST context")
    else:
        AUTOCAST = torch.autocast(device_type="cuda", dtype=torch.float16)
        log.debug("Using float16 autocast in AUTOCAST context")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
elif device.type == "mps":
    log.warning("MPS is experimental for this model and may be slower/different vs CUDA.")

# ---------------------------
# Hydra-friendly config resolution
# ---------------------------
SAM2_PKG_DIR = Path(sam2.__file__).resolve().parent   # .../site-packages/sam2/sam2
SAM2_ROOT    = SAM2_PKG_DIR.parent                    # .../site-packages/sam2
log.debug("Found SAM2 installed under: %s", SAM2_ROOT)

_override = os.getenv("SAM2_ROOT")
if _override:
    SAM2_ROOT = Path(_override).expanduser().resolve()
    SAM2_PKG_DIR = SAM2_ROOT / "sam2"                 # expected repo layout
    log.debug("SAM2_ROOT overridden via env var: %s", SAM2_ROOT)

MODEL_CFGS = {
    "tiny":      "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small":     "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large":     "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam3":      None,  # SAM3 needs no Hydra config; see traceme.sam2.sam3_backend
}

MODEL = os.environ.get("SAM2_MODEL", "large")
if MODEL not in MODEL_CFGS:
    raise ValueError(f"Unknown MODEL '{MODEL}'. Choose from: {list(MODEL_CFGS)}")
IS_SAM3 = MODEL == "sam3"
log.debug("Model: %s", MODEL)

SAM2_CHECKPOINTS = {
    "tiny":      SAM2_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
    "small":     SAM2_ROOT / "checkpoints" / "sam2.1_hiera_small.pt",
    "base_plus": SAM2_ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt",
    "large":     SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt",
    "sam3":      SAM2_ROOT / "checkpoints" / "sam3.pt",
}

# Absolute paths for validation
_cfg_rel  = MODEL_CFGS[MODEL]                            # what Hydra expects (relative)
_cfg_abs  = (SAM2_PKG_DIR / _cfg_rel) if _cfg_rel else None  # must exist (SAM2 only)
_ckpt_abs = SAM2_CHECKPOINTS[MODEL]                      # must exist
_ckpt_err = None

# An explicit SAM2_CHECKPOINT always wins (e.g. a checkpoint the TraceME app
# already downloaded), even if a same-named file exists under SAM2_ROOT.
_env_ckpt = os.getenv("SAM2_CHECKPOINT")
if _env_ckpt:
    _ckpt_abs = Path(_env_ckpt).expanduser()
elif not _ckpt_abs.exists():
    ckpt_dir_env = os.getenv("SAM2_CHECKPOINT_DIR")
    try:
        _ckpt_abs = resolve_checkpoint(
            MODEL,
            sam2_root=SAM2_ROOT,
            checkpoint_dir=Path(ckpt_dir_env).expanduser() if ckpt_dir_env else None,
        )
    except Exception as e:
        _ckpt_err = str(e)

missing = []
if _cfg_abs is not None and not _cfg_abs.exists():
    missing.append(("MODEL_CFG (expected relative)", _cfg_abs))
if not _ckpt_abs.exists():
    missing.append(("SAM2_CHECKPOINT", _ckpt_abs))

if missing:
    msg = [
        "Required files are missing:",
        *(f" - {name}: {pth}" for name, pth in missing),
        f"Resolved SAM2_ROOT: {SAM2_ROOT}",
        f"SAM2_PKG_DIR (Hydra CWD): {SAM2_PKG_DIR}",
        f"Default checkpoint cache: {default_checkpoint_dir()}",
        "Hints:",
        " - Hydra expects the YAML config path to be relative to a known search path.",
        " - We pass the *relative* path to Hydra and switch CWD to the package dir automatically.",
        " - If you overrode SAM2_ROOT, ensure it contains 'sam2/' and 'checkpoints/'.",
        " - Set SAM2_CHECKPOINT or SAM2_CHECKPOINT_DIR to point to checkpoints.",
        " - Or run: traceme-download-checkpoints",
        " - To disable auto-download, set TRACEME_AUTO_DOWNLOAD=0.",
        " - Example layout:",
        "     $SAM2_ROOT/",
        "       ├─ sam2/",
        "       │   └─ configs/sam2.1/sam2.1_hiera_*.yaml",
        "       └─ checkpoints/sam2.1_hiera_*.pt",
        "",
        "Troubleshooting:",
        " - Set HYDRA_FULL_ERROR=1 to see the full Hydra trace.",
        " - If running from a notebook, ensure the working directory is not read-only.",
    ]
    if _ckpt_err:
        msg.append(f"Checkpoint resolution error: {_ckpt_err}")
    raise FileNotFoundError("\n".join(msg))

log.debug("Config OK: %s", _cfg_abs)
log.debug("Checkpoint OK: %s", _ckpt_abs)

# Exported values used by the rest of your code
SEED_DIRNAME     = "_seeds"
MODEL_CFG        = _cfg_rel                 # keep this RELATIVE for Hydra
SAM2_CHECKPOINT  = str(_ckpt_abs)           # abs is fine for the checkpoint
  


def log_precision_summary(logger=log) -> None:
    logger.info(
        "Precision: device=%s tf32=%s bf16=%s",
        device.type,
        torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else False,
        use_bf16 if device.type == "cuda" else False,
    )


def log_model_summary(logger=log) -> None:
    logger.info(
        "Model config: model=%s cfg=%s ckpt=%s",
        MODEL,
        _cfg_abs if _cfg_abs is not None else "<none: sam3>",
        _ckpt_abs,
    )
