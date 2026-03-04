"""Environment information capture for reproducibility."""

from __future__ import annotations

import datetime
import subprocess
import sys

from .types import EnvInfo


def capture_env(cmdline: str = "") -> EnvInfo:
    info = EnvInfo()
    info.timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    info.cmdline = cmdline

    # Torch
    try:
        import torch
        info.torch_version = torch.__version__
        if torch.cuda.is_available():
            info.cuda_version = torch.version.cuda or ""
            info.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info.gpu_driver = f"SM {props.major}.{props.minor}"
    except Exception:
        pass

    # Triton
    try:
        import triton
        info.triton_version = triton.__version__
    except Exception:
        pass

    # Git
    try:
        info.git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info.git_dirty = bool(dirty)
    except Exception:
        pass

    return info
