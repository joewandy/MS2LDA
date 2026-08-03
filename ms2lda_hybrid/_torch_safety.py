"""Shared PyTorch version guard for checkpoint deserialization."""

from __future__ import annotations

import re
from typing import Any

MINIMUM_SAFE_TORCH_VERSION = (2, 6, 0)


def require_patched_torch(torch_module: Any, *, operation: str) -> None:
    """Reject PyTorch releases affected by CVE-2025-32434."""
    match = re.match(
        r"^(\d+)\.(\d+)\.(\d+)(?:(a|b|rc|dev)\d*)?",
        str(torch_module.__version__),
    )
    if match is None:
        raise RuntimeError(
            f"{operation} requires a recognizable PyTorch version of at least 2.6"
        )
    version = tuple(int(value) for value in match.groups()[:3])
    is_minimum_prerelease = (
        version == MINIMUM_SAFE_TORCH_VERSION and match.group(4) is not None
    )
    if version < MINIMUM_SAFE_TORCH_VERSION or is_minimum_prerelease:
        raise RuntimeError(
            f"{operation} requires PyTorch 2.6 or newer; found "
            f"{torch_module.__version__}"
        )
