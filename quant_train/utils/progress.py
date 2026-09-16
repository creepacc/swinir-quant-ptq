from __future__ import annotations

import sys
from typing import Iterable, Optional


def progress(it: Iterable, desc: str = "", total: Optional[int] = None):
    """tqdm wrapper; falls back to the raw iterable if tqdm is missing."""
    try:
        from tqdm import tqdm
    except ImportError:
        return it
    return tqdm(
        it,
        desc=desc,
        total=total,
        dynamic_ncols=True,
        leave=True,
        mininterval=0.3,
        file=sys.stderr,
    )
