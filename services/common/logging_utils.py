import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional


def make_next_indexed_log_filename(
    log_dir: str,
    prefix: str,
    extension: str,
) -> str:
    """
    Returns the next ``{prefix}_{N}{extension}`` name in ``log_dir``.

    Scans existing files matching the pattern and picks ``max(N) + 1``.
    Creates ``log_dir`` if missing.

    Args:
        log_dir: Directory to search and create.
        prefix: Filename prefix before the numeric index.
        extension: File suffix (with or without a leading dot).

    Returns:
        New filename.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ext = extension if extension.startswith(".") else f".{extension}"
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(ext)}$")

    max_index = 0
    for file_name in os.listdir(log_dir):
        match = pattern.match(file_name)
        if match:
            max_index = max(max_index, int(match.group(1)))

    next_index = max_index + 1
    return f"{prefix}_{next_index}{ext}"


def log_data(
    data: Mapping[str, Any],
    log_dir: str,
    *,
    prefix: str = "log",
    extension: str = ".txt",
    log_filename: Optional[str] = None,
    separator: str = "=",
) -> str:
    """
    Writes key-value lines to an indexed file under ``log_dir``.

    Each line has the form ``{key}{separator}{value}\\n``.

    Args:
        data: Mapping of metric names to scalar or string values.
        log_dir: Output directory (created if needed).
        prefix: Used when auto-generating the filename.
        extension: File extension for auto-generated names.
        log_filename: Explicit basename; if None, use the next indexed name.
        separator: Delimiter between key and value on each line.

    Returns:
        Absolute path to the written file.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fname = log_filename or make_next_indexed_log_filename(
        log_dir=log_dir,
        prefix=prefix,
        extension=extension,
    )
    out_path = os.path.join(log_dir, fname)

    with open(out_path, "w", encoding="utf-8") as f:
        for key, value in data.items():
            f.write(f"{key}{separator}{value}\n")

    return out_path
