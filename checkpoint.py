"""
checkpoint.py
==============
Atomic checkpointing utility shared by all run_*.py scripts.

Design:
- One checkpoint file per experiment, e.g. ``experiments/case1.ckpt.pkl``
- Each (epsilon, target, run) result is added to a dict and saved IMMEDIATELY
- Atomic write via tempfile+rename so crashes never produce a corrupt file
- ``load_or_init`` returns the existing checkpoint if any, else an empty one
- ``mark_done(...)`` and ``is_done(...)`` are the only operations needed
"""
from __future__ import annotations
import os, pickle, tempfile, json, time
from typing import Any


def _ckpt_path(name: str) -> str:
    os.makedirs("experiments", exist_ok=True)
    return os.path.join("experiments", f"{name}.ckpt.pkl")


def load_or_init(name: str, restart: bool = False) -> dict:
    """
    Returns a checkpoint dict for the experiment ``name``.

    Schema:
        {
            "name":      str,                  # the experiment name
            "started":   float (unix time),
            "tasks":     dict[str, Any],       # key -> result payload
            "logs":      list[str],            # human-readable log lines
        }
    """
    path = _ckpt_path(name)
    if not restart and os.path.exists(path):
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        ckpt.setdefault("logs", []).append(
            f"[resumed at {time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"already done: {len(ckpt.get('tasks', {}))}"
        )
        return ckpt
    return dict(name=name, started=time.time(), tasks={}, logs=[])


def save(ckpt: dict) -> None:
    """Atomic save: write to a temp file in the same dir, then rename."""
    path = _ckpt_path(ckpt["name"])
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".ckpt-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(ckpt, f)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def is_done(ckpt: dict, key: str) -> bool:
    return key in ckpt.get("tasks", {})


def mark_done(ckpt: dict, key: str, value: Any, log_msg: str | None = None) -> None:
    """Record ``value`` under ``key`` and immediately persist to disk."""
    ckpt.setdefault("tasks", {})[key] = value
    if log_msg:
        ckpt.setdefault("logs", []).append(
            f"[{time.strftime('%H:%M:%S')}] {log_msg}"
        )
    save(ckpt)


def get(ckpt: dict, key: str, default: Any = None) -> Any:
    return ckpt.get("tasks", {}).get(key, default)


def summary(ckpt: dict) -> str:
    n = len(ckpt.get("tasks", {}))
    started = time.strftime("%H:%M:%S", time.localtime(ckpt.get("started", 0)))
    return f"checkpoint '{ckpt['name']}' started {started}, {n} tasks done"
