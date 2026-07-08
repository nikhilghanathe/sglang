"""Opt-in CUDA-event phase timing for speculative decoding (offline benchmarking).

Gated by ``envs.SGLANG_DEBUG_SPEC_PHASE_TIMING``. When on, ``record(phase)`` wraps a
spec-decode phase with a CUDA-event pair and accumulates per-phase total GPU time +
step counts. The Engine runs the spec worker in a subprocess, so the totals are dumped
to ``envs.SGLANG_DEBUG_SPEC_PHASE_TIMING_OUT`` (a json path) at process exit — that file
is how timing crosses back to the benchmark process (sd_bench_sglang_offline.py reads it).

Zero overhead when the flag is off: ``record`` yields immediately and allocates nothing.
Only rank 0 writes the file (TP ranks are synchronized, so rank 0 is representative).

Phase names follow the spec-naming convention (verb form, no ``-ed``):
``draft`` | ``target_forward`` | ``tree_build`` | ``accept``.
"""

from __future__ import annotations

import atexit
import json
from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch

from sglang.srt.environ import envs

# phase -> list of (start_event, end_event) not yet folded into totals
_PENDING: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
# phase -> [total_ms, step_ct]
_TOTALS: Dict[str, List[float]] = {}
_FLUSH_EVERY = 32  # fold (sync once) + write the file after this many records per phase
_ENABLED = None
_REGISTERED = False


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def enabled() -> bool:
    global _ENABLED, _REGISTERED
    if _ENABLED is None:
        _ENABLED = bool(
            envs.SGLANG_DEBUG_SPEC_PHASE_TIMING.get()
        ) and torch.cuda.is_available()
        if _ENABLED and not _REGISTERED:
            atexit.register(dump)
            _REGISTERED = True
    return _ENABLED


@contextmanager
def record(phase: str):
    """Time a phase with a CUDA-event pair. No-op (and no allocation) when disabled."""
    if not enabled():
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        pending = _PENDING.setdefault(phase, [])
        pending.append((start, end))
        if len(pending) >= _FLUSH_EVERY:
            _fold(phase)
            _write()  # write incrementally: worker is killed by signal, atexit may not fire


def _fold(phase: str) -> None:
    pending = _PENDING.get(phase)
    if not pending:
        return
    torch.cuda.synchronize()  # required before Event.elapsed_time
    total, ct = _TOTALS.get(phase, [0.0, 0])
    for s, e in pending:
        total += s.elapsed_time(e)
        ct += 1
    _TOTALS[phase] = [total, ct]
    pending.clear()


def _write() -> None:
    """Write the current per-phase {total_ms, ct} totals to the json sink (rank 0)."""
    if _rank() != 0:
        return
    out = envs.SGLANG_DEBUG_SPEC_PHASE_TIMING_OUT.get()
    if not out:
        return
    data = {ph: {"total_ms": t, "ct": c} for ph, (t, c) in _TOTALS.items()}
    try:
        with open(out, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def dump() -> None:
    """Fold any pending events and write per-phase {total_ms, ct} to the json sink.
    Best-effort atexit flush; the incremental _write() in record() is the reliable path
    (the worker subprocess is usually killed by signal, so atexit may not run)."""
    if not _ENABLED:
        return
    for phase in list(_PENDING):
        _fold(phase)
    _write()
