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
import os
from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch

from sglang.srt.environ import envs

# phase -> list of (start_event, end_event) not yet folded into totals
_PENDING: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
# phase -> [total_ms, step_ct]
_TOTALS: Dict[str, List[float]] = {}
# Fold (sync once) + write the file after this many records per phase.
#
# 32 is right for the four top-level phases, which record ONE pair per phase per
# spec-decode step. It is badly wrong for per-layer sub-phase timers: a 36-layer model
# recording attn/moe/comm per layer produces ~36 pairs per phase per forward, so a
# 32-record flush would call torch.cuda.synchronize() several times *inside* a single
# forward, serialising the GPU and distorting the very thing being measured. Recording an
# event is cheap and async; only _fold() has to synchronise. So when sub-phase timing is
# on, raise this well above (layers x forwards-per-fold) and pay one sync per many
# forwards instead.
_FLUSH_EVERY = int(os.environ.get("SGLANG_DEBUG_SPEC_PHASE_FLUSH_EVERY", "32"))
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


def layer_timing_enabled() -> bool:
    """Separate gate for PER-LAYER sub-phase timers (attn / moe / comm).

    Deliberately NOT the same switch as the four top-level phases, because the two have
    opposite validity conditions:

      * the top-level phases (draft / tree_build / target_forward / accept) are recorded
        at the call site in eagle_worker_v2.py, OUTSIDE any captured region, so they are
        valid with CUDA graphs on -- i.e. in production configuration;
      * the per-layer timers sit INSIDE the model forward. record() no-ops while compiling
        or capturing, so on a graph-on run they would fire only on the minority of
        forwards that fall back to eager (measured: ~8% of them) and quietly emit a biased
        sample under the same key names.

    Sharing one env var would therefore poison every §9 phase JSON with layer_* numbers
    drawn from an unrepresentative subset. Require SGLANG_DEBUG_SPEC_LAYER_TIMING=1
    explicitly, and only use it on a run started with --disable-prefill-cuda-graph.
    """
    return bool(os.environ.get("SGLANG_DEBUG_SPEC_LAYER_TIMING", "")) and enabled()


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


def _capturing() -> bool:
    """True while this region is being traced/captured rather than plainly executed.

    Timing must be skipped then, for two independent reasons:
      * the prefill path is torch.compile'd (prefill_cuda_graph_runner). Dynamo tries to
        trace Event.record(), which is on its skip list, and aborts the whole capture with
        `torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped`;
      * an aborted trace unwinds leaving `start` recorded and `end` not, so a later
        _fold() dies with "Both events must be recorded before calculating elapsed time".

    torch.compiler.is_compiling() covers the dynamo case and is itself traced as a
    constant, so the branch folds away and the compiled graph contains no timing at all.
    is_current_stream_capturing() covers raw CUDA-graph capture. A compiled/replayed
    region cannot be timed with host-recorded events regardless, so skipping loses nothing
    -- it does mean a graph-captured shape reports no sub-phase time, which is why the
    decomposition run disables the prefill graph rather than silently measuring nothing.
    """
    try:
        if torch.compiler.is_compiling():
            return True
    except Exception:
        pass
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


@contextmanager
def record_layer(phase: str):
    """record(), but behind the separate per-layer gate. See layer_timing_enabled()."""
    if not layer_timing_enabled():
        yield
        return
    with record(phase):
        yield


@contextmanager
def record(phase: str):
    """Time a phase with a CUDA-event pair. No-op (and no allocation) when disabled."""
    if not enabled() or _capturing():
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
        # A pair can be half-recorded if the region unwound between start and end (an
        # aborted CUDA-graph capture does exactly this). Dropping it is right: the
        # alternative is ValueError("Both events must be recorded ...") which would take
        # down the atexit dump and lose every other phase's totals too.
        try:
            total += s.elapsed_time(e)
        except Exception:
            continue
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
