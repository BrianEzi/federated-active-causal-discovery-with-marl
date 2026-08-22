"""Test-session setup.

Two things happen here: the repository root goes on `sys.path`, and `build_graph_space` is
memoised for the duration of the session. See below for why the second is safe.
"""
import functools
import os
import sys

# Ensure repository root is on sys.path for test discovery across all platforms
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sa.graphs as _graphs  # noqa: E402  (must follow the sys.path insertion)


# --------------------------------------------------------------------------------------
# Memoise graph-space construction, for the test session only
# --------------------------------------------------------------------------------------
#
# `build_graph_space(d)` enumerates every DAG on d nodes and groups them into equivalence
# classes. It is a PURE FUNCTION OF `(d, fast)` -- same arguments, same answer, every time --
# and it is not cheap: measured 22 August, a single d=6 build costs ~33 s. The suite called
# it from roughly 55 sites across 15 files, rebuilding the same handful of spaces over and
# over, and that was the single largest cost in an 876 s run.
#
# Safe to cache because `GraphSpace` is a frozen dataclass and no test assigns into its
# arrays (checked, 22 August). The arrays themselves are ordinary mutable numpy arrays, so
# this is a convention rather than a guarantee: **a test that mutates a returned space would
# now corrupt every later test that asks for the same `d`.** If that is ever needed, copy
# explicitly rather than removing the cache.
#
# Deliberately NOT cached in `sa/graphs.py` itself. Production jobs build large spaces and an
# unbounded process-lifetime cache there is a memory leak waiting to happen at d >= 8; the
# test session builds a few small ones and exits. The cost belongs where the benefit is.
#
# This must run at conftest IMPORT time, not in a fixture: test modules do
# `from sa.graphs import build_graph_space`, which binds the function object at their own
# import time. conftest is imported first, so patching the module attribute here is picked up
# by those later imports. A fixture would run too late to matter.

_uncached_build_graph_space = _graphs.build_graph_space


@functools.lru_cache(maxsize=None)
def _cached_build_graph_space(d: int, fast: bool = True):
    return _uncached_build_graph_space(d, fast=fast)


_graphs.build_graph_space = _cached_build_graph_space
