"""GPU-specific tests for Jasc."""

# XXX: Remove paths to `jax*` packages installed from pip by Bazel rules.
import sys
sys.path = [p for p in sys.path if "/pip_deps_jax" not in p]

import chex
import pytest


# ===----------------------------------------------------------------------=== #
# Import Jasc CPU tests.
# ===----------------------------------------------------------------------=== #

# XXX: This "imports" the CPU tests but does not run them on a GPU. How to
#      achieve that? Maybe we need to rethink the whole GPU vs. CPU mechanism...
"""Imports common tests."""
from jasc.test.cpu_integration import *


# ===----------------------------------------------------------------------=== #
# Jasc GPU-specific tests.
# ===----------------------------------------------------------------------=== #
def test_running_on_gpu():
  chex.assert_gpu_available()


if __name__ == "__main__":
    args = sys.argv[1:] or ["-s", "-v"]
    sys.exit(pytest.main([__file__] + args))
