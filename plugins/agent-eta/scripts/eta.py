#!/usr/bin/env python3
"""Agent ETA entry point (hooks, status line, CLI). Standard library only, Python >= 3.8."""
import time as _time

T0 = _time.time()  # event time: taken before imports so async hook latency does not skew it

import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_eta.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], T0))
