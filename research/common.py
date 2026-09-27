"""Per-model turn statistics across all imported histories (each home = one user or one dataset)."""
import math
import os
import sqlite3
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E  # noqa: E402

import json

# Histories to analyse: research/histories.json (or $HISTORIES), see histories.example.json.
_CONFIG = os.environ.get("HISTORIES") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".", "histories.json")
_H = json.load(open(_CONFIG))
HOMES = {name: (h["home"], h.get("kind", "")) for name, h in _H.items()}  # label: (home, kind)


def short(m):
    m = (m or "?").replace("moonshotai/", "").replace("minimax/", "")
    for cut in ("-2026", "-2025"):
        m = m.split(cut)[0]
    return m.replace("claude-", "")

