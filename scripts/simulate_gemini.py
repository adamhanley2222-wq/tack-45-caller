#!/usr/bin/env python3
"""Simulate one Gemini cascade call locally.

No Twilio call is placed and no Deepgram or Gemini request is made.
The run checks barge-in, sentence overlap, immediate opt-out, and a 429
fallback, then prints what happened.

    python scripts/simulate_gemini.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gemini_sim import run_scenarios


def main() -> int:
    try:
        summary = asyncio.run(run_scenarios())
    except Exception as exc:
        print(f"simulation failed: {exc}")
        return 1
    print("Gemini cascade simulation (no network, no dial)")
    for line in summary["lines"]:
        print(f"  {line}")
    print("all scenarios passed" if summary["ok"] else "scenarios failed")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
