"""Headroom adapter (the pinned PyPI `headroom-ai`).

Wraps Headroom's `compress(messages, ...)` behind the Competitor interface. Headroom is
imported from the pinned PyPI build unless HEADROOM_SRC is explicitly set (a dev checkout can
be mid-upgrade and internally inconsistent, which breaks the JSON path and makes the numbers
irreproducible).
"""
import os
import statistics
import sys
import time
from pathlib import Path

from . import register
from .base import Competitor

# Only use a local Headroom checkout when the user EXPLICITLY opts in via HEADROOM_SRC.
HEADROOM_SRC = Path(os.environ["HEADROOM_SRC"]) if os.environ.get("HEADROOM_SRC") else None

# Headroom swept across the axes that drive its aggressiveness. Keep the grid small but
# spanning no-op→max so the Pareto curve is real, not two dots.
HEADROOM_GRID = [
    ("hr-default", dict(compress_user_messages=False, compress_system_messages=True,
                        protect_recent=4, target_ratio=None, min_tokens_to_compress=250)),
    ("hr-0.6", dict(compress_user_messages=True, compress_system_messages=True,
                    protect_recent=2, target_ratio=0.6, min_tokens_to_compress=100)),
    ("hr-0.4", dict(compress_user_messages=True, compress_system_messages=True,
                    protect_recent=0, target_ratio=0.4, min_tokens_to_compress=50)),
    ("hr-max", dict(compress_user_messages=True, compress_system_messages=True,
                    protect_recent=0, target_ratio=0.2, min_tokens_to_compress=50)),
]

# The `model` field handed to Headroom's local compress() so it picks a tokenizer. NOT an
# API call. Kept in sync with lib.BODY_MODEL.
BODY_MODEL = "gpt-4o"


@register
class HeadroomCompetitor(Competitor):
    name = "headroom"
    display = "Headroom"

    def __init__(self):
        self._client = self._make_client()

    # ── client + ML toggle ────────────────────────────────────────────────────
    def _make_client(self):
        if HEADROOM_SRC and HEADROOM_SRC.exists() and str(HEADROOM_SRC) not in sys.path:
            print(f"headroom: using explicit checkout {HEADROOM_SRC} (not the pinned PyPI build)",
                  file=sys.stderr)
            sys.path.insert(0, str(HEADROOM_SRC))
        try:
            from headroom import compress
            from headroom.transforms.kompress_compressor import is_kompress_available
        except Exception as e:  # noqa: BLE001
            print(f"headroom not importable: {e}", file=sys.stderr)
            return None
        if not is_kompress_available():
            print("headroom: Kompress ML path NOT available (install headroom-ai[ml]); "
                  "running its deterministic path only", file=sys.stderr)
        return compress

    @property
    def installed(self):
        return self._client is not None

    def disable_ml(self):
        """Disable Headroom's ML (Kompress/ModernBERT) for the whole process BEFORE the
        pipeline is built, so the deterministic routers run alone. Must run before the client
        is constructed - Headroom builds its transform pipeline once and caches the
        availability result, so a per-call toggle doesn't work."""
        from headroom.transforms import kompress_compressor as _kc
        for fn in ("is_kompress_available", "_is_onnx_available", "_is_pytorch_available"):
            if hasattr(_kc, fn):
                setattr(_kc, fn, lambda *a, **k: False)
        print("Headroom ML disabled (deterministic routers only)", file=sys.stderr)

    # ── Competitor interface ───────────────────────────────────────────────────
    def config_grid(self):
        return HEADROOM_GRID

    def compress(self, messages, cfg, repeats):
        durations = []
        res = None
        for _ in range(repeats):
            t = time.perf_counter()
            res = self._client(messages, model=BODY_MODEL, **cfg)
            durations.append((time.perf_counter() - t) * 1000)
        return res.messages, list(res.transforms_applied), statistics.median(durations)

    def ml_fired(self, transforms):
        return any(("router:text" in t) or ("kompress" in t.lower()) for t in transforms)

    def notes(self):
        return {
            "noml": ("Headroom no-ML is 0% here because the corpora are prose; its "
                     "deterministic routers (JSON/code/log) have nothing to bite on. On "
                     "JSON/code/log inputs no-ML would compress - that path is just out of "
                     "scope for these text corpora."),
            "ml_cap": ("At the aggressive point the match is near-iso, not exact: Headroom's "
                       "ML caps its reduction (~24% here) so llmtrim's more-aggressive arm is "
                       "a few pp ahead on compression - read its quality/cost next to that gap."),
            "rtk": ("RTK scope: Headroom's bundled RTK shell-output rewriter is active only in "
                    "its `wrap`/proxy mode, not in `headroom.compress`; it is out of scope for "
                    "this library-vs-library comparison."),
        }
