"""Entroly adapter (the PyPI `entroly` package).

Wraps Entroly's `compress_messages(messages, ...)` behind the Competitor interface. That call
is the library's real message-list API, so it is the BEST-mode entrypoint here: the
engine hands it the same message list it hands every other tool, and it returns a compressed
message list scored over the same o200k_base span.

Entroly's compression is local and deterministic. The SDK docstrings state it plainly ("100%
deterministic, zero LLM calls, zero network requests") and a network-blocked run of
compress_messages confirms it: no API key, no HTTP, no provider call. The only optional ML in
the package is an NLI cross-encoder used by its verification layer, which needs
`sentence_transformers`; when that is absent (as in this bench environment) Entroly falls back
to a deterministic local path. compress_messages does not invoke that verifier, so the body
compression measured here is deterministic either way.

The SDK records local-only, fail-open telemetry to a JSON sink under ENTROLY_DIR. It never
reaches the network and never affects compress() output. We still point ENTROLY_DIR at a temp
directory so a bench run leaves no trace in the user's home.
"""
import os
import statistics
import sys
import tempfile
import time

from . import register
from .base import Competitor

# The `model` field handed to Entroly so it picks a provider-aware budget cap. NOT an API call;
# it only looks up a context window locally. Kept in sync with lib.BODY_MODEL.
BODY_MODEL = "gpt-4o"

# Entroly swept from no-op to max aggressiveness. compress_messages reduction is driven by the
# token `budget`, how many recent turns are kept verbatim (`preserve_last_n`), and `distill`
# (strip filler from older assistant turns). The library's `profile` knob (safe/balanced/max)
# does NOT reach compress_messages - that path ignores it and only the standalone compress()
# reads it - so the grid leaves profile out rather than imply a knob that does nothing here.
# Keep the grid small but spanning no-op->max so the Pareto curve is real, not two dots.
ENTROLY_GRID = [
    ("en-default", dict(budget=50000, preserve_last_n=4, distill=False)),
    ("en-0.6", dict(budget=4000, preserve_last_n=2, distill=True)),
    ("en-0.4", dict(budget=2000, preserve_last_n=1, distill=True)),
    ("en-max", dict(budget=800, preserve_last_n=0, distill=True)),
]


@register
class EntrolyCompetitor(Competitor):
    name = "entroly"
    display = "Entroly"

    def __init__(self):
        # Keep telemetry out of the user's home: local-only, fail-open, but still no trace.
        os.environ.setdefault("ENTROLY_DIR", tempfile.mkdtemp(prefix="entroly-bench-"))
        self._client = self._make_client()

    # ── client + ML toggle ────────────────────────────────────────────────────
    def _make_client(self):
        try:
            from entroly import compress_messages
        except Exception as e:  # noqa: BLE001
            print(f"entroly not importable: {e}", file=sys.stderr)
            return None
        try:
            from entroly import nli_available
            if not nli_available():
                print("entroly: NLI verifier ML path NOT available (install "
                      "sentence_transformers); running its deterministic path only",
                      file=sys.stderr)
        except Exception:  # noqa: BLE001 (probe only; absence just means deterministic)
            pass
        return compress_messages

    @property
    def installed(self):
        return self._client is not None

    def disable_ml(self):
        """Force Entroly's optional NLI cross-encoder off for the whole process so only the
        deterministic local path runs. compress_messages does not call the verifier, so this is
        belt-and-suspenders; it keeps a --no-ml run honest if a future Entroly wires the NLI
        model into the compression path."""
        patched = False
        # The NLI cross-encoder lives in entroly.verifiers.local_nli and is re-exported at the
        # top level. Force every entrypoint to report unavailable so the deterministic local
        # path (PAV fallback) runs alone.
        try:
            from entroly.verifiers import local_nli
            for fn in ("is_available", "nli_score", "batch_nli_scores"):
                if hasattr(local_nli, fn):
                    setattr(local_nli, fn, lambda *a, **k: False)
                    patched = True
        except Exception:  # noqa: BLE001
            pass
        try:
            import entroly
            for fn in ("nli_available", "nli_score"):
                if hasattr(entroly, fn):
                    setattr(entroly, fn, lambda *a, **k: False)
                    patched = True
        except Exception:  # noqa: BLE001
            pass
        if patched:
            print("Entroly ML disabled (deterministic path only)", file=sys.stderr)
        else:
            print("entroly: no ML path to disable", file=sys.stderr)

    # ── Competitor interface ───────────────────────────────────────────────────
    def config_grid(self):
        return ENTROLY_GRID

    def compress(self, messages, cfg, repeats):
        if self._client is None:
            raise RuntimeError("entroly not installed; check `installed` before compress()")
        durations = []
        out = None
        for _ in range(repeats):
            t = time.perf_counter()
            out = self._client(messages, model=BODY_MODEL, **cfg)
            durations.append((time.perf_counter() - t) * 1000)
        # Entroly returns a compressed message list directly; there is no per-stage transform
        # log, so transforms records the budget, the one knob the engine can attribute.
        transforms = [f"budget:{cfg.get('budget')}"]
        return out, transforms, statistics.median(durations)

    def ml_fired(self, transforms):
        # Entroly's compress_messages path is deterministic here: the only ML in the package is
        # the optional NLI verifier (sentence_transformers), which it does not load for
        # compression and which is absent in this environment. So the ML path never fires.
        return False

    def notes(self):
        return {
            "noml": ("Entroly's compress_messages is deterministic: its only ML is an optional "
                     "NLI cross-encoder used by the verification layer, not by compression. "
                     "With sentence_transformers absent it falls back to a local path, so "
                     "no-ML and default numbers are identical here."),
            "ml_cap": ("Entroly has no ML compression arm to cap; its reduction is set by the "
                       "budget/preserve/distill knobs in the grid, not by a learned model."),
            "rtk": ("Entroly ships a wider control plane (MCP tools, context receipts, image "
                    "and shell codecs). Only its `compress_messages` SDK call is in scope for "
                    "this library-vs-library comparison."),
        }
