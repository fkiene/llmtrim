"""leanctx adapter (the pinned PyPI `leanctx`).

Wraps leanctx's headline compressor, `Lingua`, behind the Competitor interface. Lingua is
leanctx's local extractive path: it wraps Microsoft's LLMLingua-2, a BERT-style token
classifier that keeps the highest-scoring tokens and drops the rest. The surviving wording is
a verbatim subset of the input, the model runs locally, and no request leaves the machine, so
the leg stays $0. The library's other compressor, SelfLLM, summarizes by calling a hosted LLM
(needs an API key, costs money) and is deliberately not wired here.

Lingua already returns messages in the same `{role, content}` shape it received and preserves
`tool_use` / image / thinking blocks verbatim, so the engine's o200k span over the returned
messages measures it on exactly the same basis as every other competitor. No remapping needed.

leanctx is imported lazily inside the client constructor: importing this module (e.g. when the
registry is populated) never requires leanctx or its ~1.2 GB model to be installed. A missing
install degrades to `installed = False`, which the sweep skips with a printed note.

Validated against leanctx 0.3.1 / llmlingua 0.2.2, where `ratio` is the keep-fraction passed
straight to LLMLingua-2's `rate`. Recheck that mapping before bumping either pin: a future
release could redefine `ratio` and silently shift every arm's reduction.
"""
import statistics
import sys
import time

from . import register
from .base import Competitor

# Lingua's `ratio` is the fraction of tokens to KEEP (same meaning as LLMLingua-2's `rate`):
# 1.0 keeps everything, lower is more aggressive. The grid spans the near-no-op anchor
# (keep 1.0) to aggressive (keep 0.20) so the Pareto curve is a real arc, not two dots.
# Labels are stable arm names used in the report.
LEANCTX_GRID = [
    ("lc-keep1.00", dict(ratio=1.00)),
    ("lc-keep0.75", dict(ratio=0.75)),
    ("lc-keep0.50", dict(ratio=0.50)),
    ("lc-keep0.33", dict(ratio=0.33)),
    ("lc-keep0.20", dict(ratio=0.20)),
]

# LLMLingua-2 picks its own device; pin CPU for deterministic, reproducible numbers across
# machines with and without a GPU (CUDA vs CPU paths can diverge slightly).
DEVICE = "cpu"


@register
class LeanctxCompetitor(Competitor):
    name = "leanctx"
    display = "leanctx"

    def __init__(self):
        self._ml_disabled = False
        self._compressors = {}
        self._available = self._probe()

    # ── availability + ML toggle ───────────────────────────────────────────────
    def _probe(self):
        """True when leanctx and its LLMLingua-2 backend can both load. Import lazily so the
        registry import never pulls leanctx in."""
        try:
            from leanctx import Lingua  # noqa: F401
        except Exception as e:  # noqa: BLE001
            print(f"leanctx not importable: {e}", file=sys.stderr)
            return False
        try:
            import llmlingua  # noqa: F401
        except Exception as e:  # noqa: BLE001
            print("leanctx: Lingua backend (llmlingua) NOT available "
                  f"(install 'leanctx[lingua]'): {e}", file=sys.stderr)
            return False
        return True

    @property
    def installed(self):
        return self._available

    def disable_ml(self):
        """Lingua has no deterministic non-ML fallback: LLMLingua-2 IS the compressor. With
        --no-ml there is nothing left to run, so flag the process to pass content through
        unchanged. This keeps --no-ml honest (a 0% arm) rather than silently keeping the ML
        path on."""
        self._ml_disabled = True
        print("leanctx ML disabled: Lingua is ML-only, so it passes through unchanged",
              file=sys.stderr)

    def _compressor(self, cfg):
        """One cached Lingua per ratio. The first call per ratio loads the model; later calls
        reuse it so the timing reflects steady-state compress(), not the one-time model load."""
        from leanctx import Lingua

        ratio = cfg["ratio"]
        if ratio not in self._compressors:
            self._compressors[ratio] = Lingua(ratio=ratio, device=DEVICE)
        return self._compressors[ratio]

    # ── Competitor interface ────────────────────────────────────────────────────
    def config_grid(self):
        return LEANCTX_GRID

    def compress(self, messages, cfg, repeats):
        if self._ml_disabled:
            # No ML path means no compression; return the span untouched, timed at ~0.
            return messages, ["passthrough:ml_disabled"], 0.0
        compressor = self._compressor(cfg)
        durations = []
        out_messages = messages
        for _ in range(repeats):
            t = time.perf_counter()
            out_messages, _stats = compressor.compress(messages)
            durations.append((time.perf_counter() - t) * 1000)
        return out_messages, ["lingua:llmlingua-2"], statistics.median(durations)

    def ml_fired(self, transforms):
        return any(t.startswith("lingua:") for t in transforms)

    def notes(self):
        return {
            "noml": ("leanctx no-ML is 0%: its only compressor here, Lingua, is LLMLingua-2, "
                     "an ML token classifier with no deterministic fallback. Strip the model "
                     "and there is nothing left to compress, so --no-ml is a passthrough."),
            "ml_cap": ("Lingua's reduction is set by its keep-ratio (`ratio`), so its arms land "
                       "where you ask them to. Read its quality next to that ratio: it keeps a "
                       "verbatim token subset, so structure and exact wording survive but "
                       "dropped tokens are gone."),
            "scope": ("Scope: leanctx's other compressor, SelfLLM, summarizes by calling a "
                      "hosted LLM (API key, real cost) and is out of scope for this $0 "
                      "library-vs-library run. Only the local Lingua path is measured."),
        }
