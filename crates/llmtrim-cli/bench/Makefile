# llmtrim vs Headroom benchmark runner. See BENCH_SPEC.md.
#
# Quick start (from anywhere):
#   make -C crates/llmtrim-cli/bench setup    # build+install the wheel, pinned deps, corpora
#   make -C crates/llmtrim-cli/bench bench     # deterministic token + latency sweep ($0)
#
# The deterministic targets cost $0 (no API); only `bench LIVE=1` spends, capped by BUDGET.
# Headroom is imported from the pinned PyPI build (requirements-vs-headroom.txt); set
# HEADROOM_SRC=<path> to test a local checkout instead.

PY       ?= python3
REPO     := ../../..
SCR      := scripts/bench.py
COMPET   ?= headroom            # which competitor to benchmark against (bench.py <competitor>)
MANIFEST := data/manifest.json
DATA_N   ?= 100
LIMIT    ?= 40
LIVE_N   ?= 15
SEEDS    ?= 3
MAX_OUT  ?= 400
BUDGET   ?= 0.90
ARGS     ?=
NOML     ?=                       # NOML=1  -> disable Headroom's ML (deterministic routers only)
LIVE     ?=                       # LIVE=1  -> also run the live CPCA leg (needs OPENROUTER_API_KEY)
_NOML    := $(if $(NOML),--no-ml,)
_LIVE    := $(if $(LIVE),--live --live-n $(LIVE_N) --seeds $(SEEDS) --max-out $(MAX_OUT) --budget $(BUDGET),)
# Blank the proxy ONLY for the live leg: the A/B must reach the model DIRECTLY, since routing
# it through the llmtrim proxy would re-compress every arm and void the comparison.
_PROXY   := $(if $(LIVE),env HTTPS_PROXY= HTTP_PROXY=,)

.PHONY: help setup install deps data bench check baseline

help:
	@echo "make setup              - build+install the llmtrim wheel, pinned deps, and corpora"
	@echo "make bench              - deterministic token + latency sweep (\$$0)   [LIMIT=$(LIMIT)]"
	@echo "make bench NOML=1       - same, with Headroom's ML disabled"
	@echo "make bench LIVE=1       - + live CPCA leg (needs OPENROUTER_API_KEY)  [BUDGET=$(BUDGET)]"
	@echo "make check              - CI gate: invariants + reduction vs baseline.json (\$$0)"
	@echo "make baseline           - (re)generate baseline.json (do this intentionally)"
	@echo "make install / deps / data  - wheel only / Python deps only / corpora only"
	@echo "extra flags via ARGS=, e.g.  make bench ARGS=\"--repeats 3\""

# One-shot environment: the llmtrim wheel, the pinned Python deps, and the corpora.
setup: install deps data

# Build + install the llmtrim wheel (the library under test).
install:
	$(REPO)/crates/llmtrim-uniffi/scripts/build-wheel.sh --release
	$(PY) -m pip install --user $(REPO)/target/wheels/llmtrim-*.whl

deps:
	$(PY) -m pip install -r scripts/requirements-vs-headroom.txt

# Force a fresh download (re-pins data/manifest.json).
data:
	PYTHONPATH=scripts $(PY) -m benchkit.data.download $(DATA_N)

# Download only if the corpora are missing (so `make bench` on a fresh checkout just works).
$(MANIFEST):
	PYTHONPATH=scripts $(PY) -m benchkit.data.download $(DATA_N)

# Deterministic Pareto + latency axes ($0). NOML=1 disables the competitor's ML; LIVE=1 adds
# the budget-capped cost-per-answer-quality leg. Auto-fetches the corpora if absent.
bench: $(MANIFEST)
	$(_PROXY) $(PY) $(SCR) $(COMPET) --limit $(LIMIT) $(_NOML) $(_LIVE) $(ARGS)

# The CI gate. Runs at the baseline's limit; exits non-zero on a regression.
check: $(MANIFEST)
	$(PY) $(SCR) $(COMPET) --check --limit 5

baseline: $(MANIFEST)
	$(PY) $(SCR) $(COMPET) --write-baseline --limit 5
