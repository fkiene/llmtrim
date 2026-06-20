# bench/scripts

The benchmark, packaged as `benchkit` with one CLI entry. Run it through the
[`Makefile`](../Makefile) (`make -C crates/llmtrim-cli/bench help`) where possible. Method and
limitations are in [`../BENCH_SPEC.md`](../BENCH_SPEC.md).

## Single entry

```
python3 scripts/bench.py <competitor> [flags]
```

`bench.py` is a thin wrapper onto `benchkit.cli:main`. The first positional argument is the
competitor to benchmark llmtrim against. Flags: `--limit`, `--repeats`, `--live`, `--live-n`,
`--seeds`, `--max-out`, `--budget`, `--check`, `--write-baseline`, `--no-ml`.

```
python3 scripts/bench.py headroom --check --limit 5     # CI gate ($0)
python3 scripts/bench.py headroom --limit 40            # deterministic sweep ($0)
OPENROUTER_API_KEY=... python3 scripts/bench.py headroom --live --budget 0.90   # + CPCA
python3 scripts/bench.py leanctx --limit 8              # on-grid local-ML sweep ($0)
python3 scripts/bench.py entroly --limit 8             # on-grid deterministic sweep ($0)
python3 scripts/bench.py caveman                        # self-contained system-prompt A/B
python3 scripts/bench.py rtk                            # self-contained tool-output filters ($0)
python3 scripts/bench.py snip                           # self-contained tool-output filters ($0)
```

## Package layout

```
benchkit/
  cli.py            single argparse entry; resolves the competitor and runs the engine
  lib.py            primitives: o200k_base encoder, the llmtrim driver, scorers, OpenRouter client
  config.py         generic constants: corpora, scorers, llmtrim presets, match points, paths
  corpora.py        load_corpus + the choice / rouge scorers on top of lib.score()
  stats.py          bootstrap CI of reduction + paired-bootstrap quality diff
  pricing.py        load_pricing + the USD helper
  sweep.py          deterministic ($0) token + latency sweep, generic over a Competitor
  live.py           budget-capped live CPCA leg, generic over a Competitor
  gate.py           --check / --write-baseline / data integrity / provenance
  report.py         render the snapshot README from results (competitor.notes() supplies caveats)
  competitors/
    base.py         the Competitor interface
    __init__.py     the registry + get(name)
    headroom.py     Headroom adapter (on-grid: config grid, compress, ml_fired, disable_ml, notes)
    leanctx.py      leanctx / LLMLingua-2 adapter (on-grid, local ML, $0)
    entroly.py      entroly adapter (on-grid, deterministic local path, $0)
    caveman.py      caveman adapter (self-contained; see below)
    rtk.py          RTK tool-output adapter (self-contained; rtk_fixtures/)
    snip.py         snip tool-output adapter (self-contained; snip_fixtures/)
  data/             download.py, fetch_pricing.py, synth_toolout.py
  tools/            chart.py, synth_readme.py
```

## The engine is generic over a Competitor

The sweep, the live leg, and the report never name a specific tool. They iterate the
competitor's `config_grid()`, call its `compress()`, ask `ml_fired()`, and pull
competitor-specific caveats from `notes()`. The llmtrim side and all numeric logic (scorers,
stats, CPCA, significance) are shared.

### Adding a competitor

1. Add `benchkit/competitors/<name>.py` with a `Competitor` subclass: set `name` + `display`,
   implement `compress()`, `config_grid()`, `ml_fired()`, `notes()`, and `disable_ml()` (a
   no-op when there is no ML path). Decorate it with `@register`.
2. Import it in `benchkit/competitors/__init__.py` so the registry sees it.
3. Run `python3 scripts/bench.py <name> --limit 5`.

That is the **on-grid** shape: `headroom`, `leanctx`, and `entroly` all use
it. They compress the same message arrays over the same prose corpora, so they ride the shared
engine.

If a competitor's comparison shape does not fit the corpora x grid model, keep it
**self-contained**: give it a `run(argv)` it owns and list its name in `cli.SELF_CONTAINED`, so
the CLI dispatches to `run()` instead of the engine. `caveman`, `rtk`, and `snip` use this
shape. `caveman` compares system-prompt strategies on output tokens (no corpora, no config
grid). `rtk` and `snip` rewrite tool OUTPUT (shell-command text), not message arrays, so they
run on their own real-output fixtures (`rtk_fixtures/`, `snip_fixtures/`) and write a runtime
`snapshots/vs-rtk` / `vs-snip` folder that is not committed.

## Data and pricing

| module | role | run |
|---|---|---|
| `benchkit.data.download` | Fetch + normalize the public corpora into `../data/*.jsonl`, sha-pinned in `../data/manifest.json`. | `make data` |
| `benchkit.data.fetch_pricing` | Refresh `../pricing.json` (per-model input/output/cache-read rates). | `PYTHONPATH=scripts python3 -m benchkit.data.fetch_pricing` |
| `benchkit.data.synth_toolout` | Generate the self-authored synthetic tool-output corpus. Excluded from the headline benchmark; kept for ad-hoc analysis. | `PYTHONPATH=scripts python3 -m benchkit.data.synth_toolout` |

## Reporting tools

| module | role | run |
|---|---|---|
| `benchkit.tools.chart` | Render the frontier SVGs in `../` (`frontier-{light,dark}.svg`). | `PYTHONPATH=scripts python3 -m benchkit.tools.chart` |
| `benchkit.tools.synth_readme` | Regenerate `../README.md` from the snapshot data. | `PYTHONPATH=scripts python3 -m benchkit.tools.synth_readme` |

## Dependencies

`requirements-vs-headroom.txt` pins the shared Python deps (`headroom-ai`, `tiktoken`,
`rouge-score`, plus `torch`/`transformers` for the ML competitors). `make deps` installs them.
Each on-grid competitor adds its own pin on top: `requirements-vs-leanctx.txt`,
`requirements-vs-entroly.txt`. `rtk` and `snip` shell
out to their own CLIs (no extra Python deps). The llmtrim wheel is built and installed
separately by `make install`.
