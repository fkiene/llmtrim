"""benchkit: the pluggable-competitor benchmark for llmtrim.

The engine (sweep + live + report) is generic over a `Competitor` (see
`benchkit.competitors.base`). Each competitor is a small adapter that knows how to
compress messages and how to span its aggressiveness grid; the numbers, scorers, and
report layout live in the engine, not the adapter. The CLI entry is `benchkit.cli:main`,
dispatched from the single `scripts/bench.py` wrapper as `bench.py <competitor> [flags]`.
"""
