# RTK fixtures: real tool output

These four files are the verbatim output of real commands, captured once so the RTK
benchmark runs on the exact text RTK was built to filter. They are checked in so the
benchmark is reproducible without re-running the commands.

How they were captured (throwaway project: `src/config.py` with a repeated
`timeout_seconds` pattern, `tests/test_mod.py` with 41 passing and 2 failing tests, two
git commits that bump a default):

- `pytest.txt`   = `pytest -q tests/`            (41 passed, 2 failed, with tracebacks)
- `grep.txt`     = `grep -rn timeout_seconds src/`
- `git-log.txt`  = `git --no-pager log --stat`
- `git-diff.txt` = `git --no-pager diff HEAD~1 HEAD`

Each file maps to the RTK `--filter` of the same name in `rtk.py` (`pytest`, `grep`,
`git-log`, `git-diff`). To regenerate, recreate the project and rerun the four commands.
