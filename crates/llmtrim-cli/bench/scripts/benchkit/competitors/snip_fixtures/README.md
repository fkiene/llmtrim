# snip fixtures: real tool output

These files are real command output, taken verbatim from snip's own repo
(`github.com/edouard-claude/snip`, `tests/fixtures/*_raw.txt`, commit
`fab371a`). They are checked in so the benchmark is reproducible without
re-running the commands or cloning snip.

Each file maps to the snip filter of the same name and the command argv in
`snip.py`'s `CASES`:

- `cargo-test.txt`     <- `cargo test`        (cargo_test_raw.txt)
- `rspec.txt`          <- `rspec`             (rspec_raw.txt)
- `rails-routes.txt`   <- `rails routes`      (rails_routes_raw.txt)
- `rails-migrate.txt`  <- `rails db:migrate`  (rails_migrate_raw.txt)
- `bundle-install.txt` <- `bundle install`    (bundle_install_raw.txt)
- `ls.txt`             <- `ls -la -R`         (ls_long_recursive_raw.txt)

Only filters that do NOT inject args are used. For a non-injecting filter snip
runs the command and applies its pipeline to the raw stdout, so feeding the
captured raw text reproduces snip's live behavior exactly. The benchmark drives
the real snip binary via `snip run -- <argv>` with a one-line shim that emits
the fixture, then counts the o200k tokens before and after.

snip's injecting filters (git-log, git-diff, git-status, go-test) are left out:
snip re-runs those commands with extra flags (`--pretty=format:...`, `-json`,
...), so the pipeline sees a different stream than a static raw fixture and the
number would be wrong. snip's own integration test makes the same split.

To refresh: clone snip, copy the listed `tests/fixtures/*_raw.txt` to the names
above, and rerun `bench.py snip`.
