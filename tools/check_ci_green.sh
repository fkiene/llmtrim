#!/bin/sh
# cargo-release pre-release-hook: refuse to cut a release unless the commit being
# released is pushed and every CI check on it is green. Needs `gh` (authenticated).
set -eu

HEAD=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "")

if [ "$HEAD" != "$REMOTE" ]; then
    echo "release blocked: HEAD ($HEAD) is not origin/main ($REMOTE) — push first so CI can run" >&2
    exit 1
fi

# All check runs for HEAD must exist and be completed+successful (neutral/skipped ok).
# The "Dependabot" check-run is excluded: it reports the result of Dependabot's scheduled
# dependency-update job, not a CI check on the release commit's code. It goes red on a
# `security_update_not_possible` (an advisory whose fixed version isn't resolvable in our
# tree yet), which is a standing dependency-tree fact, not a reason to block a release.
# The exclusion is by name, so it drops that check-run for any conclusion, not just the
# advisory case — intentional: the scheduled dependency-update job is never a CI check on
# the release commit's code, so it should not gate a release whatever its result.
NOT_DEPENDABOT='.name != "Dependabot"'
PENDING=$(gh api "repos/{owner}/{repo}/commits/$HEAD/check-runs" \
    --jq "[.check_runs[] | select($NOT_DEPENDABOT and .status != \"completed\")] | length")
RED=$(gh api "repos/{owner}/{repo}/commits/$HEAD/check-runs" \
    --jq "[.check_runs[] | select($NOT_DEPENDABOT and .status == \"completed\" and (.conclusion != \"success\" and .conclusion != \"neutral\" and .conclusion != \"skipped\"))] | .[].name")
TOTAL=$(gh api "repos/{owner}/{repo}/commits/$HEAD/check-runs" \
    --jq "[.check_runs[] | select($NOT_DEPENDABOT)] | length")

if [ "$TOTAL" = "0" ]; then
    echo "release blocked: no CI checks found for $HEAD (still queued? docs-only push skips CI — make sure the release commit's parent ran; the Dependabot check-run is excluded from this count)" >&2
    exit 1
fi
if [ "$PENDING" != "0" ]; then
    echo "release blocked: $PENDING CI check(s) still running for $HEAD — wait for green" >&2
    exit 1
fi
if [ -n "$RED" ]; then
    echo "release blocked: failing checks on $HEAD:" >&2
    echo "$RED" >&2
    exit 1
fi

echo "CI green on $HEAD ($TOTAL checks) — proceeding"
