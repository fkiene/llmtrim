#!/usr/bin/env bash
# seed-dev-ledger.sh
#
# Seeds a development copy of the llmtrim ledger with realistic breakdown_turns
# rows so `get_dashboard` returns data without needing a running proxy session.
#
# Usage:
#   ./scripts/seed-dev-ledger.sh            # seeds ~/.local/share/llmtrim/tracking.db
#   LLMTRIM_DB_PATH=/tmp/dev.db ./scripts/seed-dev-ledger.sh  # custom path
#
# After seeding, launch the tray app:
#   cargo run -p llmtrim-tray               # (requires macOS or Windows)
#   LLMTRIM_DB_PATH=/tmp/dev.db cargo run -p llmtrim-tray
#
# Manual verification steps:
#   1. Tray icon appears in the menubar (macOS) or system tray (Windows).
#   2. Click the icon — popover window appears near the tray, showing agent cards.
#   3. macOS: no Dock icon visible.
#   4. Click outside the popover — window hides automatically (blur).
#   5. macOS menubar shows "NN% saved" next to the icon.
#   6. Open the browser console (right-click → Inspect) and run:
#        window.__TAURI__.core.invoke('get_dashboard')
#      — should return a Dashboard JSON matching the seeded rows.
#
# Dependencies: sqlite3 (brew install sqlite / apt install sqlite3)
set -euo pipefail

DB_PATH="${LLMTRIM_DB_PATH:-${XDG_DATA_HOME:-$HOME/.local/share}/llmtrim/tracking.db}"
echo "Seeding: $DB_PATH"
mkdir -p "$(dirname "$DB_PATH")"

sqlite3 "$DB_PATH" <<'SQL'
-- Create the schema that the proxy normally migrates (idempotent).
CREATE TABLE IF NOT EXISTS breakdown_turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now')),
    session_id      TEXT NOT NULL,
    agent           TEXT NOT NULL,
    project         TEXT,
    session_name    TEXT,
    provider        TEXT NOT NULL DEFAULT '',
    model           TEXT,
    window          INTEGER NOT NULL DEFAULT 0,
    fresh_input     INTEGER NOT NULL DEFAULT 0,
    cache_read      INTEGER NOT NULL DEFAULT 0,
    cache_write     INTEGER NOT NULL DEFAULT 0,
    output_tok      INTEGER NOT NULL DEFAULT 0,
    input_rate      REAL NOT NULL DEFAULT 0,
    output_rate     REAL NOT NULL DEFAULT 0,
    cache_read_rate REAL NOT NULL DEFAULT 0,
    cache_write_rate REAL NOT NULL DEFAULT 0,
    bill_micros     INTEGER NOT NULL DEFAULT 0,
    input_before    INTEGER,
    input_after     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_breakdown_turns_agent ON breakdown_turns(agent);

-- Seed: claude-code — 10 days of turns, ~60% savings.
INSERT INTO breakdown_turns
    (ts, session_id, agent, provider, model, window, fresh_input,
     cache_read, cache_write, output_tok,
     input_rate, output_rate, cache_read_rate, cache_write_rate,
     bill_micros, input_before, input_after)
SELECT
    strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now', '-' || (10 - i) || ' days'),
    'dev-sess-cc-' || i,
    'claude-code',
    'anthropic',
    'claude-sonnet-4-5',
    200000, 800, 3000, 400, 300,
    3.0, 15.0, 0.3, 3.75,
    12000,
    10000,
    4000
FROM (
    SELECT 0 AS i UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
    UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9
);

-- Seed: codex — 5 days, ~40% savings.
INSERT INTO breakdown_turns
    (ts, session_id, agent, provider, model, window, fresh_input,
     cache_read, cache_write, output_tok,
     input_rate, output_rate, cache_read_rate, cache_write_rate,
     bill_micros, input_before, input_after)
SELECT
    strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now', '-' || (5 - i) || ' days'),
    'dev-sess-codex-' || i,
    'codex',
    'openai',
    'gpt-4o',
    128000, 500, 1000, 200, 200,
    5.0, 15.0, 1.25, 5.0,
    8000,
    6000,
    3600
FROM (
    SELECT 0 AS i UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
);

-- Seed: gemini — 2 turns without input_before/after (pre-meter rows).
INSERT INTO breakdown_turns
    (ts, session_id, agent, provider, model, window, fresh_input,
     cache_read, cache_write, output_tok,
     input_rate, output_rate, cache_read_rate, cache_write_rate,
     bill_micros, input_before, input_after)
VALUES
    (strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now', '-2 days'),
     'dev-sess-gem-1', 'gemini', 'google', 'gemini-1.5-pro',
     1000000, 1000, 500, 100, 400,
     3.5, 10.5, 0.35, 3.5, 5000, NULL, NULL),
    (strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now', '-1 day'),
     'dev-sess-gem-2', 'gemini', 'google', 'gemini-1.5-pro',
     1000000, 1200, 600, 150, 500,
     3.5, 10.5, 0.35, 3.5, 7000, NULL, NULL);

SELECT 'Seeded ' || COUNT(*) || ' rows into ' || '$DB_PATH' FROM breakdown_turns;
SQL

echo "Done. Rows by agent:"
sqlite3 "$DB_PATH" "SELECT agent, COUNT(*) FROM breakdown_turns GROUP BY agent ORDER BY agent;"
