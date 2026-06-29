// Mirror of the Rust serialize contract in
// `crates/llmtrim-ledger/src/dashboard.rs`. These structs derive `Serialize`
// with NO rename, so JSON keys are the Rust field names verbatim (snake_case).

export interface AgentCard {
  agent: string;
  display_name: string;
  input_before: number;
  input_after: number;
  saved_pct: number;
  has_savings_data: boolean;
  bill_micros: number;
  cache_read_tokens: number;
  trend: number[];
  last_event_ts: string | null;
}

export interface DashboardTotals {
  input_before: number;
  input_after: number;
  saved_pct: number;
  bill_micros: number;
}

export interface Dashboard {
  cards: AgentCard[];
  totals: DashboardTotals;
  generated_at: string;
  next_update_secs: number;
}
