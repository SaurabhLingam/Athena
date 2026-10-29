# Changelog

## 0.3.0 — 2026-10-29

### Added
- MCP server for AI agents (`pip install "athena-eda[mcp]"`, run with `athena-mcp` or `athena mcp`). Tools: `describe_dataset`, `check_leakage`, `run_baselines`, `check_drift`. Each takes CSV paths, accepts an optional `max_rows` to sample large files, and returns a compact JSON summary.
- Bundled Claude Code skill and `athena install-skill` command, which copies it to `~/.claude/skills/athena/`.

### Fixed
- Near-unique float columns (prices, measurements) were treated as identifiers and silently skipped by drift detection and leakage checks. Floats are now treated as IDs only when their name looks like one (`id`, `*_id`, `uuid`, `key`, …).
- The leakage check missed near-copies of the target in numeric columns with few distinct values, because those columns were routed to the categorical path. Every numeric feature is now also scored by |correlation| (regression) or single-feature AUC (classification) against the existing 0.95 threshold.
