# Heavy Launch Wrapper Notes

## Purpose
`scripts/launch_heavy_stream.sh` is the single auditable entrypoint for starting the heavy `dit_stream.py` worker.

It is intentionally narrower than a full operating policy.
It does **not** decide RAM/VRAM thresholds, OOM handling, or travel-safe recovery.
It only enforces a minimal truthful launch contract.

## Current wrapper contract
- requires explicit `--scene`, `--quality`, and `--steps`
- denies launch if protected prerequisites are not up:
  - `llama-main.service`
  - port `18080`
  - `openclaw-gateway.service`
  - port `18789`
  - Docker availability
  - `inspatio-world` container availability
- denies launch if another heavy worker is already active
- can temporarily gate `gemma-e2b.service` on `:18081`
- records wrapper truth in `interactive_io/heavy_launch_state.json`
- starts a restore watcher so gated `:18081` service is brought back after worker exit
- distinguishes `denied`, `launching`, `running` / `degraded`, `stopped`, and `crashed`

## Validation history
### 2026-04-15 wrapper hardening validation
Confirmed after hardening:
- shell syntax passed
- viewer compile passed
- dry-run preflight returned truthful degraded state when Gemma would be gated:
  - `dry_run_preflight_ok_gemma_would_be_gated`
- viewer service on `:7861` still recovered cleanly after restart

### 2026-04-16 light validation pass
Ran:
```bash
scripts/launch_heavy_stream.sh --scene IMG_7643.mp4 --quality draft --steps 25 --dry-run
```
Observed result:
```json
{"status":"denied","reason":"llama_main_service_inactive","gated_gemma":false,"pid":null}
```
Supporting host truth at validation time:
- `openclaw-gateway.service` was active
- port `18789` was listening
- Docker container `inspatio-world` was up
- `llama-main.service` was failed
- `gemma-e2b.service` was failed
- neither `:18080` nor `:18081` was listening

Interpretation:
- the wrapper is behaving truthfully under unmet prerequisites
- this is a **good deny**, not a wrapper malfunction
- current block is upstream service readiness, not heavy-worker launch syntax

## Operator guidance
- Treat `denied` as a prerequisite/operator state problem first.
- Treat `crashed` as a heavy worker/runtime failure after a valid launch attempt.
- Keep explicit threshold policy separate until launch behavior and operator flow stay stable.
- Do not call the system travel-safe or self-recovering based on this wrapper alone.
