# Codex Turn Log

## 2026-04-14 10:46 EDT — Phase 0 viewer supervision
- **Goal:** Make the lightweight InSpatio viewer recoverable without touching heavy-path behavior.
- **Changed:** Added user-systemd supervision for `stream_viewer.py`, switched viewer runtime from ad hoc process to managed service, verified `:7861` listener, verified local HTTP reachability, and verified restart by killing the service PID and confirming systemd brought it back.
- **Why:** Protect the remote control path first, without building on unproven heavy-path assumptions.
- **Files touched:**
  - `ops/systemd/inspatio-stream-viewer.service`
  - deployed copy at `~/.config/systemd/user/inspatio-stream-viewer.service`
- **Important file locations:**
  - viewer code: `stream_viewer.py`
  - turn log: `reports/codex_turn_log.md`
  - deployed unit: `~/.config/systemd/user/inspatio-stream-viewer.service`
- **Deliberately not changed:** heavy worker logic, helper gating, threshold logic, OOM automation, anything touching `:18080`.
- **Status:** Done and verified.
- **Next step:** Run the heavy-path proof pass only.

## 2026-04-14 11:00 EDT — Heavy-path proof pass
- **Goal:** Prove whether the heavy InSpatio path is healthy enough to wrap before adding policy logic.
- **What changed:** Ran one controlled heavy proof on `IMG_7643.mp4`, captured live status/log/resource evidence, and wrote proof artifacts.
- **Why:** Avoid building wrapper or threshold policy around an unproven runtime.
- **Files touched:**
  - `reports/heavy_path_proof_2026-04-14.md`
  - `reports/heavy_path_proof_2026-04-14.json`
  - `reports/codex_turn_log.md`
- **Important file locations:**
  - proof summary: `reports/heavy_path_proof_2026-04-14.md`
  - proof data: `reports/heavy_path_proof_2026-04-14.json`
  - active viewer/service context: `stream_viewer.py`, `ops/systemd/inspatio-stream-viewer.service`
- **Deliberately not changed:** wrapper logic, threshold logic, OOM automation, `:18080` coordination, architecture.
- **Status:** Proof pass succeeded; heavy path streamed and looped stably for the tested scene, with tight resource headroom.
- **Next step:** Build the minimum heavy-launch wrapper only, using measured guardrails from this pass.

## 2026-04-15 09:40 EDT — Minimum heavy-launch wrapper
- **Goal:** Isolate heavy worker start behavior into one auditable launch path without adding threshold policy yet.
- **What changed:** Moved the `dit_stream.py` spawn command behind `scripts/launch_heavy_stream.sh`, added a lightweight `interactive_io/heavy_launch_request.json` handoff record, and updated `stream_viewer.py` to use the wrapper and surface launch failures clearly.
- **Why:** Make heavy startup easier to reason about, validate, and evolve without mixing in resource-policy logic too early.
- **Files touched:**
  - `stream_viewer.py`
  - `scripts/launch_heavy_stream.sh`
  - `reports/codex_turn_log.md`
- **Important file locations:**
  - viewer entrypoint: `stream_viewer.py`
  - heavy launch wrapper: `scripts/launch_heavy_stream.sh`
  - launch handoff record: `interactive_io/heavy_launch_request.json`
- **Validation:**
  - `python3 -m py_compile stream_viewer.py` with a temp pycache prefix
  - `bash -n scripts/launch_heavy_stream.sh`
  - `scripts/launch_heavy_stream.sh --scene ScreenRecording_04-14-2026_01-17-32_1.mp4 --quality scout --steps 2 --dry-run`
  - restarted `inspatio-stream-viewer.service`, re-verified HTTP `200`, websocket responsiveness, and `status.json == stopped`
- **Status:** Done and validated without launching a new heavy render.
- **Next step:** Use this wrapper as the seam for a later preflight/policy layer, keeping threshold logic separate until the wrapper shape settles.

## 2026-04-15 18:15 EDT — Wrapper hardening: truthful preflight + gemma restore
- **Goal:** Make the existing launch wrapper actually enforce the smallest safe preflight/cleanup contract without widening scope.
- **What changed:** Added preflight checks for protected prerequisites (`llama-main.service` / `:18080`, OpenClaw gateway / `:18789`), deny-on-unsafe behavior for duplicate heavy workers and broken prerequisites, optional `gemma-e2b.service` gating on `:18081`, background restore of `:18081` after worker exit, and truthful wrapper state recording in `interactive_io/heavy_launch_state.json`. Updated `stream_viewer.py` to surface `denied` vs `crashed`, mark `launching` before wrapper handoff, and restore the gateable lane during cleanup.
- **Why:** Finish the minimum real wrapper so heavy launch is protected, reversible, and auditable before any threshold policy exists.
- **Files touched:**
  - `stream_viewer.py`
  - `scripts/launch_heavy_stream.sh`
  - `reports/codex_turn_log.md`
- **Important file locations:**
  - wrapper state: `interactive_io/heavy_launch_state.json`
  - wrapper request handoff: `interactive_io/heavy_launch_request.json`
  - heavy launch wrapper: `scripts/launch_heavy_stream.sh`
- **Validation:**
  - `PYTHONPYCACHEPREFIX=/tmp/inspatio_pycache python3 -m py_compile stream_viewer.py`
  - `bash -n scripts/launch_heavy_stream.sh`
  - dry-run preflight returned truthful degraded state: `dry_run_preflight_ok_gemma_would_be_gated`
  - force-cycled `inspatio-stream-viewer.service` and re-verified HTTP `200` on `:7861` and `/api/library`
- **Status:** Done and validated at the wrapper/preflight level without adding thresholds or OOM policy.
- **Next step:** Do one light validation/documentation pass on the wrapper behavior, then separately decide explicit threshold policy.

## 2026-04-16 02:20 EDT — Wrapper validation + operator notes
- **Goal:** Validate the hardened wrapper one more time under real current host conditions and write down the operator meaning of its states.
- **What changed:** Added `interactive_io/heavy_launch_notes.md` documenting the wrapper contract, the existing 2026-04-15 validation, and a fresh 2026-04-16 dry-run validation result.
- **Why:** Make it obvious whether a failed heavy start is a bad launch, a valid deny, or a post-launch crash before adding threshold policy.
- **Files touched:**
  - `interactive_io/heavy_launch_notes.md`
  - `reports/codex_turn_log.md`
- **Validation:**
  - `scripts/launch_heavy_stream.sh --scene IMG_7643.mp4 --quality draft --steps 25 --dry-run`
  - result was a truthful deny: `llama_main_service_inactive`
  - confirmed supporting host state: gateway active on `:18789`, container up, `llama-main.service` failed, `gemma-e2b.service` failed, and neither `:18080` nor `:18081` listening
- **Status:** Done. Wrapper behavior remains truthful; current blocker is prerequisite service readiness, not wrapper ambiguity.
- **Next step:** If heavy use is needed again, restore the protected llama lane first, then re-run a controlled launch check before any threshold work.

## 2026-04-19 04:06 EDT — Viewer supervision recovery + runtime artifact cleanup
- **Goal:** Restore the lightweight viewer path and remove recurring git noise from generated heavy-launch artifacts.
- **What changed:** Re-enabled and started `inspatio-stream-viewer.service` on the host so the viewer stays up across future user-session starts, and added the generated `interactive_io/heavy_launch_request.json` / `interactive_io/heavy_launch_state.json` files to `.gitignore`.
- **Why:** The morning health check found `:7861` fully down, with HTTP and websocket unavailable, even though the wrapper truth files were merely runtime artifacts. Restoring the viewer fixes the live control surface now, and ignoring those files keeps future stabilization wakes focused on real source deltas.
- **Files touched:**
  - `.gitignore`
  - `reports/codex_turn_log.md`
- **Validation:**
  - `systemctl --user enable --now inspatio-stream-viewer.service`
  - verified `systemctl --user status` showed `active (running)` and `enabled`
  - verified HTTP `200` on `http://127.0.0.1:7861/`
  - verified websocket responsiveness on `ws://127.0.0.1:7861/ws`
  - verified `http://127.0.0.1:7861/interactive_io/status.json` returned `status=stopped`
  - verified generated heavy-launch JSON files no longer appear as untracked repo noise
- **Status:** Done and validated. Viewer supervision is back, and the repo surface is cleaner for future real fixes.
- **Next step:** Keep the next wake focused on the actual heavy prerequisite path, especially restoring truthful first-time launch checks once the protected llama lane is available again.

## 2026-04-19 06:09 EDT — Clear stale crash history after clean stop
- **Goal:** Keep the viewer status honest when a prior crash marker lingers after a later clean operator stop.
- **What changed:** Updated `stream_viewer.py` so `read_status_for_viewer()` now folds in `interactive_io/heavy_launch_state.json`, exposes `launch_reason`, and clears stale `previous_status: crashed` when a newer clean stop reason such as `operator_shutdown` or `worker_exited_after_cleanup` supersedes it. Added the same `launch_reason` to `/health`.
- **Why:** The live viewer and websocket were up, but `status.json` was still surfacing an old crash shadow even though the heavy wrapper recorded a later clean stop. That made the control surface less honest than the real workflow state.
- **Files touched:**
  - `stream_viewer.py`
  - `reports/codex_turn_log.md`
- **Validation:**
  - restarted `inspatio-stream-viewer.service`
  - verified `http://127.0.0.1:7861/interactive_io/status.json` now reports `previous_status: null` and `launch_reason: operator_shutdown`
  - verified `http://127.0.0.1:7861/health` includes `launch_reason: operator_shutdown`
  - verified websocket bootstrap on `ws://127.0.0.1:7861/ws` still returns `active_scene`, `status`, and `quality_sync`
- **Status:** Done and validated locally.
- **Next step:** Push the validated commit once GitHub credentials are available, then keep the next wake aimed at real heavy-path health instead of stale status residue.
