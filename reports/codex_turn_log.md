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
