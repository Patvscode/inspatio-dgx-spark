# Heavy Path Proof Pass

Date: 2026-04-14 11:00 EDT
Scene: `IMG_7643.mp4`
Scope: one controlled heavy-path proof run only, with no wrapper logic, threshold policy, OOM automation, or `:18080` negotiation.

## Result
- Heavy path healthy enough to wrap: **YES, for this proof pass**
- Final observed state before stop: sustained `streaming`, repeated `looping`, then clean operator stop

## Evidence
- Worker reported startup memory gate: `Free VRAM: 13.4 GB, low_memory=True`
- Worker reached ready state: log contains `[STATUS] ready`
- Worker entered generation: log contains `Starting streaming: 25 blocks, 75 frames`
- Live status during proof: `{"status":"streaming","block":1,"frame":2718,"block_time":0.69,"fps":4.33}`
- Recent log near stop showed steady throughput around `4.3 to 4.5 FPS` and repeated loop restarts
- Live compute-app GPU snapshot during proof included InSpatio worker: `python3, 37207 MiB`
- Baseline summed compute-app GPU usage before launch was about `39668 MiB`
- Host RAM snapshot during proof: `avail=4997 MiB`

## Interpretation
- This pass disproved the earlier crash-only assumption for the current active scene and current settings.
- The heavy path is now healthy enough to justify a minimal wrapper design, but resource headroom is tight enough that threshold policy should still be based on measured guardrails, not guesswork.
- The immediate constraint exposed by this proof is not startup correctness, it is **resource pressure** during sustained streaming.

## Deliberately not changed
- no wrapper logic
- no threshold policy
- no OOM automation
- no `:18080` negotiation
- no architecture changes
