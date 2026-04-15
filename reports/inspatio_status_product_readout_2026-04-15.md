# InSpatio Status Product Readout

Date: 2026-04-15
Prepared for: Pat

## Where we stand
InSpatio is past the pure prototype stage, but it is not yet a finished product. It is now a working specialized remote tool with a real control surface, a supervised viewer path, and a proven heavy render path. The core capability exists. The main remaining work is productization, operational safety, and remote-use trust.

## What is genuinely working now
- The core pipeline works.
- The interactive viewer exists and is reachable remotely.
- The lightweight viewer path on `:7861` is supervised under user-systemd.
- The heavy render path has been proven to work for a tested scene and settings.
- The UI now includes an explicit kill/cancel control to stop the current session and free GPU.
- Heavy launch now goes through a dedicated wrapper script instead of scattered process logic.

## The real product shape
The app is really two systems joined together:
1. A lightweight remote control surface that should always be safe, reachable, and understandable.
2. A heavy GPU job runner that is expensive, risky, and should only start deliberately.

That split is the core design truth. Many current issues come from the fact that the control plane is getting productized while the compute plane still behaves like an operator-driven runtime.

## Main UX truth
For mobile use, especially from an iPhone somewhere else in the world, the app has to feel safe to open, safe to leave alone, safe to stop, and honest about what it is doing. It is improved, but it still exposes too much backend fragility to the user.

## Biggest current gaps
1. The control path is getting solid, but the compute path is still risky.
2. Resource policy is not yet first-class in the product.
3. Recovery is better, but not yet strong enough for fully relaxed remote use.
4. The UI is functional, but it is still shaped partly around system internals rather than the user journey.
5. The app still needs a clearer session/job lifecycle model for remote users.

## What needs to be done next
### 1. Keep the viewer as the permanent safe shell
The viewer should remain the stable front door. It should always load, always show system state, always let the user stop work, and never itself be the dangerous part.

### 2. Treat heavy generation as a managed job
The app should feel like: request heavy session, run preflight, allow or degrade or deny clearly, run, cancel safely, restore services, and report the outcome truthfully.

### 3. Add explicit preflight allow/degrade/deny policy
This is the highest-value next product step. Before heavy start, the app should check enough state to decide whether it is safe to proceed, whether it should degrade, or whether it should deny.

### 4. Keep remote-first controls simple
The top-level controls should stay centered on the real user jobs: start, stop, switch scene, quality mode, session timer, and health/status.

### 5. Harden remote recovery
True travel-safe confidence still needs stronger recovery behavior and eventually external recovery support, because SSH is not always enough when memory pressure goes bad.

## Current strategic conclusion
InSpatio is past “can it work?” and now in “can it be trusted?”

That is a good place to be. The hard capability is there. The main work left is lifecycle design, resource policy, remote-safe UX, and recovery confidence.

## Best way to go about it
The best next move is:
- keep the viewer as the safe control plane
- keep the heavy path on-demand only
- add explicit preflight allow/degrade/deny logic before heavy session start
- keep threshold policy, OOM policy, and broader orchestration separate until the wrapper/operator flow is settled

## Recommended next 3 implementation steps
1. Validate and lightly document the current heavy-launch wrapper behavior.
2. Add explicit threshold-backed allow/degrade/deny policy as a separate pass.
3. Tighten the UI around clear lifecycle states and remote-safe operator actions.
