# ADR-0005: Multi-agent research runs on a shared, versioned ArtifactStore, not agent-to-agent messages

**Status:** Accepted · 2026-09-21
**Deciders:** Sean Gilmore (product ask), implemented this session

## Context

We needed a deep-research capability: given a question, discover sources, extract claims, challenge
them, synthesize an outline and draft, and assemble a cited report — usable in both a strictly
sequential mode and a concurrent multi-agent mode, without corrupting shared state when several
agents write at once.

The alternative on the table was the conventional "manager agent chats with worker agents" pattern
(a controller LLM call that dispatches sub-agents via direct function calls / message passing and
collects their return values). That pattern was explicitly out of scope for this build: it makes the
manager a single point of failure and a hidden global lock, it has no natural audit trail beyond
whatever the manager logs, and "concurrent" workers under it are really just concurrent function
calls whose results the manager serializes by hand — the manager itself never operates safely under
real concurrency without becoming the thing this system exists to avoid needing.

## Decision

Agents never call each other and never pass messages directly. All coordination is mediated by one
`ArtifactStore` (`backend/orchestrators/stigmergy/store.py`): every agent's only inputs and outputs
are reads and writes against it, filtered through a `MyopicView` scoped to the artifact kinds that
role is allowed to see (`backend/orchestrators/stigmergy/agents.py`). A `MetaController`
(`meta_controller.py`) decides what work is missing by reading the store's coverage — never by
talking to an agent — and deposits `TASK` artifacts for an orchestrator to claim and execute
(`orchestrator.py`). Every artifact is append-only and optimistically-versioned; see
`docs/reference/stigmergy-research.md` for the concurrency model in full.

## Why

- **No single point of failure or hidden serialization.** With environment-mediated coordination, a
  crashed or slow agent doesn't block a manager's call stack — it just leaves a gap in store
  coverage that the next round's `MetaController` decision (or another swarm worker) can fill.
- **The audit trail is the mechanism, not a side effect.** Because every write already has to name
  `agent_id`, `confidence`, `parent_versions`, and `rationale` to satisfy the store's contract, the
  audit log isn't a logging feature bolted on afterward — it's what coordination *is* in this design.
- **Swarm mode falls out for free.** Once coordination is "read/write a shared store," running three
  Scouts concurrently is a scheduling decision (`workers_per_role` in `run_swarm`), not a redesign.
  A manager-chats-with-workers architecture would need real work to make that safe; here it's the
  same `write(expected_version=...)` path pipeline mode already uses.
- **Matches an existing local precedent.** `backend/services/state_store.py` (ORCA / The Field)
  already established "durable JSON, atomic replace, fail loud on read, degrade loud on write" as
  this repo's house pattern for shared state. `FileArtifactStore` is that same precedent applied to
  a versioned/append-only shape instead of a flat key-value one, rather than introducing a second
  durability philosophy.

## Consequences

- Every new agent role must be written against `ArtifactStore`/`MyopicView`, not given a reference to
  another agent. A future contributor reaching for direct agent-to-agent calls is reintroducing the
  pattern this ADR rejected.
- `FileArtifactStore`'s cross-process safety is bounded (documented in the store's own module
  docstring): it protects one process with many concurrent coroutines, not many OS processes sharing
  one file. Multi-process fan-out needs a Redis/Postgres-backed `ArtifactStore` implementation before
  it can be trusted — the interface is designed for that swap, but no such backend exists yet.
- `MAX_TASK_ATTEMPTS` in `meta_controller.py` means a persistently broken downstream (an expired API
  key, a dead search index) halts a round after two failed attempts rather than producing a
  degraded report. That is deliberate (fail closed per house style), but it means "no report" is a
  possible and expected outcome, not just a bug symptom — callers must check for `None`.
