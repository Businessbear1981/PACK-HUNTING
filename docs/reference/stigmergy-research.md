# Stigmergy multi-agent research system

Code: `backend/contracts/stigmergy_contracts.py`, `backend/orchestrators/stigmergy/`.
Tests: `backend/tests/test_stigmergy_store.py`, `backend/tests/test_stigmergy_orchestrator.py`.
Decision record: [ADR-0005](../adr/0005-stigmergy-multi-agent-research.md).

## Philosophy

Agents never talk to each other. Every coordination signal is a read or a write against one shared
`ArtifactStore`. No agent — including the MetaController — sees the whole store; each gets a
`MyopicView` scoped to the artifact kinds its role needs. The colony's output (a cited report) is a
side effect of agents depositing findings, insights, critiques, and drafts into the environment over
one or more rounds; nothing routes a message from one named agent to another.

## The shared artifact store

### Interface

`ArtifactStore` (`store.py`) is the one abstraction every agent, the orchestrator, and the
MetaController depend on:

```python
class ArtifactStore(abc.ABC):
    async def write(self, request: WriteRequest, *, on_conflict: "reject"|"deposit" = "reject") -> WriteResult: ...
    async def read(self, key: str) -> ArtifactRecord | None: ...          # latest version
    async def read_all_versions(self, key: str) -> list[ArtifactRecord]: ...
    async def query(self, q: StoreQuery) -> list[ArtifactRecord]: ...      # filtered "myopic view"
    async def audit_log(self) -> list[ArtifactRecord]: ...                 # every version, in order
    async def conflicts(self, *, unresolved_only: bool = True) -> list[ConflictRecord]: ...
    def status(self) -> dict: ...                                          # health, never raises
```

Two concrete implementations ship: `InMemoryArtifactStore` (default for tests and cheap swarm demos)
and `FileArtifactStore` (durable, built on `backend/services/state_store.py`'s atomic
temp-file-then-`os.replace` pattern — the same one ORCA/The Field use).

### What every artifact carries

`ArtifactRecord` (Pydantic v2, `strict=True, frozen=True` — a true value object) is the unit of
audit: `key`, `kind`, `version`, `round`, `role`, `agent_id`, `content`, `confidence`, `rationale`,
`parent_versions`, `written_at`. `rationale` is validated non-blank at construction — a write with no
stated reason is rejected before it ever reaches the store. This is the "every write is auditable"
requirement satisfied structurally, not by convention.

Kinds: `finding` (Scout), `insight` (Analyst), `critique` (Critic), `outline` / `draft_section`
(Synthesizer), `report` (Writer), `conflict` (deposited on a lost optimistic write), `task`
(MetaController's dispatch unit).

### Concurrency model

Every artifact is **append-only**: `write()` never overwrites a version, it adds a new one under the
same key. That one mechanism is deliberately used to implement *both* patterns the design calls for,
rather than building two separate code paths:

1. **Optimistic concurrency.** Pass `expected_version`. If the key's current head isn't at that
   version, the write returns `WriteOutcome.CONFLICT` — and, if the caller passed
   `on_conflict="deposit"`, a `ConflictRecord` (who tried, what they expected, what was actually
   there, their attempted content) is added to the store's conflict log for a Critic or
   MetaController to resolve later. Nothing is silently overwritten.
   `ArtifactStore.write_with_retry(key, build_request)` implements the "retry" half directly: it
   re-reads the current head, lets the caller rebuild its write against it, and retries up to a
   bound.
2. **Append-only / event-sourced.** Pass no `expected_version`. The write is accepted unconditionally
   as another proposal. `read_all_versions(key)` returns the full history; a Synthesizer reconciles
   multiple agents' proposals by reading that history, never by one of them overwriting another's.

This is why concurrent Scouts or Analysts in swarm mode are safe by construction: they write distinct
finding/insight keys (no collision) or race for the *same* logical unit of work — a `TASK`
artifact — using optimistic writes, where exactly one winner is guaranteed
(`orchestrator._claim_task`, covered directly by
`test_concurrent_claim_of_the_same_task_has_exactly_one_winner`).

### Conflict resolution strategy (explicit, not last-write-wins)

A losing write is told it lost, and gets the winner's actual version back so it can decide what to
do — retry against the new head, or stop. Nothing is silently clobbered. The two situations that
actually generate conflicts in this system are: (a) two swarm workers racing to claim the same
`TASK`, where losing is the *expected*, harmless outcome (the loser just tries the next pending
task), and (b) a caller deliberately using `on_conflict="deposit"` to make a stale write visible in
the audit trail rather than raising or silently discarding it.

### Graceful degradation

`FileArtifactStore` wraps `StateStore`, which is fail-loud on read (a state file that can't be read
raises rather than starting empty and pretending nothing was lost) and fail-soft-but-honest on write
(if the durable backend stops accepting writes, it degrades to in-memory and every subsequent
`WriteResult.outcome` is `DEGRADED` with the reason in `note` — never a silent success that quietly
stopped persisting).

### What this does *not* cover

True cross-process mutual exclusion. The lock (`threading.Lock` for the file backend, `asyncio.Lock`
for the in-memory one) is process-local. Two separate OS processes writing the same
`FileArtifactStore` file can still race between "load" and "atomic replace" — the same boundary
`StateStore` itself documents. Multi-process fan-out needs a real broker behind the same
`ArtifactStore` interface (see "Swapping the backend" below), not a bigger version of this file.

## Swapping the backend

Nothing above `ArtifactStore` cares which concrete class it's holding. To add Redis or Postgres:

- **Redis:** store each key's version list as a Redis list (or a sorted set keyed by version) and use
  `WATCH`/`MULTI`/`EXEC` (or a Lua script) to make the "read current version, compare, append" step
  atomic across processes — the same three-step shape `write()` already implements in-process.
- **Postgres:** one `artifacts` table (`key`, `version`, `kind`, `role`, `agent_id`, `content jsonb`,
  `confidence`, `rationale`, `parent_versions text[]`, `written_at`), unique constraint on
  `(key, version)`, and an `INSERT ... WHERE (SELECT max(version) FROM artifacts WHERE key = ...) =
  expected_version` (or a `SERIALIZABLE` transaction) to get the same conflict-on-mismatch behavior
  for free from the database's own concurrency control.
- **Vector DB (for the search side, not the artifact store):** `SearchProvider` is the seam — a
  Pinecone/pgvector-backed implementation of `search(query, limit) -> list[SearchHit]` slots in next
  to `LocalCorpusSearchProvider` without touching any agent.

No such backend ships in this change — the interface is designed for the swap, but building and
testing it against a real Redis/Postgres instance is future work, not something to fake here.

## Agent roles and myopia

`Agent` (`agents.py`) is the base class every role extends. Each instance holds a `MyopicView`, not
the raw store — a wrapper that raises `PermissionError` if the agent's code asks to read a kind
outside its role's allowed set (`ROLE_VIEWS`). This is the "no agent has the full picture" constraint
enforced in code (`test_myopic_view_blocks_reading_a_kind_outside_its_role`), not left to each
implementation's self-discipline:

| Role | Reads | Writes | Job |
|---|---|---|---|
| Scout | `TASK` | `FINDING` | Runs a `SearchProvider` query, deposits one finding per hit |
| Analyst | `FINDING`, `TASK` | `INSIGHT` | Extracts claims + confidence via `LLMProvider.extract_insight` |
| Critic | `INSIGHT`, `FINDING`, `TASK` | `CRITIQUE` | Grounds each insight against its cited finding, scores quality |
| Synthesizer | `INSIGHT`, `CRITIQUE`, `FINDING`, `TASK` | `OUTLINE`, `DRAFT_SECTION` | Clusters insights into headings weighted by confidence × critique quality, drafts each section |
| Writer | `OUTLINE`, `DRAFT_SECTION`, `CRITIQUE`, `TASK` | `REPORT` | Assembles the final report plus a Limitations section built from logged critique issues |
| MetaController | everything (structural counts only, by design — see below) | `TASK` | Decides what's missing and deposits the next unit of work |

The MetaController is the one role given a broad `ArtifactStore` view, because its entire job is
seeing what's missing across the whole board — but its policy (`decide_next_tasks`) only inspects
counts and keys (`Coverage` in `meta_controller.py`), never an artifact's actual content payload.

### Providers — the only place the system touches the outside world

`SearchProvider` and `LLMProvider` (`providers.py`) are the seams between agents and reality:

- `LocalCorpusSearchProvider` does real term-overlap search over local `.md`/`.txt` files (stopwords
  excluded from scoring; headings, tables, fenced code blocks, and other non-prose lines excluded
  from what can be indexed or cited as a source, so a claim never turns out to be a section heading).
  No web-search API is wired into this environment; this is an honest default, not a stand-in for one
  — point it at a real corpus, or implement `SearchProvider` against Tavily/Serper/Brave for live web
  search without touching any agent.
- `AnthropicLLMProvider` wraps the existing `backend/agents/_claude.py` client (Anthropic direct, or
  OpenRouter fallback, same priority order).
- `HeuristicLLMProvider` is a real, deterministic, dependency-free fallback (keyword-overlap claim
  extraction, grounding-based critique scoring) used when no API key is configured. It is weaker than
  an LLM and every result it produces says so via a `method` field — it is not a mock, it computes a
  genuine answer from genuine input every time, just a cruder one.
- `select_llm_provider()` picks once per run (not per call), so a report never silently mixes
  LLM-grade and heuristic-grade sections without the run knowing which it got.

## Execution modes

`orchestrator.py` implements both required modes over the same claim/execute mechanics
(`_claim_task`, `_finish_task`, `_claim_and_run_one`):

- **`run_pipeline`** — linear, ant-like accumulation. Each tick: `decide_next_tasks(fanout=1)`
  deposits exactly one task, `_drain_role_sequential` claims and runs it to completion, then the next
  tick's decision is made. One thing happens at a time.
- **`run_swarm`** — concurrent accumulation. Each tick: `decide_next_tasks(fanout=N)` deposits N
  independent tasks (e.g. N distinct Scout sub-queries), and `workers_per_role` agent instances race
  via `asyncio.gather` to claim and drain them (`_drain_role_concurrent`). Their results simply
  accumulate in the store; nothing coordinates the workers directly.
- **`run_event_driven`** — instead of a caller explicitly stepping the pipeline, this polls the
  store's audit-log length as its trigger signal: growth means an agent just wrote something, which
  is the cue to re-evaluate what's next. It stops a round once neither the log nor the coverage
  decision has changed for a few consecutive polls.

MetaController policy (`decide_next_tasks`) is a strict promotion ladder — no findings yet dispatches
Scouts; findings outrunning insights dispatches an Analyst; any uncritiqued insight dispatches a
Critic; only once everything is critiqued does a Synthesizer run; only once an outline exists does
the Writer run. A stage can't be skipped, mirroring "don't synthesize ungrounded insight, don't write
an uncritiqued report."

### Failure containment (fail closed, not infinite retry)

A `TASK` key is deterministic per logical gap (same round, role, and missing work), so redispatching
it after a failure would otherwise retry the identical request forever against a persistently broken
downstream (this was caught live: an expired `ANTHROPIC_API_KEY` in this environment made the
Analyst stage fail every attempt, and the first version of this policy burned all 200 ticks
redispatching it before halting). `MAX_TASK_ATTEMPTS` (2) bounds this: `_dispatch_or_halt` checks
each task key's failure history before dispatching, and if a gap has already failed twice, the round
halts (`decide_next_tasks` returns `[]`, logged as `stigmergy_round_stalled`) instead of spinning.
`run_pipeline`/`run_swarm` return `None` when a round halts without ever producing a report — callers
must check for that rather than assume a report always comes back.

## Configuration: pipeline vs. swarm

Both modes take the same `ArtifactStore` and `AgentFactory(search_provider, llm_provider)`:

```python
factory = AgentFactory(LocalCorpusSearchProvider(corpus_dir), select_llm_provider())

# Sequential — one task at a time, deterministic ordering
report = await run_pipeline(store, question, factory, rounds=1)

# Concurrent — N sub-queries fanned out, M workers racing per role
report = await run_swarm(store, question, factory, rounds=1, fanout=3, workers_per_role=3)

# Event-driven — reacts to store growth instead of a per-tick caller
report = await run_event_driven(store, question, factory, fanout=3, workers_per_role=3)
```

`backend/orchestrators/stigmergy/demo.py` is a runnable CLI wired to all three modes, both store
backends, and an explicit heuristic-vs-auto LLM switch:

```bash
cd backend
python -m orchestrators.stigmergy.demo "What database does NEST use and why?"
python -m orchestrators.stigmergy.demo "..." --mode swarm --workers 4 --fanout 3
python -m orchestrators.stigmergy.demo "..." --mode event
python -m orchestrators.stigmergy.demo "..." --backend file --namespace my_run   # durable
python -m orchestrators.stigmergy.demo "..." --llm heuristic                     # force offline fallback
```

## Known limitations (stated, not hidden)

- `FileArtifactStore` is safe within one process; multi-process fan-out needs a real broker (see
  "Swapping the backend"). Not built here.
- `LocalCorpusSearchProvider` searches local files, not the live web. It is the honest default for an
  environment with no search API configured, not a placeholder for one.
- `HeuristicLLMProvider`'s confidence and quality scores are simple overlap heuristics, not calibrated
  probabilities — every score they produce is labeled with its `method` so a report can be judged
  accordingly.
- A round that halts (`MAX_TASK_ATTEMPTS` exceeded) returns `None`, not a partial report. Producing a
  best-effort report from an incomplete round was deliberately not built — it would mean fabricating
  a claim of completeness the run doesn't have.
