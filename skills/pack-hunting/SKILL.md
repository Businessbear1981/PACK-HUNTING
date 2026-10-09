---
name: pack-hunting
description: Run a multi-agent research round where Scout, Analyst, Critic, Synthesizer and Writer roles coordinate only through a shared append-only artifact store and never message each other. Use when the user asks for multi-agent deep research, a cited report built by several roles, or a swarm run over a document corpus.
---

# Pack Hunting

Pack Hunting is a stigmergy-style research procedure. Agents never talk to each other. Every
coordination signal is a read or a write against one shared artifact store, and the cited report
is what remains after the roles have deposited their findings over one or more rounds.

## When to use

- The user wants a sourced report assembled by several independent passes (search, analysis,
  critique, synthesis, writing) rather than one answer.
- The work should be auditable: every claim traces back to the finding it came from.
- Several workers may run concurrently and must not overwrite each other.

Do not use it for a simple question with one answer. It costs several passes for no gain.

## Roles

| Role | Reads | Writes | Job |
|---|---|---|---|
| Scout | task | finding | Runs a search query and deposits one finding per hit |
| Analyst | finding, task | insight | Extracts claims with a confidence score |
| Critic | insight, finding, task | critique | Checks each insight against the finding it cites |
| Synthesizer | insight, critique, finding, task | outline, draft_section | Groups insights under headings, weighted by confidence and critique quality |
| Writer | outline, draft_section, critique, task | report | Assembles the cited report and a Limitations section from logged critique issues |
| MetaController | counts and keys only | task | Checks what is missing and deposits the next unit of work |

Each agent sees only the kinds its role is allowed to read. Reading anything else is refused in
code, not by convention.

## Procedure

1. Deposit the question as a task.
2. Scout runs the search and deposits findings.
3. Analyst deposits insights, each with a confidence score.
4. Critic deposits a critique for each insight, grounded in its cited finding.
5. Synthesizer deposits an outline and draft sections.
6. Writer deposits the report.
7. MetaController checks coverage and deposits the next tasks. After a bounded number of
   attempts on a task that keeps failing, it halts and reports rather than retrying forever.

## Rules that hold in every run

- Every artifact carries an agent id, round, confidence, parent versions and a non-blank
  rationale. A write with no rationale is rejected.
- Artifacts are append-only. A new version is added under the same key; nothing is overwritten.
- If two writers race for the same unit of work, the loser gets the winner's current version back
  and the conflict is recorded. Nothing is silently replaced.
- Unknowns go into the Limitations section. Do not fill gaps with invented sources.

## Limits

- The store's lock is process-local. Separate processes writing the same file store can still
  race between load and replace. Multi-process runs need a shared backend behind the same store
  interface.
- The Redis and Postgres backends are designed as drop-in swaps but are not shipped.
- The only links to the outside world are the search provider and the LLM provider.

## Reference implementation

The Python package in this repo is the reference implementation:
`backend/orchestrators/stigmergy/` (roles, store, providers, orchestrator, demo),
`backend/contracts/stigmergy_contracts.py` (shared types), and the two test files
`backend/tests/test_stigmergy_store.py` and `backend/tests/test_stigmergy_orchestrator.py`.
Design notes: `docs/reference/stigmergy-research.md` and `docs/adr/0005-stigmergy-multi-agent-research.md`.
