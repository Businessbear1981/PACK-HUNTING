"""
MetaController — decides which agents to spawn next by reading the store's
current coverage, never by talking to an agent. It is a pure function of
store state: same coverage snapshot in, same task list out. No memory of
its own beyond what's in the store.

This is the "optional" meta-controller from the spec: it does not execute
agents, it only deposits TASK artifacts describing what should run next. The
orchestrator (orchestrator.py) is what actually claims tasks and runs
agents against them — kept separate so the decision policy and the
execution mechanics can be tested independently.
"""
from __future__ import annotations

import hashlib
import logging

from contracts.stigmergy_contracts import (
    AgentRole,
    ArtifactKind,
    StoreQuery,
    TaskContent,
    TaskStatus,
    WriteRequest,
)
from orchestrators.stigmergy.store import ArtifactStore

logger = logging.getLogger(__name__)

# How many parallel Scout/Analyst/Critic tasks to fan out per round in swarm
# mode. Pipeline mode ignores this and runs one task of each kind at a time.
DEFAULT_FANOUT = 3

# A task key is deterministic per gap (same round, same role, same missing
# work), so redispatching it after a failure is retrying the identical
# request. Bounded retries tolerate a transient blip; beyond this, the gap
# is treated as a hard failure and the round halts instead of hammering a
# broken downstream (e.g. an expired API key) for every remaining tick.
MAX_TASK_ATTEMPTS = 2


class Coverage:
    """A structural snapshot — counts and keys, not content payloads. Even
    the one role allowed a broad view doesn't read what agents actually
    said; it reads how much has been said and about what."""

    def __init__(
        self,
        findings: list,
        insights: list,
        critiques: list,
        outlines: list,
        drafts: list,
        reports: list,
        pending_tasks: list,
        unresolved_conflicts: list,
    ) -> None:
        self.findings = findings
        self.insights = insights
        self.critiques = critiques
        self.outlines = outlines
        self.drafts = drafts
        self.reports = reports
        self.pending_tasks = pending_tasks
        self.unresolved_conflicts = unresolved_conflicts

    @property
    def uncritiqued_insight_keys(self) -> list[str]:
        critiqued = {c.content.get("critiqued_insight") for c in self.critiques}
        return [i.key for i in self.insights if i.key not in critiqued]


async def read_coverage(store: ArtifactStore, round_: int) -> Coverage:
    findings = await store.query(StoreQuery(kinds=(ArtifactKind.FINDING,), round_lte=round_))
    insights = await store.query(StoreQuery(kinds=(ArtifactKind.INSIGHT,), round_lte=round_))
    critiques = await store.query(StoreQuery(kinds=(ArtifactKind.CRITIQUE,), round_lte=round_))
    outlines = await store.query(StoreQuery(kinds=(ArtifactKind.OUTLINE,), round_lte=round_))
    drafts = await store.query(StoreQuery(kinds=(ArtifactKind.DRAFT_SECTION,), round_lte=round_))
    reports = await store.query(StoreQuery(kinds=(ArtifactKind.REPORT,), round_lte=round_))
    all_tasks = await store.query(StoreQuery(kinds=(ArtifactKind.TASK,), round_lte=round_))
    pending_tasks = [t for t in all_tasks if t.content.get("status") == TaskStatus.PENDING.value]
    unresolved_conflicts = await store.conflicts(unresolved_only=True)
    return Coverage(findings, insights, critiques, outlines, drafts, reports, pending_tasks, unresolved_conflicts)


def _task_request(round_: int, target_role: AgentRole, directive: str, input_keys: tuple[str, ...] = ()) -> WriteRequest:
    content = TaskContent(directive=directive, target_role=target_role, round=round_, input_keys=input_keys)
    # Hash the full directive rather than truncating it: two fanned-out
    # scout directives that only differ after character 40 (e.g. a long
    # question with "(angle 1/3)" vs "(angle 2/3)" appended) would
    # otherwise collide onto the same key and silently collapse to one
    # task. The hash must stay a pure function of the directive so the
    # same logical gap keeps the same key across repeated dispatch calls —
    # that stability is what lets MAX_TASK_ATTEMPTS track retries per gap.
    digest = hashlib.sha256(directive.encode("utf-8")).hexdigest()[:12]
    return WriteRequest(
        key=f"task:{round_}:{target_role.value}:{digest}",
        kind=ArtifactKind.TASK,
        role=AgentRole.META_CONTROLLER,
        agent_id="meta-controller",
        round=round_,
        content=content.model_dump(mode="json"),
        rationale=f"coverage-driven dispatch of {target_role.value} for round {round_}",
    )


async def decide_next_tasks(
    store: ArtifactStore,
    question: str,
    round_: int,
    *,
    fanout: int = DEFAULT_FANOUT,
) -> list[WriteRequest]:
    """
    The whole policy, in priority order:

    1. No findings yet this round -> dispatch Scouts (fanned out for swarm).
    2. Findings outrun insights -> dispatch an Analyst.
    3. Any insight has no critique -> dispatch a Critic at it.
    4. Everything critiqued but no outline -> dispatch the Synthesizer.
    5. Outline exists but no report -> dispatch the Writer.
    6. Otherwise -> nothing; the round is done.

    Each rule only fires once its predecessor's gap is closed, so the system
    can't skip a rung (mirrors the promotion-ladder discipline: don't
    synthesize ungrounded insight, don't write an uncritiqued report).
    """
    coverage = await read_coverage(store, round_)

    if coverage.unresolved_conflicts:
        logger.warning(
            "stigmergy_unresolved_conflicts",
            extra={"round": round_, "count": len(coverage.unresolved_conflicts)},
        )

    if not coverage.findings:
        # Fan out N distinct sub-questions so swarm mode has independent,
        # non-colliding Scout tasks instead of N scouts repeating one query.
        sub_queries = _fan_out_query(question, fanout)
        requests = [_task_request(round_, AgentRole.SCOUT, q) for q in sub_queries]
        return await _dispatch_or_halt(store, requests, round_)

    if len(coverage.insights) < len(coverage.findings):
        analyzed = {i.content.get("source_finding") for i in coverage.insights}
        unanalyzed = [f.key for f in coverage.findings if f.key not in analyzed]
        if unanalyzed:
            requests = [_task_request(round_, AgentRole.ANALYST, f"analyze {len(unanalyzed)} finding(s)", tuple(unanalyzed))]
            return await _dispatch_or_halt(store, requests, round_)

    uncritiqued = coverage.uncritiqued_insight_keys
    if uncritiqued:
        requests = [_task_request(round_, AgentRole.CRITIC, f"critique {len(uncritiqued)} insight(s)", tuple(uncritiqued))]
        return await _dispatch_or_halt(store, requests, round_)

    if not coverage.outlines:
        requests = [_task_request(round_, AgentRole.SYNTHESIZER, "synthesize outline and section drafts")]
        return await _dispatch_or_halt(store, requests, round_)

    if not coverage.reports:
        requests = [_task_request(round_, AgentRole.WRITER, "assemble final report")]
        return await _dispatch_or_halt(store, requests, round_)

    return []


async def _attempts_exhausted(store: ArtifactStore, key: str) -> bool:
    history = await store.read_all_versions(key)
    failed = sum(1 for r in history if r.content.get("status") == TaskStatus.FAILED.value)
    return failed >= MAX_TASK_ATTEMPTS


async def _dispatch_or_halt(store: ArtifactStore, requests: list[WriteRequest], round_: int) -> list[WriteRequest]:
    """The one choke point every dispatch decision passes through. If any
    of the tasks about to be (re)dispatched has already failed
    MAX_TASK_ATTEMPTS times, halt the round instead of retrying the same
    request forever — fail closed, don't spin."""
    for req in requests:
        if await _attempts_exhausted(store, req.key):
            logger.error(
                "stigmergy_round_stalled",
                extra={"round": round_, "key": req.key, "reason": f"exceeded {MAX_TASK_ATTEMPTS} failed attempts"},
            )
            return []
    return requests


def _fan_out_query(question: str, fanout: int) -> list[str]:
    if fanout <= 1:
        return [question]
    return [f"{question} (angle {i + 1}/{fanout})" for i in range(fanout)]
