"""
Orchestrator — pipeline (sequential, ant-like) and swarm (concurrent)
execution over the same claim/execute mechanics, plus an event-driven
variant that reacts to the store's audit log growing instead of being
driven by an explicit per-tick caller.

Nothing in here calls one agent from another. The only thing that ever
crosses from one agent's output to another agent's input is a store
write followed by a store read. `_claim_task` is the one place optimistic
concurrency actually gets exercised end to end: multiple workers can see
the same PENDING task and only one of them wins the claim, which is the
property backend/tests/test_stigmergy_orchestrator.py asserts directly.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from contracts.stigmergy_contracts import (
    AgentRole,
    ArtifactKind,
    ArtifactRecord,
    StoreQuery,
    TaskContent,
    TaskStatus,
    WriteOutcome,
    WriteRequest,
)
from orchestrators.stigmergy.agents import (
    AnalystAgent,
    Agent,
    CriticAgent,
    ScoutAgent,
    SynthesizerAgent,
    WriterAgent,
)
from orchestrators.stigmergy.meta_controller import decide_next_tasks
from orchestrators.stigmergy.providers import LLMProvider, SearchProvider
from orchestrators.stigmergy.store import ArtifactStore

logger = logging.getLogger(__name__)


class AgentFactory:
    """Builds one agent instance per (role, agent_id, store). Kept separate
    from the agents themselves so orchestrator code never hardcodes which
    provider goes with which role more than once."""

    def __init__(self, search: SearchProvider, llm: LLMProvider) -> None:
        self._search = search
        self._llm = llm

    def build(self, role: AgentRole, agent_id: str, store: ArtifactStore) -> Agent:
        if role is AgentRole.SCOUT:
            return ScoutAgent(store, agent_id, self._search)
        if role is AgentRole.ANALYST:
            return AnalystAgent(store, agent_id, self._llm)
        if role is AgentRole.CRITIC:
            return CriticAgent(store, agent_id, self._llm)
        if role is AgentRole.SYNTHESIZER:
            return SynthesizerAgent(store, agent_id, self._llm)
        if role is AgentRole.WRITER:
            return WriterAgent(store, agent_id, self._llm)
        raise ValueError(f"no executable agent for role={role}")


async def _claim_task(store: ArtifactStore, task_record: ArtifactRecord, agent_id: str) -> Optional[ArtifactRecord]:
    content = TaskContent.model_validate(task_record.content, strict=False)
    updated = content.model_copy(update={"status": TaskStatus.CLAIMED, "claimed_by": agent_id})
    request = WriteRequest(
        key=task_record.key,
        kind=ArtifactKind.TASK,
        role=AgentRole.META_CONTROLLER,
        agent_id=agent_id,
        round=task_record.round,
        content=updated.model_dump(mode="json"),
        rationale=f"claimed by {agent_id}",
        parent_versions=(task_record.versioned_key,),
        expected_version=task_record.version,
    )
    result = await store.write(request, on_conflict="reject")
    return result.record if result.outcome == WriteOutcome.OK else None


async def _finish_task(store: ArtifactStore, claimed: ArtifactRecord, status: TaskStatus, result_key: Optional[str]) -> None:
    content = TaskContent.model_validate(claimed.content, strict=False)
    updated = content.model_copy(update={"status": status, "result_key": result_key})
    request = WriteRequest(
        key=claimed.key,
        kind=ArtifactKind.TASK,
        role=AgentRole.META_CONTROLLER,
        agent_id=claimed.agent_id,
        round=claimed.round,
        content=updated.model_dump(mode="json"),
        rationale=f"marked {status.value} after execution",
        parent_versions=(claimed.versioned_key,),
        expected_version=claimed.version,
    )
    result = await store.write(request, on_conflict="deposit")
    if result.outcome != WriteOutcome.OK:
        logger.warning("stigmergy_task_finish_conflict", extra={"key": claimed.key, "status": status.value})


async def _claim_and_run_one(
    store: ArtifactStore,
    role: AgentRole,
    round_: int,
    factory: AgentFactory,
    question: str,
    agent_id: str,
) -> Optional[list[ArtifactRecord]]:
    """Try every currently-PENDING task for `role` until one is actually
    claimed (others may have been claimed by a concurrent worker between
    the query and the claim attempt — that race is expected and handled,
    not avoided). Returns None only when nothing is claimable."""
    pending = await store.query(StoreQuery(kinds=(ArtifactKind.TASK,), round_eq=round_))
    candidates = [
        t for t in pending
        if TaskContent.model_validate(t.content, strict=False).target_role == role
        and TaskContent.model_validate(t.content, strict=False).status == TaskStatus.PENDING
    ]
    for candidate in candidates:
        claimed = await _claim_task(store, candidate, agent_id)
        if claimed is None:
            continue  # lost the race for this one; try the next candidate
        task_content = TaskContent.model_validate(claimed.content, strict=False)
        agent = factory.build(role, agent_id, store)
        try:
            results = await agent.run(task_content, question)
        except Exception:
            logger.exception("stigmergy_agent_failed", extra={"role": role.value, "agent_id": agent_id, "task": candidate.key})
            await _finish_task(store, claimed, TaskStatus.FAILED, None)
            return []
        result_key = results[0].key if results else None
        await _finish_task(store, claimed, TaskStatus.DONE, result_key)
        return results
    return None


async def _drain_role_sequential(
    store: ArtifactStore, role: AgentRole, round_: int, factory: AgentFactory, question: str, agent_prefix: str
) -> list[ArtifactRecord]:
    all_results: list[ArtifactRecord] = []
    counter = 0
    while True:
        results = await _claim_and_run_one(store, role, round_, factory, question, f"{agent_prefix}-{counter}")
        counter += 1
        if results is None:
            return all_results
        all_results.extend(results)


async def _drain_role_concurrent(
    store: ArtifactStore,
    role: AgentRole,
    round_: int,
    factory: AgentFactory,
    question: str,
    agent_prefix: str,
    workers: int,
) -> list[ArtifactRecord]:
    lock = asyncio.Lock()
    collected: list[ArtifactRecord] = []

    async def worker(idx: int) -> None:
        counter = 0
        while True:
            results = await _claim_and_run_one(store, role, round_, factory, question, f"{agent_prefix}-{idx}-{counter}")
            counter += 1
            if results is None:
                return
            async with lock:
                collected.extend(results)

    await asyncio.gather(*(worker(i) for i in range(max(1, workers))))
    return collected


async def run_pipeline(
    store: ArtifactStore,
    question: str,
    factory: AgentFactory,
    *,
    rounds: int = 1,
    max_ticks_per_round: int = 200,
) -> Optional[ArtifactRecord]:
    """Linear, ant-like accumulation: one decision, drained fully, before
    the next decision. fanout=1 — exactly one task per tick."""
    for round_ in range(rounds):
        for _tick in range(max_ticks_per_round):
            decisions = await decide_next_tasks(store, question, round_, fanout=1)
            if not decisions:
                break
            role = TaskContent.model_validate(decisions[0].content, strict=False).target_role
            for req in decisions:
                await store.write(req)
            await _drain_role_sequential(store, role, round_, factory, question, f"pipeline-{role.value}")
        else:
            logger.warning("stigmergy_pipeline_max_ticks_reached", extra={"round": round_})
    return await store.read("report:final")


async def run_swarm(
    store: ArtifactStore,
    question: str,
    factory: AgentFactory,
    *,
    rounds: int = 1,
    fanout: int = 3,
    workers_per_role: int = 3,
    max_ticks_per_round: int = 200,
) -> Optional[ArtifactRecord]:
    """Concurrent accumulation: MetaController fans out `fanout` tasks per
    tick and `workers_per_role` agents of the dispatched role race to
    drain them via asyncio.gather. Their results simply accumulate in the
    store — no coordination beyond ArtifactStore.write's optimistic
    concurrency."""
    for round_ in range(rounds):
        for _tick in range(max_ticks_per_round):
            decisions = await decide_next_tasks(store, question, round_, fanout=fanout)
            if not decisions:
                break
            role = TaskContent.model_validate(decisions[0].content, strict=False).target_role
            for req in decisions:
                await store.write(req)
            await _drain_role_concurrent(store, role, round_, factory, question, f"swarm-{role.value}", workers_per_role)
        else:
            logger.warning("stigmergy_swarm_max_ticks_reached", extra={"round": round_})
    return await store.read("report:final")


async def run_event_driven(
    store: ArtifactStore,
    question: str,
    factory: AgentFactory,
    *,
    rounds: int = 1,
    fanout: int = 3,
    workers_per_role: int = 3,
    poll_interval_s: float = 0.02,
    idle_polls_before_stop: int = 3,
) -> Optional[ArtifactRecord]:
    """Instead of a caller explicitly stepping the pipeline, this watches
    the store's audit log length as its change signal: growth means an
    agent just wrote something, which is the cue to re-evaluate what's
    next. It stops a round once neither the log nor the coverage decision
    has changed for `idle_polls_before_stop` consecutive polls."""
    for round_ in range(rounds):
        idle = 0
        last_len = -1
        while idle < idle_polls_before_stop:
            log = await store.audit_log()
            changed = len(log) != last_len
            last_len = len(log)

            decisions = await decide_next_tasks(store, question, round_, fanout=fanout)
            if decisions:
                role = TaskContent.model_validate(decisions[0].content, strict=False).target_role
                for req in decisions:
                    await store.write(req)
                await _drain_role_concurrent(store, role, round_, factory, question, f"event-{role.value}", workers_per_role)
                idle = 0
                continue

            idle = 0 if changed else idle + 1
            if idle >= idle_polls_before_stop:
                break
            await asyncio.sleep(poll_interval_s)
    return await store.read("report:final")
