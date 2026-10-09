"""
Orchestrator-level tests: pipeline and swarm both produce a real report
end to end, the task-claim race actually resolves to exactly one winner
under real concurrency, myopic views are structurally enforced, and the
meta-controller halts instead of retrying a dead stage forever.

Uses HeuristicLLMProvider (no network, no API key) and a tiny fixture
corpus written to tmp_path — these are the "explicit, named fixtures"
CLAUDE.md's no-mocks rule carves out room for; nothing here fabricates an
agent's output, it just avoids a live network dependency in the test suite.
"""
from __future__ import annotations

import asyncio
import pathlib

import pytest

from contracts.stigmergy_contracts import (
    AgentRole,
    ArtifactKind,
    StoreQuery,
    TaskContent,
    TaskStatus,
    WriteRequest,
)
from orchestrators.stigmergy import meta_controller
from orchestrators.stigmergy.agents import MyopicView
from orchestrators.stigmergy.meta_controller import decide_next_tasks
from orchestrators.stigmergy.orchestrator import AgentFactory, _claim_task, run_pipeline, run_swarm
from orchestrators.stigmergy.providers import HeuristicLLMProvider, LocalCorpusSearchProvider
from orchestrators.stigmergy.store import InMemoryArtifactStore

QUESTION = "How should coffee be brewed for maximum caffeine extraction?"


@pytest.fixture()
def tiny_corpus(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "brewing.md").write_text(
        "# Brewing Notes\n\n"
        "Hotter water extracts caffeine faster than lukewarm water during brewing. "
        "A finer coffee grind also increases the surface area available for extraction, "
        "which pulls more caffeine out of the same amount of beans.\n\n"
        "# Water Temperature\n\n"
        "Water between 195 and 205 degrees Fahrenheit is considered optimal for drip brewing. "
        "Below that range, extraction slows down noticeably and the resulting cup tastes weak.\n",
        encoding="utf-8",
    )
    (tmp_path / "beans.md").write_text(
        "# Bean Origin\n\n"
        "Robusta beans contain roughly twice the caffeine of Arabica beans on average. "
        "Roast level has a much smaller effect on caffeine content than bean species does.\n",
        encoding="utf-8",
    )
    return tmp_path


def _factory(tiny_corpus: pathlib.Path) -> AgentFactory:
    return AgentFactory(LocalCorpusSearchProvider(tiny_corpus), HeuristicLLMProvider())


def test_pipeline_produces_a_grounded_report(tiny_corpus: pathlib.Path):
    store = InMemoryArtifactStore()
    factory = _factory(tiny_corpus)

    report = asyncio.run(run_pipeline(store, QUESTION, factory, rounds=1))

    assert report is not None
    assert report.kind == ArtifactKind.REPORT
    assert "caffeine" in report.content["text"].lower()

    audit = asyncio.run(store.audit_log())
    kinds_seen = {r.kind for r in audit}
    assert kinds_seen >= {
        ArtifactKind.FINDING, ArtifactKind.INSIGHT, ArtifactKind.CRITIQUE,
        ArtifactKind.OUTLINE, ArtifactKind.DRAFT_SECTION, ArtifactKind.REPORT,
    }
    # Every insight traces back to a finding it cited, and every critique to
    # the insight it challenged — the audit trail the spec requires.
    insights = [r for r in audit if r.kind == ArtifactKind.INSIGHT]
    assert all(r.parent_versions for r in insights)
    assert all(r.rationale.strip() for r in audit)  # rationale is never blank, anywhere


def test_swarm_produces_a_report_via_concurrent_agents(tiny_corpus: pathlib.Path):
    store = InMemoryArtifactStore()
    factory = _factory(tiny_corpus)

    report = asyncio.run(run_swarm(store, QUESTION, factory, rounds=1, fanout=3, workers_per_role=4))

    assert report is not None
    assert report.kind == ArtifactKind.REPORT

    audit = asyncio.run(store.audit_log())
    # Multiple scouts really did run concurrently: more than one distinct
    # agent_id shows up among the finding writers.
    scout_ids = {r.agent_id for r in audit if r.kind == ArtifactKind.FINDING}
    assert len(scout_ids) >= 2


def test_concurrent_claim_of_the_same_task_has_exactly_one_winner():
    """Directly exercises the race the swarm depends on being safe: many
    workers see the same PENDING task and try to claim it at once."""
    store = InMemoryArtifactStore()

    async def scenario():
        content = TaskContent(directive="do the thing", target_role=AgentRole.SCOUT, round=0)
        write_result = await store.write(WriteRequest(
            key="task:0:scout:x", kind=ArtifactKind.TASK, role=AgentRole.META_CONTROLLER,
            agent_id="meta", round=0, content=content.model_dump(mode="json"), rationale="seed",
        ))
        task_record = write_result.record

        claims = await asyncio.gather(*(_claim_task(store, task_record, f"worker-{i}") for i in range(15)))
        winners = [c for c in claims if c is not None]
        assert len(winners) == 1

        history = await store.read_all_versions("task:0:scout:x")
        assert len(history) == 2  # the original PENDING write + exactly one CLAIMED write
        assert TaskContent.model_validate(history[-1].content, strict=False).status == TaskStatus.CLAIMED

    asyncio.run(scenario())


def test_myopic_view_blocks_reading_a_kind_outside_its_role():
    store = InMemoryArtifactStore()
    scout_view = MyopicView(store, frozenset({ArtifactKind.TASK}))

    async def scenario():
        with pytest.raises(PermissionError):
            await scout_view.query(StoreQuery(kinds=(ArtifactKind.INSIGHT,)))

    asyncio.run(scenario())


def test_meta_controller_halts_after_repeated_failures_instead_of_looping_forever():
    """Regression test for the exact bug this system hit against a broken
    API key: redispatching an identical failing task forever. After
    MAX_TASK_ATTEMPTS failed attempts on the same gap, decide_next_tasks
    must return [] rather than dispatch the same request again."""
    store = InMemoryArtifactStore()

    async def scenario():
        # One finding exists but is never analyzed, and the analyst task
        # for it has already failed twice — coverage's real state after a
        # persistently broken LLM provider.
        await store.write(WriteRequest(
            key="finding:0:abc", kind=ArtifactKind.FINDING, role=AgentRole.SCOUT,
            agent_id="scout-0", round=0, content={"snippet": "x", "source": "y"}, rationale="seed",
        ))
        directive = "analyze 1 finding(s)"
        task_key = meta_controller._task_request(0, AgentRole.ANALYST, directive, ("finding:0:abc",)).key
        content = TaskContent(
            directive=directive, target_role=AgentRole.ANALYST, round=0,
            input_keys=("finding:0:abc",),
        )
        for status in (TaskStatus.PENDING, TaskStatus.CLAIMED, TaskStatus.FAILED,
                       TaskStatus.PENDING, TaskStatus.CLAIMED, TaskStatus.FAILED):
            updated = content.model_copy(update={"status": status})
            await store.write(WriteRequest(
                key=task_key, kind=ArtifactKind.TASK, role=AgentRole.META_CONTROLLER,
                agent_id="worker", round=0, content=updated.model_dump(mode="json"), rationale="progression",
            ))

        decisions = await decide_next_tasks(store, QUESTION, 0, fanout=1)
        assert decisions == []

    asyncio.run(scenario())
