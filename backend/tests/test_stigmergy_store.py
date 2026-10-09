"""
ArtifactStore concurrency tests — the deliverable the whole system stands
or falls on: concurrent agents writing without corrupting shared state.

No pytest-asyncio in this repo's dependencies, so async bodies are driven
with asyncio.run() inside plain `def test_...():` functions, matching how
the rest of this backend runs async code under a sync test runner.
"""
from __future__ import annotations

import asyncio
import pathlib

import pytest
from pydantic import ValidationError

from contracts.stigmergy_contracts import (
    AgentRole,
    ArtifactKind,
    ArtifactRecord,
    WriteOutcome,
    WriteRequest,
)
from orchestrators.stigmergy.store import FileArtifactStore, InMemoryArtifactStore


def _req(key: str, *, content=None, expected_version=None, agent_id="agent-1") -> WriteRequest:
    return WriteRequest(
        key=key,
        kind=ArtifactKind.FINDING,
        role=AgentRole.SCOUT,
        agent_id=agent_id,
        content=content or {"snippet": "test"},
        rationale="test write",
        expected_version=expected_version,
    )


# --------------------------------------------------------------- contracts


def test_rationale_is_mandatory():
    with pytest.raises(ValidationError):
        ArtifactRecord(
            key="k",
            kind=ArtifactKind.FINDING,
            version=1,
            round=0,
            role=AgentRole.SCOUT,
            agent_id="a",
            content={},
            rationale="   ",
        )


def test_artifact_record_is_frozen():
    record = ArtifactRecord(
        key="k", kind=ArtifactKind.FINDING, version=1, round=0, role=AgentRole.SCOUT,
        agent_id="a", content={}, rationale="ok",
    )
    with pytest.raises(ValidationError):
        record.version = 2


# ------------------------------------------------------------- in-memory


def test_optimistic_conflict_is_detected_not_silently_overwritten():
    store = InMemoryArtifactStore()

    async def scenario():
        first = await store.write(_req("k1", expected_version=None))
        assert first.outcome == WriteOutcome.OK
        assert first.record.version == 1

        stale = await store.write(_req("k1", expected_version=0, agent_id="agent-2"), on_conflict="deposit")
        assert stale.outcome == WriteOutcome.CONFLICT
        assert stale.conflict.expected_version == 0
        assert stale.conflict.actual_version == 1

        head = await store.read("k1")
        assert head.version == 1
        assert head.agent_id == "agent-1"  # the loser never touched the value

        conflicts = await store.conflicts(unresolved_only=True)
        assert len(conflicts) == 1

    asyncio.run(scenario())


def test_write_with_retry_reconciles_a_lost_race():
    store = InMemoryArtifactStore()

    async def scenario():
        await store.write(_req("k1"))  # version 1 exists before either retrier starts

        def build(current):
            version = current.version if current is not None else 0
            return _req("k1", expected_version=version, agent_id="retrier")

        result = await store.write_with_retry("k1", build)
        assert result.outcome == WriteOutcome.OK
        assert result.record.version == 2  # succeeded against the *current* head, not a stale read

    asyncio.run(scenario())


def test_concurrent_writes_to_distinct_keys_all_land_without_corruption():
    store = InMemoryArtifactStore()
    n = 50

    async def scenario():
        await asyncio.gather(*(store.write(_req(f"k{i}", content={"i": i})) for i in range(n)))
        audit = await store.audit_log()
        assert len(audit) == n
        for i in range(n):
            record = await store.read(f"k{i}")
            assert record is not None
            assert record.content == {"i": i}  # no cross-writer contamination

    asyncio.run(scenario())


def test_concurrent_optimistic_writes_to_same_key_exactly_one_wins():
    """N agents all believe the key doesn't exist yet (expected_version=0)
    and race to create it. This is the swarm's actual failure mode if the
    store were unsafe: without a real critical section, more than one
    would "win" and the version history would fork or duplicate."""
    store = InMemoryArtifactStore()
    n = 20

    async def scenario():
        results = await asyncio.gather(
            *(store.write(_req("shared", expected_version=0, agent_id=f"agent-{i}"), on_conflict="deposit") for i in range(n))
        )
        oks = [r for r in results if r.outcome == WriteOutcome.OK]
        conflicts = [r for r in results if r.outcome == WriteOutcome.CONFLICT]
        assert len(oks) == 1
        assert len(conflicts) == n - 1

        history = await store.read_all_versions("shared")
        assert len(history) == 1  # exactly one version was ever created, not n

        stored_conflicts = await store.conflicts(unresolved_only=True)
        assert len(stored_conflicts) == n - 1

    asyncio.run(scenario())


def test_append_only_history_preserves_every_proposal():
    """The event-sourced half of the required concurrency model: writers
    that don't pass expected_version never overwrite, they accumulate."""
    store = InMemoryArtifactStore()

    async def scenario():
        for i in range(5):
            await store.write(_req("append-key", content={"proposal": i}, agent_id=f"agent-{i}"))
        history = await store.read_all_versions("append-key")
        assert [h.version for h in history] == [1, 2, 3, 4, 5]
        assert [h.content["proposal"] for h in history] == [0, 1, 2, 3, 4]
        head = await store.read("append-key")
        assert head.version == 5

    asyncio.run(scenario())


# ------------------------------------------------------------------ file


def test_file_store_survives_reload_and_concurrent_writes(tmp_path: pathlib.Path):
    store = FileArtifactStore(namespace="test_ns", directory=tmp_path)
    n = 30

    async def write_all():
        await asyncio.gather(*(store.write(_req(f"k{i}", content={"i": i})) for i in range(n)))

    asyncio.run(write_all())

    status = store.status()
    assert status["durable"] is True

    # Fresh instance, same directory: durability, not just in-process caching.
    reloaded = FileArtifactStore(namespace="test_ns", directory=tmp_path)

    async def verify():
        audit = await reloaded.audit_log()
        assert len(audit) == n
        for i in range(n):
            record = await reloaded.read(f"k{i}")
            assert record is not None
            assert record.content == {"i": i}

    asyncio.run(verify())


def test_file_store_optimistic_conflict_matches_in_memory_semantics(tmp_path: pathlib.Path):
    store = FileArtifactStore(namespace="test_ns2", directory=tmp_path)

    async def scenario():
        first = await store.write(_req("k1"))
        assert first.outcome == WriteOutcome.OK

        stale = await store.write(_req("k1", expected_version=0), on_conflict="deposit")
        assert stale.outcome == WriteOutcome.CONFLICT

        conflicts = await store.conflicts(unresolved_only=True)
        assert len(conflicts) == 1

    asyncio.run(scenario())
