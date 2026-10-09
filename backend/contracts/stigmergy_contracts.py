"""
Stigmergy contracts — the shared vocabulary between the artifact store and
every agent role in the multi-agent research system.

WHY A SEPARATE MODULE
    Per AGENTS.md/CLAUDE.md convention: contracts between layers are Pydantic
    schemas in a shared contracts/ package, versioned, never mutated in place.
    Agents never import each other; they only import these types plus the
    store interface. That is what makes "no agent-to-agent messaging" true in
    code, not just in the docstring.

VERSIONING
    CONTRACT_VERSION bumps on any breaking change to ArtifactRecord's shape.
    A store can refuse to load artifacts written by an incompatible version
    instead of silently misinterpreting their fields.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

CONTRACT_VERSION = "1.0.0"


class ArtifactKind(str, Enum):
    """What a record in the shared store represents."""

    FINDING = "finding"              # Scout: raw source discovered
    INSIGHT = "insight"              # Analyst: extracted claim + confidence
    CRITIQUE = "critique"            # Critic: challenge + quality score
    OUTLINE = "outline"              # Synthesizer: report structure
    DRAFT_SECTION = "draft_section"  # Synthesizer: prose for one outline node
    REPORT = "report"                # Writer: final polished output
    CONFLICT = "conflict"            # deposited on an optimistic-write mismatch
    TASK = "task"                    # MetaController: a pending unit of work


class AgentRole(str, Enum):
    SCOUT = "scout"
    ANALYST = "analyst"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    WRITER = "writer"
    META_CONTROLLER = "meta_controller"


class TaskStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"


class WriteOutcome(str, Enum):
    OK = "ok"
    CONFLICT = "conflict"
    DEGRADED = "degraded"  # store's durable backend is unavailable; held in memory


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactRecord(BaseModel):
    """
    One immutable version of one artifact.

    Artifacts are append-only: a "write" never overwrites a prior version, it
    appends a new one under the same key. The current value of a key is its
    highest version. This is what makes the store simultaneously satisfy the
    two required concurrency patterns: optimistic-concurrency callers pass
    `expected_version` and get WriteOutcome.CONFLICT on a stale read;
    append-only callers pass no `expected_version` and simply add a new
    proposal for a later Synthesizer/Merger to reconcile.

    Every field the spec calls mandatory for audit is on the record itself,
    not bolted on to a side log: agent_id, timestamp, confidence,
    parent_version(s), rationale.
    """

    contract_version: str = CONTRACT_VERSION
    key: str                       # logical artifact id, e.g. "insight:round1:a1b2"
    kind: ArtifactKind
    version: int = Field(ge=1)     # 1-indexed, monotonic per key
    round: int = Field(ge=0)       # research round this artifact belongs to
    role: AgentRole
    agent_id: str                  # e.g. "scout-3" — which worker instance wrote this
    content: dict[str, Any]        # kind-specific payload
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rationale: str                 # short "why" — mandatory, never blank
    parent_versions: tuple[str, ...] = Field(default_factory=tuple)  # "key@version"
    written_at: datetime = Field(default_factory=_utcnow)

    model_config = ConfigDict(strict=True, frozen=True)

    @field_validator("rationale")
    @classmethod
    def _rationale_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("rationale is mandatory and cannot be blank")
        return v

    @property
    def versioned_key(self) -> str:
        return f"{self.key}@{self.version}"


class ConflictRecord(BaseModel):
    """Deposited when an optimistic write loses a race, instead of a silent
    last-write-wins. A Critic or MetaController resolves these explicitly."""

    key: str
    attempted_by: str
    attempted_role: AgentRole
    expected_version: int
    actual_version: int
    attempted_content: dict[str, Any]
    detected_at: datetime = Field(default_factory=_utcnow)
    resolved: bool = False
    resolution_note: Optional[str] = None

    model_config = ConfigDict(strict=True)


class WriteRequest(BaseModel):
    """Input to ArtifactStore.write — not stored, just the call shape."""

    key: str
    kind: ArtifactKind
    role: AgentRole
    agent_id: str
    content: dict[str, Any]
    rationale: str
    round: int = 0
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    parent_versions: tuple[str, ...] = Field(default_factory=tuple)
    expected_version: Optional[int] = None  # None => append unconditionally

    model_config = ConfigDict(strict=True, frozen=True)


class WriteResult(BaseModel):
    outcome: WriteOutcome
    record: Optional[ArtifactRecord] = None
    conflict: Optional[ConflictRecord] = None
    note: Optional[str] = None

    model_config = ConfigDict(strict=True, frozen=True)


class TaskContent(BaseModel):
    """Payload carried by a TASK artifact — the MetaController's dispatch
    unit. Tasks are deposited into the store, never handed to an agent
    directly; a worker loop claims them by optimistic write."""

    directive: str
    target_role: AgentRole
    round: int
    status: TaskStatus = TaskStatus.PENDING
    input_keys: tuple[str, ...] = Field(default_factory=tuple)
    result_key: Optional[str] = None
    claimed_by: Optional[str] = None

    model_config = ConfigDict(strict=True)


class StoreQuery(BaseModel):
    """Filter shape for ArtifactStore.query — how a role's myopic view is
    expressed without giving the role direct access to the whole store."""

    kinds: Optional[tuple[ArtifactKind, ...]] = None
    roles: Optional[tuple[AgentRole, ...]] = None
    round_eq: Optional[int] = None
    round_lte: Optional[int] = None
    key_prefix: Optional[str] = None
    only_latest: bool = True

    model_config = ConfigDict(strict=True, frozen=True)
