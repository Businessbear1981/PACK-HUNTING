"""
Agent roles. Every role is intentionally myopic: it gets a MyopicView, not
the raw ArtifactStore, and the view raises PermissionError if the agent asks
to read a kind its role isn't scoped to. That is the "no agent has the full
picture" constraint enforced in code, not just documented in a docstring.

No agent imports another agent. No agent calls another agent. The only
shared object is the store (via a MyopicView), which is the entire point.
"""
from __future__ import annotations

import abc
import logging
import re
import uuid
from collections import defaultdict
from typing import Optional

from contracts.stigmergy_contracts import (
    AgentRole,
    ArtifactKind,
    ArtifactRecord,
    StoreQuery,
    TaskContent,
    WriteRequest,
)
from orchestrators.stigmergy.providers import STOPWORDS, LLMProvider, SearchProvider
from orchestrators.stigmergy.store import ArtifactStore

logger = logging.getLogger(__name__)


class MyopicView:
    """A store handle scoped to exactly the kinds a role may see. This is
    the structural half of "agents are intentionally myopic" — the other
    half is that no role's implementation asks for more than it needs."""

    def __init__(self, store: ArtifactStore, allowed_kinds: frozenset[ArtifactKind]) -> None:
        self._store = store
        self._allowed = allowed_kinds

    async def query(self, q: StoreQuery) -> list[ArtifactRecord]:
        kinds = q.kinds if q.kinds is not None else tuple(self._allowed)
        disallowed = set(kinds) - self._allowed
        if disallowed:
            raise PermissionError(f"this role's view does not permit reading kinds={disallowed}")
        return await self._store.query(q.model_copy(update={"kinds": kinds}))

    async def read(self, key: str) -> Optional[ArtifactRecord]:
        record = await self._store.read(key)
        if record is not None and record.kind not in self._allowed:
            raise PermissionError(f"this role's view does not permit reading kind={record.kind}")
        return record

    async def write(self, request: WriteRequest, *, on_conflict: str = "reject"):
        return await self._store.write(request, on_conflict=on_conflict)  # type: ignore[arg-type]


ROLE_VIEWS: dict[AgentRole, frozenset[ArtifactKind]] = {
    AgentRole.SCOUT: frozenset({ArtifactKind.TASK}),
    AgentRole.ANALYST: frozenset({ArtifactKind.FINDING, ArtifactKind.TASK}),
    AgentRole.CRITIC: frozenset({ArtifactKind.INSIGHT, ArtifactKind.FINDING, ArtifactKind.TASK}),
    AgentRole.SYNTHESIZER: frozenset(
        {ArtifactKind.INSIGHT, ArtifactKind.CRITIQUE, ArtifactKind.FINDING, ArtifactKind.TASK}
    ),
    AgentRole.WRITER: frozenset(
        {ArtifactKind.OUTLINE, ArtifactKind.DRAFT_SECTION, ArtifactKind.CRITIQUE, ArtifactKind.TASK}
    ),
    AgentRole.META_CONTROLLER: frozenset(ArtifactKind),
}


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


class Agent(abc.ABC):
    role: AgentRole

    def __init__(self, store: ArtifactStore, agent_id: str) -> None:
        self.agent_id = agent_id
        self.view = MyopicView(store, ROLE_VIEWS[self.role])

    @abc.abstractmethod
    async def run(self, task: TaskContent, question: str) -> list[ArtifactRecord]: ...


class ScoutAgent(Agent):
    role = AgentRole.SCOUT

    def __init__(self, store: ArtifactStore, agent_id: str, search: SearchProvider) -> None:
        super().__init__(store, agent_id)
        self._search = search

    async def run(self, task: TaskContent, question: str) -> list[ArtifactRecord]:
        hits = await self._search.search(task.directive, limit=5)
        written: list[ArtifactRecord] = []
        for hit in hits:
            request = WriteRequest(
                key=f"finding:{task.round}:{_short_id()}",
                kind=ArtifactKind.FINDING,
                role=self.role,
                agent_id=self.agent_id,
                round=task.round,
                content={"query": task.directive, "title": hit.title, "snippet": hit.snippet, "source": hit.source},
                confidence=hit.score,
                rationale=f"deposited by search for {task.directive!r}; relevance score {hit.score:.3f} vs. source {hit.source}",
            )
            result = await self.view.write(request)
            if result.record is not None:
                written.append(result.record)
        return written


class AnalystAgent(Agent):
    role = AgentRole.ANALYST

    def __init__(self, store: ArtifactStore, agent_id: str, llm: LLMProvider) -> None:
        super().__init__(store, agent_id)
        self._llm = llm

    async def run(self, task: TaskContent, question: str) -> list[ArtifactRecord]:
        if task.input_keys:
            findings = []
            for k in task.input_keys:
                found = await self.view.read(k)
                if found is not None:
                    findings.append(found)
        else:
            findings = await self.view.query(StoreQuery(kinds=(ArtifactKind.FINDING,), round_lte=task.round))

        written: list[ArtifactRecord] = []
        for finding in findings:
            extraction = await self._llm.extract_insight(question, finding.content.get("snippet", ""), finding.content.get("source", finding.key))
            if not extraction.claims:
                continue
            request = WriteRequest(
                key=f"insight:{task.round}:{_short_id()}",
                kind=ArtifactKind.INSIGHT,
                role=self.role,
                agent_id=self.agent_id,
                round=task.round,
                content={"claims": list(extraction.claims), "source_finding": finding.key},
                confidence=extraction.confidence,
                rationale=f"extracted via {extraction.method} from {finding.key}",
                parent_versions=(finding.versioned_key,),
            )
            result = await self.view.write(request)
            if result.record is not None:
                written.append(result.record)
        return written


class CriticAgent(Agent):
    role = AgentRole.CRITIC

    def __init__(self, store: ArtifactStore, agent_id: str, llm: LLMProvider) -> None:
        super().__init__(store, agent_id)
        self._llm = llm

    async def run(self, task: TaskContent, question: str) -> list[ArtifactRecord]:
        if task.input_keys:
            insights = []
            for k in task.input_keys:
                found = await self.view.read(k)
                if found is not None:
                    insights.append(found)
        else:
            insights = await self.view.query(StoreQuery(kinds=(ArtifactKind.INSIGHT,), round_lte=task.round))

        written: list[ArtifactRecord] = []
        for insight in insights:
            source_key = insight.content.get("source_finding")
            supporting_text = ""
            if source_key:
                finding = await self.view.read(source_key)
                if finding is not None:
                    supporting_text = finding.content.get("snippet", "")
            claim_text = " ".join(insight.content.get("claims", []))
            critique = await self._llm.critique(question, claim_text, supporting_text)
            request = WriteRequest(
                key=f"critique:{task.round}:{_short_id()}",
                kind=ArtifactKind.CRITIQUE,
                role=self.role,
                agent_id=self.agent_id,
                round=task.round,
                content={"issues": list(critique.issues), "critiqued_insight": insight.key},
                confidence=critique.quality_score,
                rationale=f"assessed via {critique.method} against {source_key or 'no cited source'}",
                parent_versions=(insight.versioned_key,),
            )
            result = await self.view.write(request)
            if result.record is not None:
                written.append(result.record)
        return written


def _heading_for(claim_text: str) -> str:
    tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", claim_text.lower()) if t not in STOPWORDS]
    return tokens[0].capitalize() if tokens else "General"


class SynthesizerAgent(Agent):
    role = AgentRole.SYNTHESIZER

    def __init__(self, store: ArtifactStore, agent_id: str, llm: LLMProvider) -> None:
        super().__init__(store, agent_id)
        self._llm = llm

    async def run(self, task: TaskContent, question: str) -> list[ArtifactRecord]:
        insights = await self.view.query(StoreQuery(kinds=(ArtifactKind.INSIGHT,), round_lte=task.round))
        critiques = await self.view.query(StoreQuery(kinds=(ArtifactKind.CRITIQUE,), round_lte=task.round))
        quality_by_insight: dict[str, list[float]] = defaultdict(list)
        for critique in critiques:
            target = critique.content.get("critiqued_insight")
            if target and critique.confidence is not None:
                quality_by_insight[target].append(critique.confidence)

        groups: dict[str, list[ArtifactRecord]] = defaultdict(list)
        for insight in insights:
            for claim in insight.content.get("claims", []):
                groups[_heading_for(claim)].append(insight)

        headings = []
        for heading, members in groups.items():
            weights = []
            for m in members:
                quality_scores = quality_by_insight.get(m.key, [])
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
                weights.append((m.confidence or 0.0) * avg_quality)
            headings.append({"heading": heading, "insight_keys": [m.key for m in members], "weight": round(sum(weights), 4)})
        headings.sort(key=lambda h: h["weight"], reverse=True)

        written: list[ArtifactRecord] = []
        outline_request = WriteRequest(
            key=f"outline:{task.round}",
            kind=ArtifactKind.OUTLINE,
            role=self.role,
            agent_id=self.agent_id,
            round=task.round,
            content={"headings": headings},
            rationale=f"grouped {len(insights)} insight(s) into {len(headings)} heading(s) weighted by confidence x critique quality",
            parent_versions=tuple(i.versioned_key for i in insights),
        )
        outline_result = await self.view.write(outline_request)
        if outline_result.record is not None:
            written.append(outline_result.record)

        insight_by_key = {i.key: i for i in insights}
        for h in headings:
            members = [insight_by_key[k] for k in h["insight_keys"]]
            bullets = tuple(claim for m in members for claim in m.content.get("claims", []))
            draft = await self._llm.draft_section(h["heading"], question, bullets)
            draft_request = WriteRequest(
                key=f"draft_section:{task.round}:{_short_id()}",
                kind=ArtifactKind.DRAFT_SECTION,
                role=self.role,
                agent_id=self.agent_id,
                round=task.round,
                content={"heading": h["heading"], "text": draft.text},
                rationale=f"drafted via {draft.method} from {len(members)} insight(s) under heading {h['heading']!r}",
                parent_versions=tuple(m.versioned_key for m in members),
            )
            draft_result = await self.view.write(draft_request)
            if draft_result.record is not None:
                written.append(draft_result.record)
        return written


class WriterAgent(Agent):
    role = AgentRole.WRITER

    def __init__(self, store: ArtifactStore, agent_id: str, llm: LLMProvider) -> None:
        super().__init__(store, agent_id)
        self._llm = llm

    async def run(self, task: TaskContent, question: str) -> list[ArtifactRecord]:
        outline = await self.view.read(f"outline:{task.round}")
        drafts = await self.view.query(StoreQuery(kinds=(ArtifactKind.DRAFT_SECTION,), round_lte=task.round))
        critiques = await self.view.query(StoreQuery(kinds=(ArtifactKind.CRITIQUE,), round_lte=task.round))

        draft_by_heading = {d.content["heading"]: d for d in drafts}
        ordered_headings = [h["heading"] for h in (outline.content.get("headings", []) if outline else [])]
        sections = tuple((h, draft_by_heading[h].content["text"]) for h in ordered_headings if h in draft_by_heading)

        limitations: list[str] = []
        seen: set[str] = set()
        for critique in critiques:
            for issue in critique.content.get("issues", []):
                if issue not in seen:
                    seen.add(issue)
                    limitations.append(issue)

        final = await self._llm.polish_report(question, sections, tuple(limitations))
        parents = tuple(d.versioned_key for d in drafts) + ((outline.versioned_key,) if outline else ())
        request = WriteRequest(
            key="report:final",
            kind=ArtifactKind.REPORT,
            role=self.role,
            agent_id=self.agent_id,
            round=task.round,
            content={"text": final.text, "section_count": len(sections), "limitation_count": len(limitations)},
            rationale=f"assembled via {final.method} from {len(sections)} section(s) and {len(limitations)} logged limitation(s)",
            parent_versions=parents,
        )
        result = await self.view.write(request)
        return [result.record] if result.record is not None else []
