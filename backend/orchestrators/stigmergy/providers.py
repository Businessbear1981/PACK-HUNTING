"""
Providers — the only place this system touches the outside world. Agents
never call these directly; each role wraps exactly the provider its job
needs (Scout -> SearchProvider, Analyst/Critic/Synthesizer/Writer ->
LLMProvider), and every method here returns real, derived output — never a
placeholder.

WHY NO REAL WEB SEARCH
    This environment has no web-search API wired in (no Tavily/Serper/Brave
    key, no browser). Rather than fabricate search results, LocalCorpusSearchProvider
    does real term-overlap search over real local text files. It is a
    legitimate default for "point this at a document corpus and research
    it," and the SearchProvider ABC is the swap-in seam for a real web
    search backend later — implement `search()` against a live API and
    nothing above this layer changes.

WHY A HEURISTIC LLM FALLBACK
    ANTHROPIC_API_KEY (checked via backend/agents/_claude.py, which also
    tries OPENROUTER_API_KEY) may be unset in a given environment. Rather
    than halting the whole pipeline or fabricating analysis text,
    HeuristicLLMProvider does real, deterministic, honestly-labeled
    computation: keyword-overlap extraction and grounding checks. It is
    weaker than an LLM and its output says so (`method` field on every
    result). This is a fallback implementation, not a stub — it produces
    genuine output from genuine input every time it runs.
"""
from __future__ import annotations

import abc
import asyncio
import json
import logging
import pathlib
import re

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}")

# Excluded from every overlap score below. Without this, a generic heading
# like "What shipped today, and why you'd care" scores artificially high
# against any question, purely because "what"/"why" appear in most
# questions — the ranking signal was noise, not relevance.
STOPWORDS = frozenset(
    "the a an of to in on for and or is are was were be by with as at from "
    "this that it its into their his her over under about not no what why "
    "how when where who which does do did can could should would will".split()
)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in STOPWORDS]


_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_BOX_DRAWING_CHARS = "│├└─┌┐┘┬┴┼╭╮╰╯▪▸►"
_FUNCTION_CALL_RE = re.compile(r"\w+\([^)]*\)\s*;?\s*$")  # catches indented code blocks a fence-stripper misses


def _prose_only(text: str) -> str:
    """Drop markdown headings/list markers/table rows/fenced code blocks/
    directory trees/inline-code lines, keeping only actual sentence prose.
    A heading, a JSON literal, or a `tree`-style diagram line is a label or
    data, not evidence — citing one as a claim's source would be citing a
    table of contents. Fenced code blocks are stripped as whole regions
    first because a line *inside* one (e.g. a JSON field) doesn't start
    with any of the per-line markers below."""
    text = _CODE_FENCE_RE.sub(" ", text)
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "|", "-", "*", "```", ">")):
            continue
        if any(ch in _BOX_DRAWING_CHARS for ch in line):
            continue
        if _FUNCTION_CALL_RE.search(line):
            continue
        letters = sum(ch.isalpha() for ch in line)
        if letters < 0.5 * len(line):
            continue  # symbol-heavy: code, JSON, a table row without leading '|'
        lines.append(line)
    return "\n".join(lines)


def _overlap_score(a_tokens: list[str], b_tokens: list[str]) -> float:
    """Jaccard-style overlap over token sets — a small, real, defensible
    heuristic. Not a calibrated probability; callers must label it as such."""
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------- search


class SearchHit(BaseModel):
    title: str
    snippet: str
    source: str
    score: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(strict=True, frozen=True)


class SearchProvider(abc.ABC):
    @abc.abstractmethod
    async def search(self, query: str, *, limit: int = 5) -> list[SearchHit]: ...


class LocalCorpusSearchProvider(SearchProvider):
    """Real term-overlap search over markdown/text files under `root`."""

    def __init__(self, root: pathlib.Path, patterns: tuple[str, ...] = ("*.md", "*.txt")) -> None:
        self._root = pathlib.Path(root)
        self._paragraphs: list[tuple[str, str, list[str]]] = []  # (source, text, tokens)
        self._index(patterns)

    def _index(self, patterns: tuple[str, ...]) -> None:
        for pattern in patterns:
            for path in sorted(self._root.rglob(pattern)):
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for para in re.split(r"\n\s*\n", text):
                    para = para.strip()
                    if len(para) < 40 or len(_prose_only(para)) < 40:
                        continue  # too short, or nothing but headings/tables/lists — no citable evidence
                    rel = str(path.relative_to(self._root))
                    self._paragraphs.append((rel, para, _tokenize(para)))
        logger.info("stigmergy_corpus_indexed", extra={"root": str(self._root), "paragraphs": len(self._paragraphs)})

    async def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        q_tokens = _tokenize(query)
        scored = [
            (source, text, _overlap_score(q_tokens, tokens))
            for source, text, tokens in self._paragraphs
        ]
        scored = [s for s in scored if s[2] > 0]
        scored.sort(key=lambda s: s[2], reverse=True)
        top = scored[:limit]
        return [
            SearchHit(
                title=f"{source} (local corpus)",
                snippet=text[:600],
                source=source,
                score=round(score, 4),
            )
            for source, text, score in top
        ]


# ------------------------------------------------------------------------------ LLM


class ClaimExtraction(BaseModel):
    claims: tuple[str, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    method: str

    model_config = ConfigDict(strict=True, frozen=True)


class CritiqueResult(BaseModel):
    quality_score: float = Field(ge=0.0, le=1.0)
    issues: tuple[str, ...]
    method: str

    model_config = ConfigDict(strict=True, frozen=True)


class DraftResult(BaseModel):
    text: str
    method: str

    model_config = ConfigDict(strict=True, frozen=True)


class LLMUnavailable(RuntimeError):
    pass


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    async def extract_insight(self, question: str, finding_text: str, source: str) -> ClaimExtraction: ...

    @abc.abstractmethod
    async def critique(self, question: str, insight_text: str, supporting_text: str) -> CritiqueResult: ...

    @abc.abstractmethod
    async def draft_section(self, heading: str, question: str, bullets: tuple[str, ...]) -> DraftResult: ...

    @abc.abstractmethod
    async def polish_report(
        self, question: str, sections: tuple[tuple[str, str], ...], limitations: tuple[str, ...]
    ) -> DraftResult: ...


class HeuristicLLMProvider(LLMProvider):
    """Deterministic, dependency-free fallback. No API key required."""

    def name(self) -> str:
        return "heuristic:keyword_overlap"

    async def extract_insight(self, question: str, finding_text: str, source: str) -> ClaimExtraction:
        q_tokens = _tokenize(question)
        prose = _prose_only(finding_text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prose) if len(s.strip()) > 15]
        scored = sorted(
            ((s, _overlap_score(q_tokens, _tokenize(s))) for s in sentences),
            key=lambda x: x[1],
            reverse=True,
        )
        best = [s for s, score in scored[:2] if score > 0] or (sentences[:1] if sentences else [finding_text[:280]])
        top_score = scored[0][1] if scored else 0.0
        confidence = min(0.95, max(0.05, top_score))
        return ClaimExtraction(claims=tuple(best), confidence=confidence, method=self.name())

    async def critique(self, question: str, insight_text: str, supporting_text: str) -> CritiqueResult:
        grounding = _overlap_score(_tokenize(insight_text), _tokenize(supporting_text))
        issues: list[str] = []
        if grounding < 0.05:
            issues.append("insight shares almost no vocabulary with its cited source — weak grounding")
        if len(insight_text.split()) < 4:
            issues.append("insight is too short to carry a checkable claim")
        quality = max(0.0, min(1.0, grounding * 2))
        return CritiqueResult(quality_score=quality, issues=tuple(issues), method=self.name())

    async def draft_section(self, heading: str, question: str, bullets: tuple[str, ...]) -> DraftResult:
        body = "\n".join(f"- {b}" for b in bullets) or "- (no supporting insights survived critique for this section)"
        return DraftResult(text=f"## {heading}\n\n{body}\n", method=self.name())

    async def polish_report(
        self, question: str, sections: tuple[tuple[str, str], ...], limitations: tuple[str, ...]
    ) -> DraftResult:
        parts = [f"# Research Report: {question}\n"]
        parts.extend(text for _heading, text in sections)
        if limitations:
            parts.append("## Limitations & Open Critiques\n")
            parts.extend(f"- {item}" for item in limitations)
        return DraftResult(text="\n".join(parts), method=self.name())


class AnthropicLLMProvider(LLMProvider):
    """Wraps the existing backend/agents/_claude.py client (Anthropic direct,
    or OpenRouter fallback — same priority order, same failure mode)."""

    def __init__(self) -> None:
        from agents._claude import complete  # local import: keeps Flask app-context import cost out of module load

        self._complete = complete

    def name(self) -> str:
        return "llm:anthropic_or_openrouter"

    async def _call(self, system_prompt: str, user_prompt: str) -> str:
        return await asyncio.to_thread(self._complete, system_prompt, user_prompt)

    async def extract_insight(self, question: str, finding_text: str, source: str) -> ClaimExtraction:
        system = (
            "You extract factual claims from a source snippet relevant to a research "
            "question. Respond ONLY with JSON: "
            '{"claims": ["..."], "confidence": 0.0-1.0}. '
            "confidence reflects how directly the source supports the claims, not how "
            "important they are. If the snippet says nothing relevant, return an empty "
            "claims list and confidence 0.0."
        )
        user = f"Research question: {question}\n\nSource ({source}):\n{finding_text}"
        raw = await self._call(system, user)
        try:
            parsed = json.loads(raw)
            claims = tuple(str(c) for c in parsed.get("claims", []))
            confidence = float(parsed.get("confidence", 0.5))
            return ClaimExtraction(claims=claims, confidence=min(1.0, max(0.0, confidence)), method=self.name())
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("stigmergy_llm_unparsed_response", extra={"stage": "extract_insight"})
            return ClaimExtraction(claims=(raw.strip()[:400],), confidence=0.5, method="llm:unparsed")

    async def critique(self, question: str, insight_text: str, supporting_text: str) -> CritiqueResult:
        system = (
            "You are a skeptical critic. Given a claim and the source text it is "
            "supposedly grounded in, respond ONLY with JSON: "
            '{"quality_score": 0.0-1.0, "issues": ["..."]}. '
            "quality_score reflects how well the source actually supports the claim. "
            "List concrete issues (unsupported leaps, missing context, overreach); "
            "empty list if the claim is well-grounded."
        )
        user = f"Research question: {question}\n\nClaim:\n{insight_text}\n\nSource text:\n{supporting_text}"
        raw = await self._call(system, user)
        try:
            parsed = json.loads(raw)
            score = float(parsed.get("quality_score", 0.5))
            issues = tuple(str(i) for i in parsed.get("issues", []))
            return CritiqueResult(quality_score=min(1.0, max(0.0, score)), issues=issues, method=self.name())
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("stigmergy_llm_unparsed_response", extra={"stage": "critique"})
            return CritiqueResult(quality_score=0.5, issues=("critic response was not parseable JSON",), method="llm:unparsed")

    async def draft_section(self, heading: str, question: str, bullets: tuple[str, ...]) -> DraftResult:
        system = (
            "You write one section of a cited research report. Write 2-4 tight "
            "sentences of prose synthesizing the given points. Do not invent facts "
            "beyond what the points state. Plain text, no markdown heading."
        )
        user = f"Research question: {question}\nSection: {heading}\nPoints:\n" + "\n".join(f"- {b}" for b in bullets)
        text = await self._call(system, user)
        return DraftResult(text=f"## {heading}\n\n{text}\n", method=self.name())

    async def polish_report(
        self, question: str, sections: tuple[tuple[str, str], ...], limitations: tuple[str, ...]
    ) -> DraftResult:
        system = (
            "You are the final editor of a multi-section research report. Improve "
            "flow and remove redundancy between sections. Preserve every citation "
            "marker and every factual claim verbatim — do not add new facts. Keep "
            "the markdown heading structure."
        )
        body = "\n".join(text for _h, text in sections)
        limitations_block = "\n".join(f"- {item}" for item in limitations)
        user = (
            f"# Research Report: {question}\n\n{body}\n\n"
            f"## Limitations & Open Critiques\n\n{limitations_block}"
        )
        text = await self._call(system, user)
        return DraftResult(text=text, method=self.name())


def select_llm_provider() -> LLMProvider:
    """Chosen once per run, not per call — every agent in a run sees the
    same capability, so a report never mixes LLM-grade and heuristic-grade
    sections without saying so."""
    try:
        from config import Config
    except ImportError:
        Config = None  # running outside the Flask app package layout

    import os

    has_key = bool(getattr(Config, "ANTHROPIC_API_KEY", "") if Config else os.getenv("ANTHROPIC_API_KEY")) or bool(
        os.getenv("OPENROUTER_API_KEY")
    )
    if has_key:
        try:
            return AnthropicLLMProvider()
        except Exception:
            logger.exception("stigmergy_llm_provider_init_failed")
    logger.info("stigmergy_llm_provider_selected", extra={"provider": "heuristic", "reason": "no API key or init failed"})
    return HeuristicLLMProvider()
