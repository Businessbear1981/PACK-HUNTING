"""
Stigmergy-inspired multi-agent deep research system.

Core idea: agents never talk to each other. Every coordination signal is a
read or a write against ArtifactStore. See docs/reference/stigmergy-research.md
for the full design writeup (concurrency model, merge strategy, pipeline vs.
swarm execution) and docs/adr/0005-stigmergy-multi-agent-research.md for why.
"""
from __future__ import annotations

from orchestrators.stigmergy.store import ArtifactStore, FileArtifactStore, InMemoryArtifactStore
from orchestrators.stigmergy.orchestrator import run_pipeline, run_swarm, run_event_driven

__all__ = [
    "ArtifactStore",
    "FileArtifactStore",
    "InMemoryArtifactStore",
    "run_pipeline",
    "run_swarm",
    "run_event_driven",
]
