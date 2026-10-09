"""
Runnable end-to-end example: multi-round deep research over NEST's own
docs/ corpus (real files, not fixtures) using the stigmergy system.

    cd backend
    python -m orchestrators.stigmergy.demo "What is NEST's canonical database and why?"
    python -m orchestrators.stigmergy.demo "..." --mode swarm --workers 4
    python -m orchestrators.stigmergy.demo "..." --mode event
    python -m orchestrators.stigmergy.demo "..." --backend file   # durable, survives process exit

With no ANTHROPIC_API_KEY/OPENROUTER_API_KEY set, this still runs end to
end using HeuristicLLMProvider — weaker analysis, same pipeline, and the
printed summary says which provider produced the report.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # `backend/` importable when run directly

from orchestrators.stigmergy.orchestrator import AgentFactory, run_event_driven, run_pipeline, run_swarm
from orchestrators.stigmergy.providers import HeuristicLLMProvider, LocalCorpusSearchProvider, select_llm_provider
from orchestrators.stigmergy.store import FileArtifactStore, InMemoryArtifactStore

REPO_DOCS = pathlib.Path(__file__).resolve().parents[3] / "docs"


async def _main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Corpus text (real markdown from docs/) can contain characters outside
    # the Windows console's default cp1252 codepage (box-drawing glyphs in
    # tables, em dashes). Reconfigure rather than silently drop or corrupt.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    corpus_root = pathlib.Path(args.corpus) if args.corpus else REPO_DOCS
    search = LocalCorpusSearchProvider(corpus_root)
    llm = HeuristicLLMProvider() if args.llm == "heuristic" else select_llm_provider()
    store = FileArtifactStore(namespace=args.namespace) if args.backend == "file" else InMemoryArtifactStore()
    factory = AgentFactory(search, llm)

    print(f"[stigmergy-demo] corpus={corpus_root} llm={llm.name()} backend={args.backend} mode={args.mode}")

    if args.mode == "pipeline":
        report = await run_pipeline(store, args.question, factory, rounds=args.rounds)
    elif args.mode == "swarm":
        report = await run_swarm(
            store, args.question, factory, rounds=args.rounds, fanout=args.fanout, workers_per_role=args.workers
        )
    else:
        report = await run_event_driven(
            store, args.question, factory, rounds=args.rounds, fanout=args.fanout, workers_per_role=args.workers
        )

    print("\n" + "=" * 88)
    if report is None:
        print("No report was produced — inspect the audit log below for where coverage stalled.")
    else:
        print(report.content["text"])
    print("=" * 88 + "\n")

    audit = await store.audit_log()
    print(f"[stigmergy-demo] {len(audit)} artifact version(s) written across {len({r.key for r in audit})} key(s).")
    counts: dict[str, int] = {}
    for record in audit:
        counts[record.kind.value] = counts.get(record.kind.value, 0) + 1
    for kind, count in sorted(counts.items()):
        print(f"    {kind:14s} {count}")

    conflicts = await store.conflicts(unresolved_only=False)
    if conflicts:
        print(f"[stigmergy-demo] {len(conflicts)} conflict(s) recorded during this run.")

    print(f"[stigmergy-demo] store status: {store.status()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stigmergy multi-agent deep research demo")
    parser.add_argument("question", help="Research question to investigate")
    parser.add_argument("--mode", choices=["pipeline", "swarm", "event"], default="pipeline")
    parser.add_argument("--llm", choices=["auto", "heuristic"], default="auto", help="auto tries a real LLM key first; heuristic forces the offline fallback")
    parser.add_argument("--backend", choices=["memory", "file"], default="memory")
    parser.add_argument("--namespace", default="stigmergy_demo")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--fanout", type=int, default=3, help="swarm/event: scout sub-queries per round")
    parser.add_argument("--workers", type=int, default=3, help="swarm/event: concurrent workers per role")
    parser.add_argument("--corpus", default=None, help="Directory of .md/.txt files to search (default: repo docs/)")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
