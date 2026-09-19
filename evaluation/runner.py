"""Run an eval set against a live gateway and report the results."""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import sys
from pathlib import Path

import httpx
import yaml

from inagecas_shared.schemas import AskResponse

from .models import EvalReport, Ticket, TicketOutcome
from .scoring import aggregate, normalize_expected, score_ticket

_DEFAULT_SET = Path(__file__).parent / "eval_sets" / "acme_support.yaml"


def load_tickets(path: Path) -> list[Ticket]:
    data = yaml.safe_load(path.read_text("utf-8"))
    return [Ticket.model_validate(item) for item in data]


def _encode_image(image: str | None, base_dir: Path) -> str | None:
    if not image:
        return None
    return base64.b64encode((base_dir / image).read_bytes()).decode()


async def _ask(client: httpx.AsyncClient, ticket: Ticket, base_dir: Path) -> AskResponse:
    body: dict[str, object] = {"question": ticket.question}
    if ticket.history:
        body["history"] = [turn.model_dump() for turn in ticket.history]
    image = _encode_image(ticket.image, base_dir)
    if image:
        body["image_base64"] = image
    response = await client.post("/ask", json=body)
    response.raise_for_status()
    return AskResponse.model_validate(response.json())


async def run_eval(base_url: str, api_key: str, eval_set: Path) -> EvalReport:
    tickets = load_tickets(eval_set)
    base_dir = eval_set.parent
    outcomes: list[TicketOutcome] = []
    async with httpx.AsyncClient(
        base_url=base_url, headers={"X-API-Key": api_key}, timeout=180.0
    ) as client:
        for ticket in tickets:
            try:
                outcomes.append(score_ticket(ticket, await _ask(client, ticket, base_dir)))
            except (httpx.HTTPError, ValueError, OSError) as exc:
                outcomes.append(
                    TicketOutcome(
                        id=ticket.id,
                        category=ticket.category,
                        expected=normalize_expected(ticket.expect),
                        actual="error",
                        passed=False,
                        note=str(exc)[:200],
                    )
                )
    return aggregate(outcomes)


def print_report(report: EvalReport) -> None:
    print(f"tickets={report.total} passed={report.passed} pass_rate={report.pass_rate}")
    print(f"  answer_correct   = {report.answer_correct}")
    print(f"  refusal_correct  = {report.refusal_correct}")
    print(f"  retrieval_recall = {report.retrieval_recall}")
    print(f"  citation_correct = {report.citation_correct}")
    print("by category:")
    for category, rate in sorted(report.by_category.items()):
        print(f"  {category}: {rate}")
    for outcome in report.outcomes:
        if not outcome.passed:
            print(f"  FAIL {outcome.id}: {outcome.note}")
    if report.doc_gaps:
        print("doc gaps (answerable tickets the system could not answer):")
        for gap in report.doc_gaps:
            print(f"  - {gap}")


def _first_api_key() -> str | None:
    keys = [k.strip() for k in os.environ.get("INAGECAS_API_KEYS", "").split(",") if k.strip()]
    return keys[0] if keys else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Inagecas pipeline against an eval set."
    )
    parser.add_argument("eval_set", type=Path, nargs="?", default=_DEFAULT_SET)
    parser.add_argument(
        "--base-url", default=os.environ.get("INAGECAS_EVAL_BASE_URL", "http://localhost:8000")
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON")
    parser.add_argument(
        "--min-pass", type=float, default=0.0, help="Exit non-zero if pass_rate is below this"
    )
    args = parser.parse_args()

    api_key = args.api_key or _first_api_key()
    if not api_key:
        parser.error("No API key. Pass --api-key or set INAGECAS_API_KEYS.")

    report = asyncio.run(run_eval(args.base_url, api_key, args.eval_set))
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print_report(report)
    if report.pass_rate < args.min_pass:
        sys.exit(1)
