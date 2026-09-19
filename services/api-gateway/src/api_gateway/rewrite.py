"""Turn a follow-up into a question that stands on its own.

"And on the annual plan?" retrieves nothing by itself. Given the earlier turns,
the judge model restates it as "Can I get a refund on the annual plan?", and
that is what retrieval and answering see. A first message needs no rewrite,
and a rewrite that comes back empty leaves the original question alone.
"""

from __future__ import annotations

from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.schemas import Turn
from inagecas_shared.structured import parse_field

# Only the tail of a long conversation matters for resolving references.
_MAX_TURNS = 6

_SYSTEM = (
    "Rewrite the customer's latest message as one standalone support question, "
    "using the conversation for context. Resolve words like it, that, this plan. "
    "Keep the customer's meaning and details; add nothing new. If the message "
    "already stands on its own, return it unchanged.\n"
    'Reply with JSON: {"question": "..."}'
)
SCHEMA = {
    "title": "standalone",
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}


def _transcript(history: list[Turn]) -> str:
    return "\n".join(f"{turn.role}: {turn.content}" for turn in history[-_MAX_TURNS:])


async def standalone_question(
    gateway: ModelGateway, question: str, history: list[Turn], *, model: str
) -> str:
    if not history:
        return question
    output = await gateway.chat(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"{_transcript(history)}\ncustomer: {question}"},
        ],
        temperature=0.0,
        max_tokens=200,
        model=model,
        json_schema=SCHEMA,
    )
    rewritten = parse_field(output, "question").strip()
    return rewritten or question
