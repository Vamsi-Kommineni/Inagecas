"""LLM intent classification with a safe fallback.

The classifier is best-effort: it sorts a question into one support category to
inform routing and analytics. The model is asked for a one-field JSON object;
a provider that cannot do structured output answers in prose, and a reply the
parser does not recognise falls back to ``unknown``, which routes to the normal
answer path where grounding is the final backstop.
"""

from __future__ import annotations

from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.schemas import Intent
from inagecas_shared.structured import parse_field

_CATEGORIES = {
    "how_to": Intent.how_to,
    "troubleshooting": Intent.troubleshooting,
    "billing_account": Intent.billing_account,
    "sensitive": Intent.sensitive,
    "image_required": Intent.image_required,
    "off_topic": Intent.off_topic,
}

SCHEMA = {
    "title": "intent",
    "type": "object",
    "properties": {"category": {"type": "string", "enum": list(_CATEGORIES)}},
    "required": ["category"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You classify a customer-support question into exactly one category.\n"
    "Categories: how_to, troubleshooting, billing_account, sensitive, "
    "image_required, off_topic.\n"
    "- sensitive: asks us to reveal, change or delete personal or account data we "
    "hold (delete my account data, read me my card number, change the owner's email).\n"
    "- Mentioning their own email, name or account is not sensitive: [email], [phone] "
    "and similar are masked details the customer typed. Asking how a feature works "
    "is how_to.\n"
    "- image_required: cannot be answered without seeing an attached screenshot.\n"
    "- off_topic: not about this product or its support.\n"
    'Reply with JSON: {"category": "<one of the categories>"}'
)


def _match(output: str) -> Intent:
    text = parse_field(output, "category").lower()
    for key, intent in _CATEGORIES.items():
        if key in text:
            return intent
    return Intent.unknown


async def classify(gateway: ModelGateway, question: str, *, model: str | None = None) -> Intent:
    output = await gateway.chat(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=24,
        model=model,
        json_schema=SCHEMA,
    )
    return _match(output)
