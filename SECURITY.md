# Security Policy

## Reporting a vulnerability

Please report security issues privately by opening a
[GitHub security advisory](https://docs.github.com/en/code-security/security-advisories)
on this repository rather than a public issue. We aim to acknowledge reports
within a few days.

## Security posture

This project is designed so that, by default:

- **Model traffic is explicit and masked.** Chat, judge and vision calls go to
  hosted providers (Groq, OpenRouter) named in `.env`; questions, retrieved
  passages and screenshots are sent there, with emails, card numbers, phone
  numbers and SSNs replaced by their category first. Embeddings and the
  automatic fallbacks run on local Ollama, and every alias can be pointed at
  Ollama to keep all data on the machine. Nothing else leaves the host.
- **The gateway is the only public service.** Retrieval, answering, and the
  ingestion worker are reachable only on the internal Docker network. An optional
  `INAGECAS_INTERNAL_TOKEN` adds a shared-secret check between services. The
  operator ports compose publishes (Qdrant, LiteLLM, Prometheus, Tempo, Grafana)
  are bound to `127.0.0.1`.
- **Authentication is required.** Public endpoints require an API key
  (`X-API-Key`), compared in constant time. The service returns `503` until at
  least one key is configured.
- **Requests are bounded.** Body size limits and Pydantic validation guard the
  API surface; `/ask` and `POST /ingest` are rate limited per client.
- **Secrets stay out of the repo.** Configuration comes from environment
  variables / `.env` (gitignored). Logs redact common secret-bearing fields.
- **Inputs are screened.** A policy gate blocks obvious prompt-injection
  attempts in the question, the earlier turns of the conversation and the
  screenshot text, and flags personal data
  before any model call; sensitive or off-topic requests are refused or
  escalated before retrieval. Retrieved passages and screenshot text are fenced
  as data in the prompt.
- **Images are reduced to text.** Screenshots are bounded by a size limit and
  reduced to text facts by the vision model (or on-device OCR as the fallback);
  only those facts move on through the pipeline.
- **Answers are grounded.** The system only answers from indexed, approved
  documentation and refuses when evidence or confidence is insufficient.

## Known limitations

These are measured and tracked in
[docs/pipeline-report.md](docs/pipeline-report.md); plan around them until they
are closed.

- **Admin routes share the client key and are not rate limited.** `/admin/*`
  accepts the same API key as `/ask`, so there is no separation between a
  customer-facing client and an operator, and only `/ask` and `POST /ingest`
  are throttled.
- **The injection guard is pattern-based.** A paraphrased attack gets past the
  regexes; what holds then is the intent classifier and grounding, which only
  answers from the indexed documentation.
- **PII masking covers four patterns** (emails, card numbers, phone numbers,
  SSNs). Names and addresses are not detected.
- **No retention policy.** The audit tables keep the customer's original words
  and the screenshots indefinitely.

## Before you deploy beyond localhost

- Generate strong, unique values for every secret in `.env`
  (`INAGECAS_API_KEYS`, `LITELLM_MASTER_KEY`, database and Grafana passwords).
- Set `INAGECAS_INTERNAL_TOKEN` and a restrictive `INAGECAS_CORS_ORIGINS`.
- Put the gateway behind TLS (a reverse proxy such as Caddy, Nginx, or Traefik).
- Do not expose Postgres, Redis, Qdrant, Prometheus, or the dashboards publicly.
- Keep dependencies and pinned image tags up to date.
