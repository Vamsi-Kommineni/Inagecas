# Inagecas

An AI customer-support pipeline. It answers strictly from approved
documentation and **refuses or escalates when it isn't sure**. Model calls go
to hosted providers (Groq for text, OpenRouter for screenshots) with local
[Ollama](https://ollama.com) models as the automatic fallback, so a provider
outage or rate limit degrades quality instead of dropping tickets. Everything
can also run fully local on a CPU by pointing the aliases at Ollama in `.env`.

A policy gate screens every request, attached screenshots are read into
structured facts that steer retrieval, hybrid (dense + lexical) search with cross-encoder reranking finds evidence, and
every draft answer is validated for support and coverage before it is sent —
otherwise it is refused, turned into a clarifying question, or handed to a human
queue with the full context. Every stage is traced and measured, and a
regression eval harness scores the pipeline against a curated set.

## How it works

```
  client ─► api-gateway ─► policy gate ─► retrieval ──► answer ─► answer | refuse
            (public)        guardrails +   hybrid+       LiteLLM ─► Groq
                            classify       rerank        (fallback: Ollama)

  ingestion-worker (arq + Redis):  parse ─► chunk ─► embed ─► index ─► Qdrant
  Postgres records every run  ·  OpenTelemetry traces every step
```

A question flows: **gateway → policy gate (screen + classify + route) →
retrieve evidence (Qdrant) → generate grounded answer (LiteLLM → Groq, Ollama as fallback) →
answer or refuse**. The policy gate can short-circuit a request before
retrieval (blocked, off-topic, sensitive). Every step is recorded in Postgres
and traced through OpenTelemetry.

Between the gate and retrieval the gateway prepares what the models will see:
personal data in the question, the conversation and the screenshot text is
masked (`bob@x.com` becomes `[email]`), and a follow-up such as *"and on the
annual plan?"* is restated as a standalone question from the earlier turns
(`history` in the request). Repeated questions reuse their retrieval result
from a short-lived Redis cache that ingestion clears.

| Service | Responsibility |
|---|---|
| `api-gateway` | Public API: auth, rate limiting, `/ask`, `/ingest`, `/admin` |
| `policy-service` | Guardrails (PII / injection), intent classification, routing |
| `vision-service` | OCR screenshots into structured facts (RapidOCR) |
| `retrieval-service` | Hybrid dense+BM25 search, cross-encoder rerank, evidence |
| `answer-service` | Grounded generation + validation (coverage, support, confidence) |
| `escalation-service` | Human queue: work items, assignment, agent feedback |
| `ingestion-worker` | Parse → chunk → embed → index, as Redis-backed jobs |
| `litellm` | OpenAI-compatible gateway: hosted primaries, local Ollama fallbacks |
| Postgres / Qdrant / Redis | Records & audit / vectors / job queue |
| OTel + Prometheus + Grafana + Tempo | Traces, metrics, dashboards |

## Models

Everything runs through LiteLLM, so the services only ever name an alias
(`chat`, `judge`, `vision`, `embed`) and never talk to a provider directly.
Each chat-style alias has a hosted primary and a local fallback; LiteLLM
switches to the fallback when the primary errors, times out or rate-limits,
and the services never see the switch.

| Role | Primary | Local fallback | Notes |
|---|---|---|---|
| Chat | `openai/gpt-oss-120b` (Groq) | `llama3.2:3b` | Grounded answering, low temperature, short reasoning |
| Judge | `qwen/qwen3.8-27b` (Groq) | `llama3.2:3b` | Grounding verdicts, intent classification, LLM reranking; a different model than chat, so the judge is independent |
| Vision | `inclusionai/ling-3.0-flash-vl:free` (OpenRouter) | `gemma3` (4B) | Reads screenshots; OCR remains the last resort |
| Embeddings | `nomic-embed-text` (Ollama) | — | 768-dim dense vectors; local so the vectors in Qdrant keep one geometry |
| Lexical | `Qdrant/bm25` (fastembed) | — | Sparse BM25 for exact-term matching |
| Reranker | `ms-marco-MiniLM-L-6-v2` (fastembed ONNX) | — | CPU cross-encoder, on by default |

`LITELLM_*_MODEL` and `LITELLM_*_FALLBACK` in `.env` name the real models
behind each alias, with the LiteLLM provider prefix (`groq/`, `openrouter/`,
`ollama_chat/`). Keys go in `GROQ_API_KEY` and `OPENROUTER_API_KEY`. The judge
and vision primaries are reasoning models; the LiteLLM config turns thinking
off for the judge (one-word verdicts) and keeps it short for chat. The BM25 and
cross-encoder models are tiny local ONNX/statistical models via
[fastembed](https://github.com/qdrant/fastembed). If you switch the embedding
model, set `INAGECAS_EMBED_DIM` to match and recreate the collection.

Be deliberate about the hosted primaries: questions, retrieved passages and
customer screenshots are sent to the provider. To keep every byte on the
machine, point the `LITELLM_*_MODEL` aliases at the same `ollama_chat/...`
models as the fallbacks.

### Using an Ollama already on the host

If Ollama is installed on the machine itself, let LiteLLM use it instead of the
containerised one. On Docker Desktop this sidesteps the VM's memory cap (a 3B
chat model does not fit next to the rest of the stack in a 4 GB VM), and on a
Mac the host Ollama gets the GPU:

```bash
ollama pull llama3.2:3b && ollama pull gemma3 && ollama pull nomic-embed-text
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up -d
```

## Routing & policy

Before any retrieval happens, the gateway asks the `policy-service` what to do
with a request. The decision has two parts:

- **Deterministic guardrails** (no model, always reliable): detect personal data
  in the input (flagged and kept out of logs) and block obvious prompt-injection
  attempts.
- **Intent classification** (small LLM, best-effort): sort the question into
  `how_to`, `troubleshooting`, `billing_account`, `sensitive`, `image_required`,
  or `off_topic`, with a deterministic complexity score.

The combined **route** is one of `answer`, `refuse`, `escalate`, or `block`.
Anything other than `answer` short-circuits before retrieval and is recorded in
the `policy_runs` table; route counts are exported as the
`policy_route_decisions_total` metric. Toggle the layer with
`INAGECAS_POLICY_ENABLED` and choose how sensitive traffic is handled with
`INAGECAS_POLICY_SENSITIVE_ACTION` (`escalate` or `answer`).

## Retrieval

Retrieval combines semantic and exact-term matching so error strings and product
names are found as reliably as paraphrased questions:

1. **Hybrid recall** — the query is searched two ways: dense embeddings
   (semantic) and sparse **BM25** (lexical). The two rankings are merged with
   Reciprocal Rank Fusion, so a strong hit in *either* surfaces.
2. **Reranking** — a local ONNX **cross-encoder** reorders the fused candidates.
   It scores each passage under its title and heading, the same text the
   embedder saw, so a question that matches a section's heading finds it. It
   only decides the order: reranker outputs rank well but are not comparable
   across queries, so the dense cosine a chunk arrived with stays its score. Set
   `INAGECAS_RERANK_STRATEGY` to `cross_encoder` (default), `llm`, or `none`.
3. **Filtering & metadata** — only `active` documents are searched, with optional
   payload filters. Dense cosine is the score `INAGECAS_RETRIEVAL_MIN_SCORE` is
   measured against and the confidence composition builds on.
4. **Staleness & diversity** — evidence older than `INAGECAS_STALENESS_DAYS` is
   flagged `stale`, and the response reports `source_count` (distinct documents)
   as a cross-source signal.

Toggle lexical fusion with `INAGECAS_HYBRID_ENABLED`. The BM25 and cross-encoder
models download once (a few tens of MB) and are cached in a Docker volume.

> **Upgrading from an earlier phase?** The Qdrant collection schema changed
> (named dense + sparse vectors). Delete the old collection so it is recreated:
> `docker compose down -v` (or drop just the `qdrant_data` volume), then re-seed.

## Screenshots

Attach a screenshot to a question and it becomes **structured evidence**, not a
free vision-chat path. When `/ask` includes a base64 `image_base64`:

1. The `vision-service` reads it into lines of text plus facts: readability, a
   detected error message, and a likely screen title. By default that is OCR
   (RapidOCR, CPU ONNX). With `INAGECAS_VISION_MODEL=vision` a vision model
   reads the screenshot instead and returns the same facts as JSON; if it
   fails or replies with anything else, OCR takes over for that request.
2. If the image is unreadable (too little text or low confidence), the request is
   refused with `image_unreadable` — asking for a clearer image rather than
   guessing.
3. Otherwise the facts are merged into the retrieval query, so the screenshot
   steers search toward the right documentation. An attached image also
   satisfies an `image_required` intent at the policy gate.

RapidOCR's models ship inside the package, so nothing is downloaded at runtime.

## Answer validation

A draft answer is not sent just because retrieval returned something. Before
delivery the answer-service checks it:

1. **Citation coverage** — what fraction of the answer's *prose* sentences cite a
   valid evidence passage (pure and deterministic). Sentences that are nothing
   but `[n]` markers make no claim, so they are not counted, and a draft with no
   prose at all is refused rather than scored. A draft that cites less than
   `INAGECAS_ANSWER_MIN_COVERAGE` of its sentences is refused too: an answer
   that does not say where its claims come from is not grounded.
2. **Support & contradiction** — the chat model acts as a grounding judge and
   returns `SUPPORTED`, `PARTIAL`, `UNSUPPORTED`, or `CONTRADICTED`.
3. **Confidence composition** — a single explainable score from retrieval,
   source diversity, coverage, and support (weights sum to 1.0), logged on the
   `answer_runs` row alongside the support verdict.

The outcome:

- **Contradicted or unsupported** → refused (`contradicted` / `unsupported`).
- **Composite below the threshold** → refused (`low_confidence`), or — if it sits
  in the clarification band — a single **clarifying question** is returned with
  status `clarification` instead.
- **Otherwise** → answered, with the composed confidence and citations.

Toggle with `INAGECAS_VALIDATION_ENABLED` and `INAGECAS_CLARIFICATION_ENABLED`.

## Human escalation

When the AI genuinely can't help — a `sensitive` routing decision, or a refusal
for `no_evidence` / `low_confidence` / `unsupported` / `contradicted` — the
gateway files a **work item** to the escalation queue (correct rejections like
off-topic or blocked injections are *not* escalated). Each item carries the full
package so an agent doesn't redo discovery: question, why it escalated, the draft
(if any), the retrieved evidence, image facts, and trace/run ids, plus an
assembled context summary.

Agents work the queue through the gateway (API-key auth):

```bash
GET  /admin/escalations?status=open        # the queue
GET  /admin/escalations/{id}               # full package + agent actions
POST /admin/escalations/{id}/claim         # {"agent": "..."}
POST /admin/escalations/{id}/resolve       # {"agent": "...", "resolution": "..."}
POST /admin/escalations/{id}/feedback      # {"agent": "...", "kind": "bad_retrieval"}
```

Feedback (`answer_useful` / `bad_retrieval` / `missing_docs` / `other`) is stored
against the escalation, linking agent actions back to the original AI run.
Toggle with `INAGECAS_ESCALATION_ENABLED`.

## Observability

Every request is traceable and every stage is measured:

- **Traces** — OpenTelemetry auto-instruments FastAPI, HTTPX, and SQLAlchemy, so
  a request is a single distributed trace across all services (→ OTel collector
  → Tempo, viewed in Grafana). The gateway adds pipeline spans and tags the
  trace with the decision (`route`, `status`, `refusal_reason`, `escalation_id`).
- **Domain metrics** — beyond HTTP metrics, each service exports Prometheus
  series: answer outcomes and confidence, refusal reasons, escalation rate,
  retrieval top-score distribution, validation verdicts, and OCR success.
- **Dashboards** — Grafana ships two: *Inagecas Overview* (API health/latency)
  and *Inagecas Pipeline* (the domain metrics above).
- **Alerts** — Prometheus rules for high refusal rate, low retrieval confidence,
  high escalation rate, lost escalations, OCR failures, and 5xx errors
  ([alerts.yml](infra/prometheus/alerts.yml)).
- **Explain a decision** — `GET /admin/runs/{answer_run_id}` assembles the full
  trail (policy → retrieval → image → answer → escalation) into one record, so
  an operator can see *why* a request answered, refused, or escalated. Every
  request also emits a structured `ask_completed` log line.

## Evaluation

A regression harness scores the running pipeline against a curated eval set of
answerable, unanswerable, ambiguous, and sensitive tickets
([acme_support.yaml](evaluation/eval_sets/acme_support.yaml)). After seeding:

```bash
export INAGECAS_API_KEYS=<your key>
make eval                 # or: uv run python -m evaluation --json --min-pass 0.7
```

Each ticket declares its expected outcome and the sources good retrieval should
cite. The report gives **refusal correctness** (unanswerable/off-topic actually
refused), **answer correctness**, **retrieval recall**, and **citation
correctness**, broken down by category — plus a **doc-gaps** list of answerable
tickets the system couldn't answer, so documentation holes show up as product
inputs. `--min-pass` makes it a CI-style gate for prompt/model/retrieval changes.

## Quick start

**Prerequisites:** Docker + Docker Compose. ~3 GB free disk for model weights.
First run is slow on CPU; that is expected.

```bash
# 1. Configure secrets
cp .env.example .env
#    Edit .env and set at least:
#    - INAGECAS_API_KEYS   (a strong random value)
#    - LITELLM_MASTER_KEY  (and INAGECAS_MODEL_GATEWAY_API_KEY to the same value)
#    - GROQ_API_KEY, OPENROUTER_API_KEY  (the hosted primaries)
#    - POSTGRES_PASSWORD, GRAFANA_ADMIN_PASSWORD
#    Generate values with:  python -c "import secrets; print(secrets.token_urlsafe(32))"

# 2. Bring up the stack (builds images, pulls models, runs migrations)
docker compose up -d --build

# 3. Load the bundled sample documentation
make seed

# 4. Ask a question
KEY=<your INAGECAS_API_KEYS value>
curl -s http://localhost:8000/ask \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"question":"How many members can the Free plan have?"}' | jq
```

A grounded answer comes back with citations; an unsupported question
(`"What is the capital of France?"`) returns a refusal instead of a guess, with
a `refusal_reason` for your code and a short `message` for the customer.

### API examples

```bash
# Ingest a document inline (markdown | html | text). Returns a run id.
curl -s http://localhost:8000/ingest \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"title":"Refunds","source_type":"markdown","content":"# Refunds\nMonthly plans are non-refundable."}'

# Check ingestion status
curl -s http://localhost:8000/ingest/<run_id> -H "X-API-Key: $KEY"

# List indexed documents
curl -s http://localhost:8000/admin/documents -H "X-API-Key: $KEY"

# Drop vectors left behind by an interrupted ingest (preview first)
make reconcile ARGS=--dry-run
make reconcile

# Ask with a screenshot attached (base64-encoded image)
IMG=$(base64 -w0 screenshot.png)
curl -s http://localhost:8000/ask \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d "{\"question\":\"why am I seeing this error?\",\"image_base64\":\"$IMG\"}" | jq
```

Interactive API docs are at http://localhost:8000/docs (disabled in production).

### Dashboards

| Tool | URL |
|---|---|
| Grafana (dashboards: *Inagecas Overview* + *Pipeline*) | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Tempo (traces, via Grafana) | http://localhost:3200 |
| Qdrant | http://localhost:6333/dashboard |

These ports, and LiteLLM's 4000, are bound to `127.0.0.1`, so they answer only
on the machine itself. The gateway on 8000 is the one port open to the network.

## Local development

The project is a [uv](https://docs.astral.sh/uv/) workspace: one shared library
plus seven services.

```bash
uv sync            # create the venv and install everything
make check         # ruff + mypy + pytest
uv run pytest      # tests only
```

You can run a single service against the compose datastores by overriding the
`INAGECAS_*` URLs to `localhost` and starting it with
`uv run uvicorn api_gateway.main:app --reload`.

## Configuration

All configuration is environment-driven (see [.env.example](.env.example)). The
most important values:

| Variable | Purpose |
|---|---|
| `INAGECAS_API_KEYS` | Comma-separated client API keys |
| `INAGECAS_INTERNAL_TOKEN` | Optional shared secret between internal services |
| `INAGECAS_RETRIEVAL_TOP_K` / `INAGECAS_RETRIEVAL_MIN_SCORE` | Retrieval breadth and floor |
| `INAGECAS_ANSWER_MIN_CONFIDENCE` | Below this composed score, refuse or clarify |
| `INAGECAS_VALIDATION_ENABLED` / `INAGECAS_CLARIFICATION_ENABLED` | Grounding check; clarifying-question mode |
| `INAGECAS_ANSWER_MIN_COVERAGE` | Refuse a draft that cites fewer than this share of its sentences |
| `INAGECAS_HYBRID_ENABLED` | Fuse dense + BM25 lexical search |
| `INAGECAS_RERANK_STRATEGY` | `cross_encoder`, `llm`, or `none` |
| `INAGECAS_STALENESS_DAYS` | Flag evidence older than N days (0 disables) |
| `INAGECAS_POLICY_ENABLED` / `INAGECAS_POLICY_SENSITIVE_ACTION` | Enable the policy gate; `escalate` or `answer` sensitive traffic |
| `INAGECAS_RETRIEVAL_CACHE_TTL` | Seconds a repeated question reuses its retrieval result (0 disables) |
| `INAGECAS_ESCALATION_LLM_SUMMARY` | Add a model-written briefing to hand-off notes |
| `LITELLM_*_MODEL` / `LITELLM_*_FALLBACK` | Hosted primary and local fallback behind each alias |

## Model I/O discipline

- **Outside text is data.** Retrieved passages and screenshot text are fenced
  in the prompt and the model is told they carry no instructions. The
  injection rules that screen the question also screen the conversation
  history and the screenshot text, so an "ignore previous instructions" banner
  in an image, or planted in an earlier turn, is blocked the same way.
- **Structured replies.** The judge, the intent classifier and the follow-up
  rewrite ask the provider for a JSON schema. A provider that cannot do that
  (LiteLLM drops the parameter) answers in prose, and every parser still
  accepts the old keyword form, so the local fallbacks keep working. The
  screenshot reader asks for JSON in its prompt instead, because vision
  providers reject the parameter.
- **Hand-off carries the conversation.** An escalation stores the earlier turns
  and opens its notes with them. `INAGECAS_ESCALATION_LLM_SUMMARY=true` adds a
  three-sentence briefing written by the chat model on top; the notes stand on
  their own if that call fails.

## Security

Designed to be safe to run and to publish — see [SECURITY.md](SECURITY.md).
In short: the gateway is the only public service, API keys are required,
secrets stay in `.env` (gitignored), logs are redacted, requests are bounded,
and answers are grounded. Review SECURITY.md before exposing it beyond localhost.

## Project status

Everything described above is implemented and covered by tests: grounded
answering with refusal, policy routing, hybrid retrieval with reranking,
screenshot reading, answer validation, the human escalation queue,
observability, and the evaluation harness.

[docs/pipeline-report.md](docs/pipeline-report.md) walks through how a question
moves through the system, with measured results and the known gaps. The next
work is there in priority order: a harder eval set, recording which model
answered and what it cost, and stronger guardrails.

## License

Released under the [MIT License](LICENSE).
