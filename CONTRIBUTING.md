# Contributing

Thanks for your interest in improving Inagecas.

## Development setup

```bash
uv sync                 # install the workspace
uv run pre-commit install   # optional: run checks on commit
make check              # ruff + mypy + pytest
```

## Guidelines

- Keep code simple and readable; match the style of the surrounding modules.
- Every service is built from `create_service_app` and shares the `inagecas_shared`
  library — put cross-service logic there, not duplicated per service.
- Add or update tests for behaviour changes. Tests must not require external
  services (mock the model gateway, vector store, and database).
- Run `make check` before opening a pull request; CI runs the same checks plus a
  bandit scan and a dependency audit.
- Never commit secrets. Configuration belongs in `.env` (gitignored).

## Code review

Branches are reviewed commit by commit, so shape the history for a reader:

- One commit per change. A commit should do a single thing and still pass
  `make check` on its own, so it can be reverted or bisected in isolation.
- Say *why* in the message. The subject is the change; the body is the evidence
  that made it necessary — the failing query, the measured numbers, the
  behaviour that was wrong. A reviewer should not need the original bug report.
- A behaviour fix carries the test that would have caught it, and the test name
  states the rule ("refuses an answer of only citation markers"), not the
  mechanics.
- Keep unrelated cleanups out. A drive-by rename in a bug-fix commit costs the
  reviewer the diff they actually needed to read.

To review a branch locally:

```bash
git log --oneline main..HEAD       # the story, in order
git show <sha>                     # one change at a time
git diff main...HEAD --stat        # the whole surface
make check                         # lint, types, tests
```

## Database changes

Update the ORM models in `packages/shared/.../models.py` and add an Alembic
revision under `migrations/versions/`. Keep migrations and models in sync.
