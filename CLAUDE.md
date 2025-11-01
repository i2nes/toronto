This file provides rules, context, and working conventions for Claude Code when operating in this repository. Follow these instructions strictly.

---

## CRITICAL RULES (READ FIRST)

- **Confirm intent before destructive changes.** Never drop data, rewrite migrations, or alter auth without explicit confirmation.
- **Prefer minimal dependencies.** Avoid adding new libraries unless necessary and approved in the plan you present first.
- **Plan → implement → verify.** For any non-trivial task, first: (1) read relevant files, (2) propose a short plan, (3) implement in small PR-sized steps, (4) run tests/linters, (5) summarize changes.
- **Respect the stack constraints:**
  - Python (venv + pip), not Poetry/Conda.
  - Tailwind **CLI binary** (no Node/npm projects unless already present).
  - E2E tests with **Playwright**.
- **Follow repo etiquette** below (branch names, Conventional Commits, code review checklist).
- **Security:** Never commit secrets. Use `.env` and example files. Treat `.env*` as sensitive.

---

## PROJECT CONTEXT

> Replace placeholders with your real project details as you go.

- **Purpose:** Local-first web app with backend in **Python** (Django *or* Quart), HTML + Tailwind CSS (CLI), minimal JS (Alpine.js acceptable).
- **Database:** Postgres (optionally pgvector). If using Postgres RLS: never bypass policies; set tenant/context GUCs before queries.
- **Testing:** Unit/integration with **pytest**; E2E with **Playwright**.
- **AI/RAG (optional):** Embeddings pipeline, background indexer; keep prompts & tools in `ai/` or `.claude/`.

**Directory hints**
```

config/         # settings, URLs (Django) or app factory (Quart)
apps/ or src/   # application modules
templates/      # HTML/Jinja templates
static/         # CSS/JS assets; Tailwind input/output paths noted below
tests/          # pytest tests; playwright/ for E2E specs
scripts/        # dev scripts & CLI utilities
.claude/        # optional: slash commands, hooks, settings.json

````

---

## CODING STANDARDS

### Python
- Use **type hints**; docstrings for public functions.
- Format with `black` and `isort`; run linters (`ruff` if configured).
- Logging: structured where available; no `print()` for app logic.

### Django
- Apps are small & cohesive; settings split by environment.
- Views: prefer class-based; keep business logic in services.
- Migrations: never edit old migrations; create new ones.

### Quart
- App factory pattern in `config/app.py`.
- Blueprints per domain; keep side-effects in `if __name__ == "__main__":` or CLI.

### Frontend
- HTML + Tailwind + DaisyUI; **no** heavy SPA frameworks unless already present.
- Alpine.js allowed for light interactivity.
- Accessibility: adhere to WCAG 2.1 AA where reasonable.

---

## COMMON COMMANDS (BASH)

> Claude: run these in a login shell unless specified. Assume macOS/Linux paths.

### Environment (pip + venv)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
````

### Run server

```bash
# Django
python manage.py migrate
python manage.py runserver

# Quart
export QUART_APP=config.app:create_app
quart run --reload
```

### Tailwind CLI (no Node project)

> The Tailwind standalone binary lives at `bin/tailwindcss` (commit it or fetch on CI).

```bash
# One-off build
./bin/tailwindcss -i static/src/styles.css -o static/dist/styles.css --minify

# Watch mode (use during dev)
./bin/tailwindcss -i static/src/styles.css -o static/dist/styles.css --watch
```

### Testing

```bash
# Unit/integration
pytest -q

# Playwright setup (first time) and run
pip install pytest-playwright
python -m playwright install
pytest -q tests/e2e

# With markers (e.g., mark e2e tests with @pytest.mark.e2e)
pytest -m "not e2e"   # fast suite
pytest -m e2e         # end-to-end only
```

### Lint & format

```bash
black .
isort .
ruff check . --fix    # if ruff is configured
```

### Database & migrations

```bash
# Django
python manage.py makemigrations
python manage.py migrate

# Alembic (if used with Quart)
alembic revision -m "desc"
alembic upgrade head
```

---

## TEST STRATEGY

* **Pytest:** fast, isolated tests; factory fixtures; mark slow/external tests with `@pytest.mark.slow`.
* **E2E (Playwright):** focus on critical user journeys; keep selectors stable via roles/data-testids; run in CI headless.
* For any new feature or bugfix: **add/update tests** before merging.

---

## REPOSITORY ETIQUETTE

* **Branches:** `feature/<short-desc>`, `fix/<short-desc>`, `chore/<short-desc>`
* **Commits (Conventional):**

  * `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, `build:`, `ci:`
  * Body: what & why; include risks/rollbacks when relevant.
* **PR checklist (Claude must verify before proposing merge):**

  * [ ] Tests added/updated and passing (unit + e2e where applicable)
  * [ ] Lint/format clean
  * [ ] Migrations (if model/schema changed)
  * [ ] Security & secrets untouched / .env unchanged
  * [ ] User-visible changes documented (README/CHANGELOG if needed)

---

## CLAUDE WORKING AGREEMENTS

* **Small steps.** Prefer a series of small PRs over one huge change.
* **Show your work.** Always include a brief plan and a summary of edits.
* **Ask before heavy refactors** or dependency changes.
* **Respect Tailwind CLI setup.** Do not introduce npm unless explicitly requested.
* **If using Postgres RLS / multitenancy:** ensure the tenant context (e.g., `app.current_tenant_id`) is set on every request and respected in all queries.

---

## TOOLS, PERMISSIONS & MEMORY

* Allowed by default: **read files**, propose diffs, run safe local commands (formatters, tests, static builds).
* **Requires confirmation:** package installs, DB schema changes, deleting files, re-initializing tools.
* Keep this file concise; update it over time. When you discover a stable command or rule, **append it here**.

**Optional hierarchy** (recommended for big repos):

```
CLAUDE.md               # repo-wide rules
tests/CLAUDE.md         # testing standards, fixtures, E2E notes
apps/<app>/CLAUDE.md    # app-specific patterns
docs/CLAUDE.md          # documentation style & structure
```

---

## QUICK START FOR CLAUDE

1. Read this file and the directory map.
2. Propose a short plan (bulleted). Get confirmation.
3. Make small, reviewable changes with tests.
4. Run formatters/linters/tests before proposing a merge.
5. Post a concise summary with follow-ups and risks.
