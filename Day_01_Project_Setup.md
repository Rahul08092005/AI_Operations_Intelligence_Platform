# Day 1: Project Architecture and Local Setup

**Project:** AI Operations Intelligence Platform
**Week:** 1 (Data Engineering + ML Foundation)
**Estimated effort:** 6 to 7 hours
**Deliverable:** A running project skeleton where **React -> FastAPI -> PostgreSQL** connectivity is proven, with a clean folder structure, Git branching in place, Docker working and the core documentation files created.

---

## Table of Contents

1. [Goals and Scope](#1-goals-and-scope)
2. [Decisions Made on Day 1](#2-decisions-made-on-day-1)
3. [Time Plan](#3-time-plan)
4. [Step 0: Prerequisites](#4-step-0-prerequisites)
5. [Step 1: Local Workspace and Folder Structure](#5-step-1-local-workspace-and-folder-structure)
6. [Step 2: Git, GitHub and Branching Strategy](#6-step-2-git-github-and-branching-strategy)
7. [Step 3: Root Configuration Files](#7-step-3-root-configuration-files)
8. [Step 4: Python Environment](#8-step-4-python-environment)
9. [Step 5: Backend (FastAPI) Skeleton](#9-step-5-backend-fastapi-skeleton)
10. [Step 6: PostgreSQL and Docker Compose](#10-step-6-postgresql-and-docker-compose)
11. [Step 7: Frontend (React + Tailwind) Skeleton](#11-step-7-frontend-react--tailwind-skeleton)
12. [Step 8: Documentation Files](#12-step-8-documentation-files)
13. [Step 9: Tooling (Makefile, Env Check, Tests, CI)](#13-step-9-tooling-makefile-env-check-tests-ci)
14. [Step 10: Dockerize Backend and Frontend](#14-step-10-dockerize-backend-and-frontend)
15. [Step 11: Verify, Commit, Merge, Tag](#15-step-11-verify-commit-merge-tag)
16. [Challenges and Troubleshooting](#16-challenges-and-troubleshooting)
17. [Definition of Done](#17-definition-of-done)
18. [What NOT to Do on Day 1](#18-what-not-to-do-on-day-1)
19. [Handoff to Day 2](#19-handoff-to-day-2)

---

## 1. Goals and Scope

### Goals

- Create a **well-organized local folder structure** that supports the whole 4-week project, including data, documents, ML code, RAG code, models, tests and docs.
- Set up **Git with a clear branching strategy** and a remote GitHub repository.
- Set up the **development environment**: Python virtual environment, Node.js, Docker, PostgreSQL.
- Build a **minimal but real** backend (FastAPI), frontend (React + Vite + Tailwind) and database connection.
- Prove the full chain works: the frontend shows **"API: healthy, DB: connected"**.
- Create the required documentation and config files: `README.md`, `architecture.md`, `requirements.txt`, `.env.example`.
- Add basic tooling so later days run faster: Makefile, environment check script, smoke tests, optional CI.

### In scope

- Project skeleton, configuration, connectivity, documentation, tooling.

### Out of scope (do not start these today)

- Dataset downloads (Day 2), data pipelines (Day 3), ML models (Days 5 and 6), real API endpoints (Day 7), authentication, deployment, PyTorch and LLM integration.

---

## 2. Decisions Made on Day 1

These decisions are recorded now so they do not cause rework later. Copy them into `docs/architecture.md` (section "Decision Log") during Step 8.

| # | Decision | Choice | Reason |
|---|---|---|---|
| D1 | Python version | **3.11** (3.12 acceptable) | Best compatibility with xgboost, shap and scikit-learn |
| D2 | Node version | **Node 22 LTS** (20 LTS acceptable) | Required by current Vite versions |
| D3 | Frontend language | **JavaScript** (React), not TypeScript | Faster for a 3-week build. Switch only if TypeScript is a goal |
| D4 | Tailwind setup | **Tailwind CSS v4 with the Vite plugin** | Current setup path, no PostCSS config needed |
| D5 | Database migrations | **No Alembic.** Use `Base.metadata.create_all()` through an init script | Enough for a portfolio project. Avoids migration overhead |
| D6 | Data folder | **One root-level `data/` folder** (raw/processed/external) | Your plan shows `ml/data/` and your dataset doc shows `data/`. Use one, the dataset doc's version |
| D7 | Documents folder | **Root-level `documents/` holds the source corpus** (PDFs). `rag/` holds only code (ingestion, retrieval, generation) | Your plan shows `rag/documents/` and your dataset doc shows `documents/`. Separating code from content is cleaner |
| D8 | Name collision | `backend/app/models/` = **database (ORM) models**. Root `models/` = **trained ML artifacts** (`.pkl`) | Both exist in your plan. The ML artifact path is set by the `MODEL_DIR` env var to avoid confusion |
| D9 | Shared feature code | Feature engineering lives in `ml/preprocessing/` and is **imported by the backend** at inference time | Prevents training/serving mismatch. The backend Docker image copies `ml/` |
| D10 | Python dependencies | One venv at the repo root. `backend/requirements.txt` holds runtime deps. `requirements-dev.txt` holds notebook and test tools | Matches your plan and keeps Docker images lean |
| D11 | Dev workflow | **Database in Docker, backend and frontend run natively** for hot reload. Full-stack Docker is for verification | Fastest feedback loop |
| D12 | Dashboard route | Add a `dashboard.py` router | Your API structure lists `/dashboard` but the repo tree does not include the file |
| D13 | Auth | **None** | Avoid over-engineering (as your plan says) |
| D14 | PyTorch and LLM SDK | **Not installed on Day 1** | PyTorch is only needed for the optional LSTM (Week 2). Install the LLM SDK on Day 11 |

---

## 3. Time Plan

| Time | Step | Output |
|---|---|---|
| 0:00 to 0:30 | Step 0: Prerequisites | All tools installed and verified |
| 0:30 to 1:00 | Step 1: Folder structure | Full directory tree with placeholders |
| 1:00 to 1:30 | Step 2: Git and branches | Local repo, GitHub remote, `main`, `develop`, feature branch |
| 1:30 to 2:15 | Step 3: Root config files | `.gitignore`, `.env.example`, `pyproject.toml`, etc. |
| 2:15 to 3:00 | Step 4: Python environment | venv created, dependencies installed, env check passing |
| 3:00 to 4:15 | Step 5: Backend skeleton | FastAPI running with `/health` and route stubs |
| 4:15 to 4:45 | Step 6: PostgreSQL in Docker | Database container healthy, backend connects |
| 4:45 to 5:45 | Step 7: Frontend skeleton | Vite + Tailwind app showing health status |
| 5:45 to 6:15 | Step 8: Documentation | README, architecture, API and methodology stubs |
| 6:15 to 7:00 | Steps 9 to 11: Tooling, Docker for app, verify, merge, tag | Everything committed, merged and tagged |

---

## 4. Step 0: Prerequisites

### Required software

| Tool | Recommended version | Check command |
|---|---|---|
| Git | Latest | `git --version` |
| Python | 3.11 | `python --version` (or `python3 --version`) |
| Node.js | 22 LTS | `node --version` and `npm --version` |
| Docker Desktop (or Docker Engine + Compose v2) | Latest | `docker --version` and `docker compose version` |
| VS Code (or your IDE) | Latest | n/a |
| GitHub account | n/a | n/a |

### Optional but recommended

- **GitHub CLI** (`gh`) for creating the repo and PRs from the terminal: `gh --version`
- **Postman** (needed on Day 7)
- **DBeaver** or **pgAdmin** to inspect PostgreSQL visually
- **VS Code extensions:** Python, Pylance, Ruff, ESLint, Tailwind CSS IntelliSense, Docker, Jupyter

### Windows users

Use **WSL2 (Ubuntu)** for this project. The Makefile, shell commands and Docker bind mounts behave much better there. Keep the project **inside the WSL filesystem** (for example `~/projects/`), **not** under `/mnt/c/...`, because file watching and I/O are slow on mounted Windows drives.

### Verify everything

```bash
git --version
python3 --version
node --version
npm --version
docker --version
docker compose version
```

If `docker compose version` fails, you have the legacy `docker-compose` binary. Install Compose v2 (bundled with current Docker Desktop).

### Configure Git identity (one time)

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
git config --global pull.rebase false
```

---

## 5. Step 1: Local Workspace and Folder Structure

### 5.1 Choose a location

```bash
mkdir -p ~/projects
cd ~/projects
mkdir ai-operations-intelligence
cd ai-operations-intelligence
```

Avoid spaces in the path and avoid OneDrive or Dropbox synced folders (they interfere with `node_modules`, `.venv` and Docker volumes).

### 5.2 Target folder structure

```text
ai-operations-intelligence/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # optional CI (lint + tests + frontend build)
│
├── .vscode/
│   └── settings.json               # shared editor settings
│
├── backend/                        # FastAPI application
│   ├── Dockerfile
│   ├── requirements.txt            # runtime dependencies
│   └── app/
│       ├── __init__.py
│       ├── main.py                 # app factory, CORS, lifespan, router registration
│       ├── config.py               # settings loaded from environment
│       ├── routes/                 # HTTP layer (thin)
│       │   ├── __init__.py
│       │   ├── health.py
│       │   ├── data.py
│       │   ├── forecast.py
│       │   ├── anomaly.py
│       │   ├── prediction.py
│       │   ├── reasoning.py
│       │   ├── insights.py
│       │   └── dashboard.py
│       ├── services/               # business logic, model inference (Day 7+)
│       │   └── __init__.py
│       ├── schemas/                # Pydantic request/response models
│       │   └── __init__.py
│       ├── models/                 # SQLAlchemy ORM models (NOT ML artifacts)
│       │   └── __init__.py
│       └── database/
│           ├── __init__.py
│           └── session.py          # engine, session, Base, connection check
│
├── frontend/                       # React + Vite + Tailwind
│   ├── Dockerfile
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css
│       ├── components/
│       ├── pages/
│       ├── charts/
│       └── services/
│           └── api.js
│
├── ml/                             # all training / analysis code (importable package)
│   ├── __init__.py
│   ├── notebooks/                  # EDA and experiments
│   ├── preprocessing/              # ingestion, validation, cleaning, features
│   │   └── __init__.py
│   ├── forecasting/
│   │   └── __init__.py
│   ├── anomaly_detection/
│   │   └── __init__.py
│   ├── prediction/
│   │   └── __init__.py
│   └── explainability/
│       └── __init__.py
│
├── rag/                            # RAG CODE only (documents live in /documents)
│   ├── __init__.py
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   └── index/                      # local vector store (git-ignored)
│
├── data/                           # datasets (large files git-ignored)
│   ├── README.md
│   ├── raw/
│   │   ├── metropt/
│   │   ├── ai4i/
│   │   ├── retail/
│   │   └── m5/
│   ├── processed/                  # sensors.parquet, machines.parquet, etc. (Day 3)
│   └── external/
│       └── cmapss/
│
├── documents/                      # RAG source corpus (synthetic + public PDFs)
│   ├── README.md
│   ├── manuals/
│   ├── maintenance/
│   ├── inspection/
│   └── specifications/
│
├── models/                         # trained ML artifacts (*.pkl, git-ignored)
│   └── .gitkeep
│
├── scripts/                        # one-off and utility scripts
│   └── check_env.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_models.py              # placeholder (Day 5/6)
│   └── test_data.py                # placeholder (Day 3)
│
├── docs/
│   ├── architecture.md
│   ├── api.md
│   ├── methodology.md
│   └── figures/                    # charts exported from EDA
│
├── .editorconfig
├── .env                            # LOCAL ONLY, never committed
├── .env.example                    # committed template
├── .gitattributes
├── .gitignore
├── docker-compose.yml
├── Makefile
├── pyproject.toml                  # ruff + pytest configuration
├── README.md
├── requirements.txt                # root entry point (includes the others)
└── requirements-dev.txt            # notebooks, testing, linting
```

### 5.3 Create the directories

Run from the project root:

```bash
# Backend
mkdir -p backend/app/{routes,services,schemas,models,database}

# Frontend placeholders (Vite scaffolding is added in Step 7)
mkdir -p frontend/src/{components,pages,charts,services}

# ML
mkdir -p ml/{notebooks,preprocessing,forecasting,anomaly_detection,prediction,explainability}

# RAG (code only)
mkdir -p rag/{ingestion,retrieval,generation,index}

# Data
mkdir -p data/raw/{metropt,ai4i,retail,m5} data/processed data/external/cmapss

# Documents (RAG corpus)
mkdir -p documents/{manuals,maintenance,inspection,specifications}

# Models, scripts, tests, docs, CI, editor
mkdir -p models scripts tests docs/figures .github/workflows .vscode
```

### 5.4 Create Python package markers

```bash
touch backend/app/__init__.py
touch backend/app/{routes,services,schemas,models,database}/__init__.py
touch ml/__init__.py
touch ml/{preprocessing,forecasting,anomaly_detection,prediction,explainability}/__init__.py
touch rag/__init__.py
touch tests/__init__.py
```

### 5.5 Create placeholder files so empty folders are tracked by Git

Git does not track empty directories. Add `.gitkeep` files:

```bash
touch models/.gitkeep
touch data/raw/{metropt,ai4i,retail,m5}/.gitkeep
touch data/processed/.gitkeep data/external/cmapss/.gitkeep
touch documents/{manuals,maintenance,inspection,specifications}/.gitkeep
touch rag/{ingestion,retrieval,generation,index}/.gitkeep
touch ml/notebooks/.gitkeep docs/figures/.gitkeep
touch frontend/src/{components,pages,charts}/.gitkeep
```

### 5.6 Create the empty stub files

```bash
touch backend/app/main.py backend/app/config.py backend/app/database/session.py
touch backend/app/routes/{health,data,forecast,anomaly,prediction,reasoning,insights,dashboard}.py
touch backend/requirements.txt backend/Dockerfile
touch tests/{test_api,test_models,test_data}.py
touch scripts/check_env.py
touch docs/{architecture,api,methodology}.md
touch data/README.md documents/README.md
touch README.md requirements.txt requirements-dev.txt pyproject.toml
touch docker-compose.yml Makefile
touch .env.example .gitignore .gitattributes .editorconfig
touch .github/workflows/ci.yml .vscode/settings.json
```

### 5.7 Verify the structure

```bash
find . -maxdepth 3 -type d -not -path "./.git*" | sort
```

Compare the output against the tree in 5.2.

### Challenges in this step

| Challenge | Mitigation |
|---|---|
| Empty folders disappear from Git | `.gitkeep` files (done in 5.5) |
| `models/` (ML artifacts) vs `backend/app/models/` (ORM) confusion | Decision D8. Add a one-line comment at the top of each folder's README or `__init__.py` |
| Duplicate `data` and `documents` folders from the two plan documents | Decisions D6 and D7. Use only the root-level folders |
| Over-nesting | Do not create folders you will not use this week. Add more only when needed |

---

## 6. Step 2: Git, GitHub and Branching Strategy

### 6.1 Branching model

A lightweight, solo-friendly version of Git Flow:

```text
main        <- stable, always demo-ready. Updated once per week via PR from develop. Tagged.
  │
develop     <- integration branch. All feature branches merge here.
  │
feature/*   <- one branch per day or task. Short-lived (ideally less than 1 day).
```

| Branch | Purpose | Who commits directly | Merges into |
|---|---|---|---|
| `main` | Stable releases. Portfolio-visible state | Nobody (only initial commit and PR merges) | n/a |
| `develop` | Integrated work in progress | Only via PR or merge from `feature/*` | `main` (weekly) |
| `feature/<scope>` | A single task | You | `develop` |
| `fix/<scope>` | Bug fixes found during testing (Week 4) | You | `develop` |
| `docs/<scope>` | Documentation-only changes | You | `develop` |

### 6.2 Naming convention

```text
feature/day-01-project-skeleton
feature/day-02-data-sources
feature/day-03-data-pipeline
feature/day-04-eda
feature/day-05-anomaly-detection
feature/day-06-failure-model
feature/day-07-ml-api
fix/api-validation-errors
docs/readme-architecture
```

Rules: lowercase, hyphens, no spaces, short and descriptive.

### 6.3 Commit message convention (Conventional Commits)

```text
<type>(<optional scope>): <short imperative summary>

Types: feat, fix, docs, chore, build, test, refactor, perf, ci
Examples:
  feat(backend): add health endpoint with database check
  chore: add project folder structure
  docs: add architecture overview
  build: add docker-compose with postgres
```

Commit often (every meaningful, working unit), not once at the end of the day.

### 6.4 Initialize the local repository

```bash
git init
git branch -M main
```

### 6.5 Create the first commit on `main` (minimal)

Only a README, `.gitignore` and optionally a LICENSE go on `main` first. You will fill these files in properly in Step 3 and Step 8. For now, add placeholders:

```bash
echo "# AI Operations Intelligence Platform" > README.md
```

Temporarily write a minimal `.gitignore` (the full version is created in Step 3):

```bash
printf ".venv/\n__pycache__/\nnode_modules/\n.env\n" > .gitignore
git add README.md .gitignore
git commit -m "chore: initial commit"
```

### 6.6 Create the GitHub repository and push

**Option A: GitHub CLI**

```bash
gh auth login
gh repo create ai-operations-intelligence --private --source=. --remote=origin --push
```

**Option B: GitHub web UI**

1. Create a new repository named `ai-operations-intelligence`. Do **not** initialize it with a README, `.gitignore` or license.
2. Then run:

```bash
git remote add origin git@github.com:<your-username>/ai-operations-intelligence.git
git push -u origin main
```

Keep the repository **private** until Day 21 (portfolio packaging). Make it public only after you confirm no secrets or large raw datasets are committed.

### 6.7 Create `develop` and the Day 1 feature branch

```bash
git checkout -b develop
git push -u origin develop

git checkout -b feature/day-01-project-skeleton
git push -u origin feature/day-01-project-skeleton
```

All remaining Day 1 work happens on `feature/day-01-project-skeleton`.

### 6.8 Branch plan for Week 1

| Day | Branch | Merge target |
|---|---|---|
| 1 | `feature/day-01-project-skeleton` | `develop` |
| 2 | `feature/day-02-data-sources` | `develop` |
| 3 | `feature/day-03-data-pipeline` | `develop` |
| 4 | `feature/day-04-eda` | `develop` |
| 5 | `feature/day-05-anomaly-detection` | `develop` |
| 6 | `feature/day-06-failure-model` | `develop` |
| 7 | `feature/day-07-ml-api` | `develop`, then `develop` -> `main` |

### 6.9 Tags

| Tag | When |
|---|---|
| `day-01-complete` | End of Day 1 (on `develop`) |
| `week-1-complete` | End of Day 7 (on `main` after the PR merge) |
| `week-2-complete`, `week-3-complete` | End of those weeks |
| `v1.0.0` | After Week 4 final demo |

### 6.10 Optional: GitHub branch protection

On GitHub: Settings -> Branches -> add a rule for `main` that requires a pull request before merging. This enforces the discipline even though you work alone.

### Challenges in this step

| Challenge | Mitigation |
|---|---|
| Accidentally committing `.env`, datasets or `.venv` | Minimal `.gitignore` is created **before** the first commit. Full version in Step 3 |
| Long-lived branches that become painful to merge | One branch per day, merge the same day |
| Forgetting which branch you are on | Show the branch in your shell prompt or VS Code status bar. Run `git status` before committing |
| SSH key not configured | Use `gh auth login` (HTTPS) or follow GitHub's SSH key guide |

---

## 7. Step 3: Root Configuration Files

### 7.1 `.gitignore` (replace the minimal version)

```gitignore
# ---------- Python ----------
.venv/
venv/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.ipynb_checkpoints/
.coverage
htmlcov/

# ---------- Node / Frontend ----------
node_modules/
frontend/dist/
frontend/.vite/
npm-debug.log*

# ---------- Environment and secrets ----------
.env
.env.*
!.env.example
frontend/.env
frontend/.env.*
!frontend/.env.example

# ---------- Data (large files stay local) ----------
data/raw/*
!data/raw/*/
data/raw/*/*
!data/raw/*/.gitkeep
data/processed/*
!data/processed/.gitkeep
data/external/*
!data/external/cmapss/
data/external/cmapss/*
!data/external/cmapss/.gitkeep
!data/README.md

# ---------- ML artifacts ----------
models/*
!models/.gitkeep

# ---------- RAG index ----------
rag/index/*
!rag/index/.gitkeep

# ---------- Docker ----------
pgdata/

# ---------- OS / Editor ----------
.DS_Store
Thumbs.db
.idea/
*.swp
```

Notes:
- Raw datasets are excluded because of size and licensing. Your dataset document recommends providing official source links and a data-preparation script instead.
- `models/*.pkl` is ignored for now. On Day 21 you can decide to commit the small final models (or publish them as release assets).
- Processed Parquet files are regenerated by `make data` (Day 3), so they stay out of Git.

Verify the ignore rules work:

```bash
git status --short          # should NOT list .venv, node_modules, or .env
git check-ignore -v .env    # should show the matching .gitignore rule
```

### 7.2 `.env.example` (committed template)

```dotenv
# ===== Application =====
ENVIRONMENT=development
APP_VERSION=0.1.0

# ===== PostgreSQL (used by docker-compose and the backend) =====
POSTGRES_USER=aiops
POSTGRES_PASSWORD=change_me_locally
POSTGRES_DB=aiops
DB_PORT=5432

# Local (non-Docker) backend connects through localhost.
# Inside docker-compose, the backend uses host "db" (set in docker-compose.yml).
# If your password contains special characters, URL-encode them.
DATABASE_URL=postgresql+psycopg://aiops:change_me_locally@localhost:5432/aiops

# ===== CORS (comma-separated list of allowed frontend origins) =====
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# ===== Paths =====
MODEL_DIR=models
DATA_DIR=data

# ===== LLM (used from Day 11, leave empty for now) =====
LLM_PROVIDER=
LLM_API_KEY=
```

Create your real local file and edit the password:

```bash
cp .env.example .env
```

`.env` must **never** be committed. If it ever is, rotate every secret in it.

### 7.3 `frontend/.env.example`

Vite only reads env files from inside the `frontend/` directory and only exposes variables prefixed with `VITE_`.

```dotenv
VITE_API_BASE_URL=http://localhost:8000
```

```bash
cp frontend/.env.example frontend/.env
```

### 7.4 `.gitattributes`

Prevents Windows/Linux line-ending problems in shell scripts and Dockerfiles.

```gitattributes
* text=auto eol=lf
*.sh text eol=lf
Dockerfile text eol=lf
Makefile text eol=lf
*.png binary
*.jpg binary
*.pdf binary
*.pkl binary
*.parquet binary
```

### 7.5 `.editorconfig`

```ini
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 4

[*.{js,jsx,json,css,html,yml,yaml,md}]
indent_size = 2

[Makefile]
indent_style = tab
```

### 7.6 `pyproject.toml`

Tool configuration only. This is not a packaging file.

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
extend-exclude = ["frontend", "data", "models"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
# "." lets tests import the ml/ and rag/ packages, "backend" lets them import app
pythonpath = [".", "backend"]
testpaths = ["tests"]
```

### 7.7 `.vscode/settings.json`

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/.ruff_cache": true
  }
}
```

On Windows without WSL, change the interpreter path to `${workspaceFolder}\\.venv\\Scripts\\python.exe`.

### Challenges in this step

| Challenge | Mitigation |
|---|---|
| `.gitignore` negation rules not working (a parent directory is ignored, so children cannot be re-included) | The patterns above ignore directory **contents** (`data/raw/*`) instead of the directory itself, and re-include `.gitkeep` files. Always verify with `git status` |
| Secrets in `.env.example` | It must contain placeholders only |
| Password with `@`, `:` or `/` breaks `DATABASE_URL` | Use a simple local password or URL-encode it |

---

## 8. Step 4: Python Environment

### 8.1 Create and activate the virtual environment

```bash
python3.11 -m venv .venv

# macOS / Linux / WSL
source .venv/bin/activate

# Windows PowerShell (without WSL)
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

### 8.2 `backend/requirements.txt` (runtime dependencies)

Lower bounds are listed so the install succeeds with current releases. You will generate an exact lock file afterward (8.5).

```text
# --- API ---
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic>=2.8
pydantic-settings>=2.4
python-dotenv>=1.0
httpx>=0.27

# --- Database ---
sqlalchemy>=2.0
psycopg[binary]>=3.2

# --- Data processing ---
pandas>=2.2
numpy>=1.26
pyarrow>=16
openpyxl>=3.1

# --- Machine learning ---
scikit-learn>=1.5
xgboost>=2.1
shap>=0.46
joblib>=1.4

# --- Added in later days (do NOT install today) ---
# torch               (Week 2, optional LSTM)
# anthropic / openai  (Day 11, LLM reasoning)
# pypdf, sentence-transformers or similar  (Day 17, RAG)
# python-multipart    (Day 17, file upload)
```

### 8.3 `requirements-dev.txt`

```text
-r backend/requirements.txt

pytest>=8.0
pytest-cov>=5.0
ruff>=0.6

jupyterlab>=4.2
ipykernel>=6.29
matplotlib>=3.9
seaborn>=0.13
plotly>=5.23
```

### 8.4 Root `requirements.txt` (entry point named in your plan)

```text
# Install everything needed for local development
-r requirements-dev.txt
```

Install:

```bash
pip install -r requirements.txt
```

### 8.5 Freeze a lock file once everything works

```bash
pip freeze > requirements.lock.txt
```

Commit `requirements.lock.txt` so you can recreate the exact working environment later (Day 20 deployment, or if a new release breaks something).

### 8.6 Register the Jupyter kernel

```bash
python -m ipykernel install --user --name aiops --display-name "Python (aiops)"
```

### 8.7 `scripts/check_env.py`

A single script that proves the ML stack is healthy. This catches the xgboost, shap and numpy compatibility issues on Day 1 instead of Day 10.

```python
"""Verify that the Python environment is correctly set up.

Run from the project root:  python scripts/check_env.py
"""
import importlib
import sys

MODULES = [
    "fastapi", "uvicorn", "pydantic", "pydantic_settings", "sqlalchemy", "psycopg",
    "pandas", "numpy", "pyarrow", "openpyxl", "sklearn", "xgboost", "shap", "joblib",
]


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    failed = []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            print(f"  OK    {name:<18} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name:<18} {exc}")

    # Smoke test: train a tiny XGBoost model and explain it with SHAP
    try:
        import numpy as np
        import shap
        import xgboost as xgb

        rng = np.random.default_rng(42)
        X = rng.random((200, 4))
        y = (X[:, 0] > 0.5).astype(int)
        model = xgb.XGBClassifier(n_estimators=10, max_depth=2).fit(X, y)
        values = shap.TreeExplainer(model).shap_values(X[:5])
        print(f"  OK    XGBoost + SHAP smoke test (shap output shape: {np.array(values).shape})")
    except Exception as exc:  # noqa: BLE001
        failed.append("xgboost+shap smoke test")
        print(f"  FAIL  XGBoost + SHAP smoke test: {exc}")

    if failed:
        print(f"\nProblems found: {failed}")
        return 1
    print("\nEnvironment looks good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run it:

```bash
python scripts/check_env.py
```

Do not move on until it prints "Environment looks good."

### Challenges in this step

| Challenge | Mitigation |
|---|---|
| `shap` or `numba` fails with a numpy version error | Let pip resolve versions together. If it fails, pin `numpy<2` temporarily, rerun `check_env.py`, then freeze the working set |
| `psycopg` cannot find libpq | Use `psycopg[binary]` (already in the file) |
| Wrong Python picked up (system 3.12/3.13) | Create the venv with the explicit `python3.11` binary and confirm with `python --version` after activation |
| Slow or heavy install (shap, numba) | Expected. Do this while doing Step 5 reading or in a second terminal |
| Forgetting to activate the venv | Prompt shows `(.venv)`. VS Code interpreter set in `.vscode/settings.json` |

---

## 9. Step 5: Backend (FastAPI) Skeleton

### 9.1 `backend/app/config.py`

Settings are read from environment variables, with a local `.env` file at the repo root as fallback.

```python
"""Application settings, loaded from environment variables."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> parents[2] is the repository root
ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Operations Intelligence Platform"
    app_version: str = "0.1.0"
    environment: str = "development"

    database_url: str = "postgresql+psycopg://aiops:change_me_locally@localhost:5432/aiops"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_dir: str = "models"
    data_dir: str = "data"

    llm_provider: str = ""
    llm_api_key: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

### 9.2 `backend/app/database/session.py`

```python
"""Database engine, session factory, declarative base and connectivity check."""
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 3},  # fail fast if the DB is unreachable
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Base class for ORM models (tables are defined from Day 2/3)."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db() -> bool:
    """Return True if the database answers a trivial query."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False
```

### 9.3 `backend/app/routes/health.py`

```python
from fastapi import APIRouter

from app.config import get_settings
from app.database.session import check_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Report API and database status. Always returns HTTP 200 so the
    frontend can display a 'degraded' state instead of a network error."""
    settings = get_settings()
    db_ok = check_db()
    return {
        "status": "healthy" if db_ok else "degraded",
        "api": "up",
        "database": "connected" if db_ok else "disconnected",
        "version": settings.app_version,
        "environment": settings.environment,
    }
```

### 9.4 Route stubs

Create the same minimal file for each of: `data`, `forecast`, `anomaly`, `prediction`, `reasoning`, `insights`, `dashboard`. Example for `backend/app/routes/forecast.py`:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/status")
def status() -> dict:
    return {"module": "forecast", "status": "not_implemented"}
```

Use these prefixes and tags:

| File | `prefix` | Implemented on |
|---|---|---|
| `data.py` | `/data` | Day 7 (`GET /machines`, `GET /machine/{id}`) |
| `forecast.py` | `/forecast` | Day 9 |
| `anomaly.py` | `/anomaly` (and `POST /detect/anomaly` on Day 7) | Day 7 |
| `prediction.py` | `/predict` (and `POST /predict/failure` on Day 7) | Day 7 |
| `reasoning.py` | `/explain` | Day 10 and 11 |
| `insights.py` | `/insights` | Day 12 |
| `dashboard.py` | `/dashboard` | Day 13 and 14 |

Naming caution: your plan uses `/predict/failure` (Day 7), `/prediction` (API structure) and `/predict` (Week 4 tests). Use **`/predict`** consistently, and record this in `docs/api.md`.

### 9.5 `backend/app/main.py`

```python
"""FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database.session import check_db
from app.routes import (
    anomaly,
    dashboard,
    data,
    forecast,
    health,
    insights,
    prediction,
    reasoning,
)

logger = logging.getLogger("aiops")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
    logger.info("Starting %s v%s [%s]", settings.app_name, settings.app_version, settings.environment)
    logger.info("Database reachable: %s", check_db())
    # Day 7: load trained models once here (anomaly_model.pkl, failure_model.pkl)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered predictive operations and business intelligence API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
for module in (data, forecast, anomaly, prediction, reasoning, insights, dashboard):
    app.include_router(module.router)
```

### 9.6 Run the backend locally

```bash
source .venv/bin/activate
cd backend
uvicorn app.main:app --reload --port 8000
```

Check:

- `http://localhost:8000/health` returns JSON. Without the database running yet, `database` will say `disconnected` and `status` will say `degraded`. That is correct behavior.
- `http://localhost:8000/docs` shows the Swagger UI with all route groups.
- `http://localhost:8000/forecast/status` returns the stub response.

### Challenges in this step

| Challenge | Mitigation |
|---|---|
| `ModuleNotFoundError: app` | Start uvicorn from inside `backend/`, or set `pythonpath` (tests are already configured) |
| `.env` not found when starting from `backend/` | `config.py` resolves the root `.env` by absolute path |
| CORS errors later in the browser | CORS origins come from `CORS_ORIGINS`. Include both `localhost` and `127.0.0.1` |
| Hanging when the DB is down | `connect_timeout=3` in the engine settings |
| Putting logic into route files | Routes stay thin. Logic goes in `services/` from Day 7 |

---

## 10. Step 6: PostgreSQL and Docker Compose

### 10.1 `docker-compose.yml`

This first version defines only the database. Backend and frontend services are added in Step 10.

```yaml
name: ai-operations-intelligence

services:
  db:
    image: postgres:16-alpine
    container_name: aiops-db
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "${DB_PORT:-5432}:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 3s
      retries: 10

volumes:
  pgdata:
```

### 10.2 Start the database and verify

```bash
docker compose up -d db
docker compose ps                      # STATUS should become "healthy"
docker compose logs db --tail 20
```

Connect with `psql` inside the container:

```bash
docker compose exec db psql -U aiops -d aiops -c "SELECT version();"
```

### 10.3 Verify the backend sees the database

Restart uvicorn (or let `--reload` pick it up) and open `http://localhost:8000/health`. Expected:

```json
{
  "status": "healthy",
  "api": "up",
  "database": "connected",
  "version": "0.1.0",
  "environment": "development"
}
```

### 10.4 Reset the database (when needed)

```bash
docker compose down -v      # stops containers AND deletes the pgdata volume (all data lost)
docker compose up -d db
```

### Challenges in this step

| Challenge | Mitigation |
|---|---|
| Port 5432 already used by a local PostgreSQL | Set `DB_PORT=5433` in `.env` and update `DATABASE_URL` to port 5433 |
| Password change in `.env` has no effect | The password is baked into the volume on first start. Run `docker compose down -v` and start again |
| `database: disconnected` although the container is up | Wrong `DATABASE_URL` (host, port, password) or the container is not healthy yet. Check `docker compose ps` |
| Docker Desktop not running | Start Docker Desktop and wait for the engine to be ready |

---

## 11. Step 7: Frontend (React + Tailwind) Skeleton

### 11.1 Scaffold Vite

Because `frontend/` already contains placeholder folders, scaffold in a way that does not abort. From the project root:

```bash
cd frontend
npm create vite@latest . -- --template react
```

If prompted that the directory is not empty, choose **"Ignore files and continue"**. Then:

```bash
npm install
```

### 11.2 Install dependencies

```bash
npm install tailwindcss @tailwindcss/vite
npm install recharts react-router-dom
```

- **Tailwind** for styling.
- **Recharts** for charts (used from Day 13).
- **react-router-dom** because the dashboard has multiple pages (Executive, Forecasting, Anomaly, Risk, Operations Intelligence, Assistant).

### 11.3 `frontend/vite.config.js`

```js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,      // required so the dev server is reachable from Docker
    port: 5173,
    // watch: { usePolling: true },  // uncomment if hot reload fails in Docker on Windows/macOS
  },
});
```

### 11.4 `frontend/src/index.css`

Replace the entire contents with:

```css
@import "tailwindcss";
```

Delete the template's `src/App.css` and the default assets you do not need.

### 11.5 `frontend/src/services/api.js`

```js
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export const getHealth = () => request("/health");
```

All later API calls (forecast, anomaly, predict, explain, insights) are added to this one file, which keeps the base URL and error handling in a single place.

### 11.6 `frontend/src/App.jsx`

```jsx
import { useEffect, useState } from "react";
import { getHealth } from "./services/api";

function StatusCard({ label, value, ok }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-sm text-slate-500">{label}</p>
      <p className={`mt-1 text-xl font-semibold ${ok ? "text-emerald-600" : "text-red-600"}`}>
        {value}
      </p>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getHealth().then(setHealth).catch((err) => setError(err.message));
  }, []);

  return (
    <main className="min-h-screen bg-slate-50 p-8">
      <div className="mx-auto max-w-3xl">
        <h1 className="text-2xl font-bold text-slate-900">AI Operations Intelligence</h1>
        <p className="mt-1 text-slate-600">AI-Powered Predictive Operations and Business Intelligence System</p>

        <section className="mt-8 grid gap-4 sm:grid-cols-3">
          {error ? (
            <StatusCard label="API" value={`Unreachable (${error})`} ok={false} />
          ) : !health ? (
            <StatusCard label="API" value="Checking..." ok={true} />
          ) : (
            <>
              <StatusCard label="API" value={health.api === "up" ? "Healthy" : "Down"} ok={health.api === "up"} />
              <StatusCard
                label="Database"
                value={health.database === "connected" ? "Connected" : "Disconnected"}
                ok={health.database === "connected"}
              />
              <StatusCard label="Version" value={`${health.version} (${health.environment})`} ok={true} />
            </>
          )}
        </section>
      </div>
    </main>
  );
}
```

### 11.7 Run the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`. With the backend and database running you should see **API: Healthy** and **Database: Connected**.

Failure checks (do them deliberately):

1. Stop the backend. The page should show "Unreachable".
2. Stop the database (`docker compose stop db`). The page should show "Disconnected".

### 11.8 Create the page folder placeholders

No code yet, only the intended structure. These pages are built on Days 13 to 16:

```text
src/pages/ExecutiveOverview.jsx      (Day 13)
src/pages/Forecasting.jsx            (Day 13)
src/pages/AnomalyIntelligence.jsx    (Day 13)
src/pages/PredictiveRisk.jsx         (Day 14)
src/pages/OperationsIntelligence.jsx (Day 15)
src/pages/Assistant.jsx              (Day 16)
```

Do not create these files today. Just note them in `docs/architecture.md`.

### Challenges in this step

| Challenge | Mitigation |
|---|---|
| Tailwind styles do not apply | Confirm `@import "tailwindcss";` is in `index.css`, the `tailwindcss()` plugin is in `vite.config.js`, and `index.css` is imported in `main.jsx` |
| Scaffolding refuses a non-empty directory | Choose "Ignore files and continue" |
| Node too old for the current Vite | Use Node 22 LTS (or 20.19+) |
| Browser CORS error in the console | Backend `CORS_ORIGINS` must include `http://localhost:5173` |
| Env var change not picked up | Restart `npm run dev`. Only `VITE_`-prefixed variables are exposed |
| Template leftovers (logos, demo counter) clutter the repo | Delete them now |

---

## 12. Step 8: Documentation Files

### 12.1 `README.md`

````markdown
# AI Operations Intelligence Platform

**AI-Powered Predictive Operations and Business Intelligence System**

> Status: under active development (Week 1: Data Engineering + ML Foundation)

## Overview

An end-to-end AI platform for a simulated smart-manufacturing company. It ingests sensor,
business and document data; detects anomalies; forecasts production and demand; predicts
machine-failure risk; explains predictions with SHAP and an LLM; and presents actionable
insights through an interactive dashboard.

## Architecture

```text
Sensors + Business + Documents
        -> Data Engineering (ETL, validation, features)
        -> ML Engine (Forecasting | Anomaly Detection | Failure Prediction)
        -> AI Reasoning (SHAP + LLM + RAG)
        -> Business Insights (KPIs, alerts, recommendations)
        -> Web Dashboard (React)
```

See [docs/architecture.md](docs/architecture.md) for details.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite, Tailwind CSS, Recharts |
| Backend | Python, FastAPI |
| Database | PostgreSQL |
| ML | scikit-learn, XGBoost, SHAP (PyTorch optional) |
| AI | LLM API, lightweight RAG |
| Infrastructure | Docker, Docker Compose |

## Quick Start

### Prerequisites

Git, Python 3.11, Node.js 22 LTS, Docker with Compose v2.

### Setup

```bash
git clone <repo-url>
cd ai-operations-intelligence

cp .env.example .env
cp frontend/.env.example frontend/.env

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py

cd frontend && npm install && cd ..
```

### Run (development)

```bash
docker compose up -d db                                # PostgreSQL
cd backend && uvicorn app.main:app --reload            # API on :8000
cd frontend && npm run dev                             # UI on :5173
```

### Run everything in Docker

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| PostgreSQL | localhost:5432 |

## Project Structure

```text
backend/    FastAPI application
frontend/   React dashboard
ml/         Data engineering, models, notebooks
rag/        Retrieval-augmented generation code
data/       Datasets (raw data is not committed)
documents/  Synthetic operational documents (RAG corpus)
models/     Trained model artifacts
tests/      Automated tests
docs/       Architecture, API and methodology
```

## Data

Public datasets (MetroPT-3, AI4I 2020, UCI Online Retail, M5) are combined with synthetic
data through a unified internal schema. Datasets are not stored in this repository. See
[data/README.md](data/README.md) for sources, licenses and preparation steps.

## Roadmap

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Data + ML | Working ML backend |
| 2 | Forecasting + AI | Intelligence engine |
| 3 | Product + Dashboard | Complete platform |
| 4 | Testing / Grace | Stable portfolio release |

## License

Code: MIT (to be confirmed). Datasets keep their original licenses. See `data/README.md`.
````

### 12.2 `docs/architecture.md`

````markdown
# Architecture

## 1. Purpose

Demonstrate one complete business workflow: from operational data to explained,
actionable decisions, in a smart-manufacturing context.

## 2. Layered Architecture

```text
DATA SOURCES        Sensors (MetroPT-3, AI4I) | Business (Online Retail, M5, synthetic) | Documents (PDF)
      |
DATA ENGINEERING    Ingestion -> Validation -> Cleaning -> Integration -> Feature Engineering
      |
ML ENGINE           Forecasting | Anomaly Detection (Isolation Forest) | Failure Prediction (XGBoost)
      |
AI REASONING        SHAP -> Operational Context -> LLM (+ RAG over documents)
      |
BUSINESS INSIGHTS   KPIs | Alerts | Recommendations
      |
WEB PLATFORM        React dashboard | AI Assistant
```

## 3. Components

| Component | Location | Responsibility |
|---|---|---|
| Frontend | `frontend/` | Dashboards, charts, assistant UI |
| API | `backend/app/routes/` | HTTP layer only, validation and routing |
| Services | `backend/app/services/` | Business logic, model inference |
| Schemas | `backend/app/schemas/` | Pydantic request/response contracts |
| ORM models | `backend/app/models/` | PostgreSQL table definitions |
| ML code | `ml/` | Preprocessing, training, evaluation, explainability |
| RAG code | `rag/` | Document ingestion, retrieval, generation |
| Datasets | `data/` | Raw (ignored), processed, external |
| Documents | `documents/` | RAG source corpus |
| Artifacts | `models/` | Trained model files (ignored) |

## 4. Data Flow

1. Raw public datasets and synthetic generators produce standardized tables.
2. The pipeline validates, cleans, integrates and engineers features.
3. Processed data is stored as Parquet/CSV and loaded into PostgreSQL.
4. Models are trained offline and saved to `models/`.
5. The API loads models at startup and serves predictions.
6. The reasoning layer combines predictions, SHAP values and retrieved document context.
7. The frontend consumes the API.

## 5. Unified Data Model (target)

`machines`, `sensor_readings`, `production`, `maintenance`, `business_transactions`,
`predictions`, `anomalies`, `ai_insights`. Public datasets are mapped into this schema through
a synthetic machine-ID layer (documented on Day 2).

## 6. Environments and Ports

| Service | Port | Notes |
|---|---|---|
| Frontend (Vite) | 5173 | |
| Backend (FastAPI) | 8000 | Swagger at `/docs` |
| PostgreSQL | 5432 | Configurable with `DB_PORT` |

Development: database in Docker, backend and frontend run natively with hot reload.
Verification: full stack with `docker compose up --build`.

## 7. Git Workflow

`main` (stable, tagged weekly) <- `develop` (integration) <- `feature/*` (one per day).
Conventional Commit messages. See the README for details.

## 8. Decision Log

| # | Decision | Choice |
|---|---|---|
| D1 | Python | 3.11 |
| D2 | Node | 22 LTS |
| D3 | Frontend language | JavaScript |
| D4 | Tailwind | v4 with Vite plugin |
| D5 | Migrations | No Alembic. `create_all()` through an init script |
| D6 | Data folder | Single root `data/` |
| D7 | Documents | Root `documents/` is content, `rag/` is code only |
| D8 | `models/` vs `backend/app/models/` | Artifacts vs ORM |
| D9 | Feature code | Shared in `ml/preprocessing/`, imported by backend |
| D10 | Dependencies | One venv, `backend/requirements.txt` plus dev file |
| D11 | Dev workflow | DB in Docker, apps native |
| D12 | Dashboard route | `/dashboard` router added |
| D13 | Auth | None |
| D14 | PyTorch / LLM SDK | Deferred |

## 9. Planned Frontend Pages

Executive Overview, Forecasting, Anomaly Intelligence, Predictive Risk, Operations
Intelligence, AI Assistant.

## 10. Open Questions

- Mapping of public datasets to demo machines (decided Day 2).
- Vibration signal: MetroPT-3 has none, so it will be derived and labeled as synthetic (Day 2).
- AI4I has no time dimension, so the failure model predicts risk for the current operating state (Day 6).
````

### 12.3 `docs/api.md`

````markdown
# API Reference

Base URL (local): `http://localhost:8000`. Interactive docs: `/docs`.

## Conventions

- JSON request and response bodies.
- Failure prediction uses the `/predict` prefix everywhere (code, docs, tests).
- Errors: 404 unknown resource, 422 invalid input, 500 unexpected.

## Endpoints

| Method | Path | Status | Day |
|---|---|---|---|
| GET | `/health` | Implemented | 1 |
| GET | `/machines` | Planned | 7 |
| GET | `/machine/{id}` | Planned | 7 |
| POST | `/predict/failure` | Planned | 7 |
| POST | `/detect/anomaly` | Planned | 7 |
| GET | `/forecast/production` | Planned | 9 |
| GET | `/forecast/demand` | Planned | 9 |
| POST | `/explain/...` | Planned | 10 to 11 |
| GET | `/insights` | Planned | 12 |
| GET | `/dashboard/...` | Planned | 13 to 14 |

## GET /health

```json
{
  "status": "healthy",
  "api": "up",
  "database": "connected",
  "version": "0.1.0",
  "environment": "development"
}
```
````

### 12.4 `docs/methodology.md`

````markdown
# ML Methodology

(Filled in as each model is built.)

## Anomaly Detection (Day 5)
Model: Isolation Forest. Data: MetroPT-3 (+ derived signals). Evaluation: event-level recall,
lead time, false alarms per day.

## Failure Prediction (Day 6)
Model: XGBoost. Data: AI4I 2020. Failure-mode columns (TWF, HDF, PWF, OSF, RNF) are excluded
from features to prevent label leakage. Metrics: precision, recall, F1, ROC-AUC, PR-AUC.

## Forecasting (Day 8)
Baseline: moving average. Comparison: XGBoost / LSTM. Metrics: MAE, RMSE, MAPE.

## Explainability (Day 10)
SHAP TreeExplainer on the failure model.

## Known Limitations
- Public datasets are from unrelated domains and are combined through a synthetic mapping layer.
- Some signals (for example vibration) are derived/synthetic and labeled as such.
- AI4I labels are rule-generated. Reported metrics do not represent real-world performance.
````

### 12.5 `data/README.md`

````markdown
# Data

Raw datasets are not committed (size and licensing). Download them from the official sources
and place them in the matching folder.

| Dataset | Folder | Source | License |
|---|---|---|---|
| MetroPT-3 | `raw/metropt/` | https://archive.ics.uci.edu/dataset/791/metropt3%2Bdataset | CC BY 4.0 |
| AI4I 2020 | `raw/ai4i/` | https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset | CC BY 4.0 |
| UCI Online Retail | `raw/retail/` | https://archive.ics.uci.edu/dataset/352/online%2Bretail | CC BY 4.0 |
| M5 Forecasting (subset) | `raw/m5/` | https://www.kaggle.com/competitions/m5-forecasting-accuracy/data | Competition terms |

`processed/` is generated by the data pipeline (`make data`, Day 3). Synthetic data is labeled
as synthetic. Check each dataset's current license before redistribution.
````

### 12.6 `documents/README.md`

````markdown
# Documents (RAG Corpus)

All documents in this folder are **synthetic/demo documents** unless stated otherwise.
Created on Day 17: machine manuals, maintenance SOPs, inspection reports, equipment
specifications. The MetroPT-3 "Data Description" PDF can be used as an initial public test
document.
````

### 12.7 Required files check

Your plan requires `README.md`, `architecture.md`, `requirements.txt` and `.env.example`. These now exist (architecture lives at `docs/architecture.md`).

### Challenges in this step

| Challenge | Mitigation |
|---|---|
| Spending too long polishing docs | Day 1 docs are **drafts**. Keep each under 15 minutes. Polish on Day 21 |
| Docs and reality drifting apart | The Decision Log and endpoint table are updated at the end of each day |
| Nested code fences breaking Markdown | Use four backticks for the outer fence (as shown above) |

---

## 13. Step 9: Tooling (Makefile, Env Check, Tests, CI)

### 13.1 `Makefile`

Recipe lines **must start with a TAB character**, not spaces.

```makefile
.PHONY: help db-up db-down up down logs backend frontend test lint format check-env data

help:
	@echo "Targets:"
	@echo "  db-up      Start PostgreSQL in Docker"
	@echo "  db-down    Stop PostgreSQL"
	@echo "  backend    Run FastAPI with reload (venv must be active)"
	@echo "  frontend   Run the Vite dev server"
	@echo "  up         Run the whole stack in Docker"
	@echo "  down       Stop the whole stack"
	@echo "  logs       Tail container logs"
	@echo "  test       Run pytest"
	@echo "  lint       Run ruff checks"
	@echo "  format     Format code with ruff"
	@echo "  check-env  Verify the Python environment"
	@echo "  data       Build datasets (implemented on Day 3)"

db-up:
	docker compose up -d db

db-down:
	docker compose stop db

backend:
	cd backend && uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f --tail 50

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

check-env:
	python scripts/check_env.py

data:
	@echo "Data pipeline will be implemented on Day 3"
```

If you are on Windows without WSL, run the underlying commands directly from the table in Step 11 of the README.

### 13.2 `tests/test_api.py`

```python
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

STUB_PREFIXES = ["data", "forecast", "anomaly", "predict", "explain", "insights", "dashboard"]


def test_health_returns_expected_keys():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    for key in ("status", "api", "database", "version", "environment"):
        assert key in body
    assert body["api"] == "up"


def test_openapi_docs_available():
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
```

The health test does not require the database to be running: it only checks the response shape.

> Make sure the prefix of the routers matches this list. Day 1 stubs should use `/data`, `/forecast`, `/anomaly`, `/predict`, `/explain`, `/insights`, `/dashboard`. If you also want a stub-route test, loop over `STUB_PREFIXES` and assert `GET /<prefix>/status` returns 200.

Placeholders for later:

```python
# tests/test_models.py
# Model tests are added on Days 5 and 6.

# tests/test_data.py
# Data pipeline tests are added on Day 3.
```

Run:

```bash
pytest -v
ruff check .
```

### 13.3 `.github/workflows/ci.yml` (optional, about 15 minutes)

```yaml
name: CI

on:
  push:
    branches: [develop, main]
  pull_request:
    branches: [develop, main]

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest

  frontend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "22"
          cache: npm
          cache-dependency-path: frontend/package-lock.json
      - run: npm ci
      - run: npm run build
```

Skip this if time is tight. It is not required for the Day 1 deliverable.

### Challenges in this step

| Challenge | Mitigation |
|---|---|
| `make: command not found` on Windows | Use WSL, or run the commands manually |
| "missing separator" error in Makefile | Recipe lines must use TAB indentation |
| Tests cannot import `app` | `pyproject.toml` sets `pythonpath = [".", "backend"]`. Run `pytest` from the repo root |
| CI fails on lint | Run `ruff check . --fix` and `ruff format .` locally first |

---

## 14. Step 10: Dockerize Backend and Frontend

Use this to verify that the whole stack also works in containers. Day-to-day development stays native (decision D11).

### 14.1 `backend/Dockerfile`

The build context is the **repository root** so the image can include the shared `ml/` package (decision D9).

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# libgomp1 is required by xgboost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
COPY ml/ ./ml/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### 14.2 `frontend/Dockerfile`

```dockerfile
FROM node:22-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
```

Add `frontend/.dockerignore`:

```text
node_modules
dist
.env
```

Add a root `.dockerignore`:

```text
.git
.venv
**/__pycache__
data/raw
data/processed
models
frontend/node_modules
```

### 14.3 Final `docker-compose.yml`

```yaml
name: ai-operations-intelligence

services:
  db:
    image: postgres:16-alpine
    container_name: aiops-db
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "${DB_PORT:-5432}:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 3s
      retries: 10

  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    container_name: aiops-backend
    env_file: .env
    environment:
      # Inside Docker the database host is the service name "db", on the internal port 5432
      DATABASE_URL: postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}
      MODEL_DIR: /models
      DATA_DIR: /data
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
    volumes:
      - ./backend:/app
      - ./ml:/app/ml
      - ./models:/models
      - ./data:/data

  frontend:
    build:
      context: ./frontend
    container_name: aiops-frontend
    environment:
      VITE_API_BASE_URL: http://localhost:8000
    ports:
      - "5173:5173"
    depends_on:
      - backend
    volumes:
      - ./frontend:/app
      - /app/node_modules

volumes:
  pgdata:
```

Notes:
- `environment` takes precedence over `env_file`, so the Docker `DATABASE_URL` (host `db`) overrides the `localhost` value in `.env`.
- The browser, not the container, calls the API, so the frontend uses `http://localhost:8000`.
- The anonymous volume `/app/node_modules` keeps container dependencies from being overwritten by the host bind mount.

### 14.4 Run and verify

```bash
# Stop native processes first (ports 8000 and 5173 must be free)
docker compose up --build
```

Verify:

- `http://localhost:5173` shows API Healthy and Database Connected
- `http://localhost:8000/docs` loads
- `docker compose ps` shows three services running and the database healthy

Stop:

```bash
docker compose down
```

### Challenges in this step

| Challenge | Mitigation |
|---|---|
| xgboost import error in the container (`libgomp.so.1`) | `libgomp1` is installed in the Dockerfile |
| Very slow first backend build (shap, numba, pandas) | Normal. Subsequent builds use the Docker cache because `requirements.txt` is copied before the code |
| Backend cannot reach the database | Inside Docker use host `db`, not `localhost` (set by the compose `environment` override) |
| Frontend hot reload not working in Docker (Windows/macOS) | Enable `usePolling` in `vite.config.js` |
| `npm ci` fails | `package-lock.json` must exist and be committed. Run `npm install` once locally |
| Port already in use | Stop native uvicorn and Vite before running full-stack Docker |
| Image grows too large | Do not install PyTorch or sentence-transformers yet |

---

## 15. Step 11: Verify, Commit, Merge, Tag

### 15.1 Final verification checklist

Run each item and tick it off.

**Environment**
- [ ] `python scripts/check_env.py` prints "Environment looks good."
- [ ] `pytest -v` passes
- [ ] `ruff check .` passes

**Backend**
- [ ] `uvicorn app.main:app --reload` starts without errors
- [ ] `GET /health` returns `database: connected` with the DB running
- [ ] `GET /health` returns `database: disconnected` (HTTP 200) with the DB stopped
- [ ] `/docs` lists all route groups

**Database**
- [ ] `docker compose ps` shows `db` as healthy
- [ ] `psql` command from 10.2 works

**Frontend**
- [ ] `npm run dev` serves the page at `localhost:5173`
- [ ] Page shows API Healthy and Database Connected
- [ ] Page shows a sensible state with the backend stopped and with the DB stopped
- [ ] `npm run build` succeeds

**Docker**
- [ ] `docker compose up --build` brings up all three services
- [ ] The page works at `localhost:5173` from the containers

**Repository**
- [ ] `git status` shows no `.env`, `.venv`, `node_modules`, datasets or model files
- [ ] `git ls-files | grep -i env` shows only `.env.example` files
- [ ] Folder structure matches section 5.2
- [ ] `README.md`, `docs/architecture.md`, `requirements.txt`, `.env.example` exist and are filled in

### 15.2 Commit sequence (on `feature/day-01-project-skeleton`)

Commit in logical, working units. Suggested order:

```bash
git add .gitignore .gitattributes .editorconfig .env.example frontend/.env.example
git commit -m "chore: add repository config, ignore rules and env templates"

git add backend/ ml/ rag/ data/ documents/ models/ docs/figures tests/ scripts/ frontend/src/*/.gitkeep
git commit -m "chore: add project folder structure and package markers"

git add requirements.txt requirements-dev.txt backend/requirements.txt pyproject.toml requirements.lock.txt
git commit -m "build: add python dependencies, lint and test configuration"

git add backend/app
git commit -m "feat(backend): add FastAPI app with config, CORS, health endpoint and route stubs"

git add docker-compose.yml
git commit -m "build: add docker-compose with PostgreSQL"

git add frontend
git commit -m "feat(frontend): scaffold React, Vite and Tailwind app with health status"

git add README.md docs data/README.md documents/README.md
git commit -m "docs: add README, architecture, API and methodology drafts"

git add tests scripts Makefile .vscode .github
git commit -m "test: add smoke tests, env check script, Makefile and CI workflow"

git add backend/Dockerfile frontend/Dockerfile frontend/.dockerignore .dockerignore docker-compose.yml
git commit -m "build: dockerize backend and frontend"
```

Adjust the groupings if a command reports "pathspec did not match". The goal is clear, small commits, not exact grouping.

Push:

```bash
git push origin feature/day-01-project-skeleton
```

### 15.3 Merge into `develop`

Pull request (recommended, even solo):

```bash
gh pr create --base develop --head feature/day-01-project-skeleton \
  --title "Day 1: Project skeleton" \
  --body "Folder structure, FastAPI + PostgreSQL + React skeleton, Docker, docs, tooling."
```

Review your own diff on GitHub, look for secrets and large files, then merge (merge commit or squash, your choice, but stay consistent). Without the GitHub CLI, do the same in the web UI, or locally:

```bash
git checkout develop
git pull origin develop
git merge --no-ff feature/day-01-project-skeleton -m "Merge Day 1: project skeleton"
git push origin develop
```

Clean up:

```bash
git branch -d feature/day-01-project-skeleton
git push origin --delete feature/day-01-project-skeleton
```

### 15.4 Tag

```bash
git tag -a day-01-complete -m "Day 1: project skeleton complete"
git push origin day-01-complete
```

### 15.5 Prepare the Day 2 branch

```bash
git checkout develop
git checkout -b feature/day-02-data-sources
```

---

## 16. Challenges and Troubleshooting

### Summary of the highest-risk issues

| # | Risk | Likelihood | Impact | Prevention |
|---|---|---|---|---|
| 1 | Secrets or datasets committed to Git | Medium | High | `.gitignore` before the first commit. Check `git status` before every commit |
| 2 | Dependency conflicts (numpy, shap, xgboost, numba) | Medium | High | `check_env.py` today. Freeze `requirements.lock.txt` |
| 3 | Docker networking confusion (`localhost` vs `db`) | High | Medium | Compose `environment` override. Remember: host = `localhost`, containers = service names |
| 4 | Port conflicts (5432, 8000, 5173) | Medium | Low | Configurable `DB_PORT`. Stop other services |
| 5 | WSL / Windows file system slowness | Medium | Medium | Keep the repo in the Linux filesystem |
| 6 | Over-engineering the skeleton | High | Medium | Strict scope (section 1). Time-box each step |
| 7 | Folder structure drift between plan documents | Medium | Medium | Decisions D6 to D9 made explicitly |
| 8 | Tailwind or Vite version mismatch | Low | Medium | Follow the Tailwind v4 + Vite plugin steps exactly |
| 9 | CORS errors | Medium | Low | Env-driven origins, both `localhost` and `127.0.0.1` |
| 10 | Skipping verification | Medium | High | Use the checklist in 15.1 |

### Troubleshooting table

| Symptom | Likely cause | Fix |
|---|---|---|
| `database: disconnected` in `/health` | DB not running, wrong URL or port | `docker compose ps`. Check `DATABASE_URL` in `.env` |
| `password authentication failed` | `.env` password differs from the one stored in the volume | `docker compose down -v`, then `up -d db` |
| `port is already allocated` (5432) | Local PostgreSQL running | Set `DB_PORT=5433` and update `DATABASE_URL` |
| `ModuleNotFoundError: app` | Wrong working directory | Run uvicorn from `backend/`, or pytest from the repo root |
| `ImportError: libgomp.so.1` | Missing system library for xgboost in a slim image | Install `libgomp1` in the Dockerfile |
| `numpy.dtype size changed` or `numba` error on `import shap` | Binary incompatibility | Reinstall in a fresh venv. If needed pin `numpy<2` and re-run `check_env.py` |
| CORS error in the browser console | Origin not allowed | Add the exact origin to `CORS_ORIGINS` and restart the backend |
| Page shows "Unreachable" | Backend not running, or wrong `VITE_API_BASE_URL` | Check the backend URL. Restart Vite after changing `.env` |
| Tailwind classes have no effect | Missing import or plugin | `@import "tailwindcss";` in `index.css`, plugin in `vite.config.js` |
| `npm create vite` aborts | Directory not empty | Choose "Ignore files and continue" |
| Vite hot reload broken in Docker | File events not propagated | `server.watch.usePolling = true` |
| `make: *** missing separator` | Spaces instead of tabs | Convert recipe indentation to TABs |
| Git reports CRLF/LF warnings | Mixed line endings | `.gitattributes` and `.editorconfig` (already set) |
| `.env` appears in `git status` | `.gitignore` not applied or already tracked | `git rm --cached .env`, then verify with `git check-ignore -v .env` |
| Docker build very slow | Large context or no layer cache | Add `.dockerignore`. Keep `requirements.txt` copy before the code copy |

---

## 17. Definition of Done

Day 1 is complete when **all** of the following are true:

1. The folder structure in section 5.2 exists locally and on GitHub.
2. `main`, `develop` branches exist on the remote, Day 1 work is merged into `develop`, and `day-01-complete` is tagged.
3. `docker compose up -d db` starts a **healthy** PostgreSQL container.
4. The backend starts, `/health` reports the real database state, and `/docs` loads.
5. The frontend renders a status page driven by the live `/health` response (healthy and degraded states both verified).
6. `docker compose up --build` runs all three services successfully.
7. `python scripts/check_env.py`, `pytest` and `ruff check .` all pass.
8. `README.md`, `docs/architecture.md`, `requirements.txt` and `.env.example` exist with real content.
9. No `.env`, virtual environment, `node_modules`, datasets or model files are tracked by Git.
10. The decision log (D1 to D14) is recorded in `docs/architecture.md`.

**Deliverable statement:** *Running project skeleton: React -> FastAPI -> PostgreSQL, containerized, documented and version-controlled.*

---

## 18. What NOT to Do on Day 1

- Do not download or process datasets (Day 2).
- Do not design the full database schema or add ORM models (Day 2 and 3).
- Do not install PyTorch, LangChain, vector databases or LLM SDKs.
- Do not build dashboard pages or charts.
- Do not add authentication, user accounts or rate limiting.
- Do not set up cloud deployment (Day 20).
- Do not spend more than 15 minutes on any single documentation file.
- Do not start Day 2 work on the Day 1 branch.

---

## 19. Handoff to Day 2

Before you finish, write these three lines at the bottom of `docs/architecture.md` under "Next Steps", so tomorrow starts without ramp-up time:

1. **Day 2 goal:** download MetroPT-3, AI4I 2020 and UCI Online Retail (plus the M5 subset), write data dictionaries, define the unified schema and fleet mapping, and build generators for maintenance and production data.
2. **Day 2 decisions to make first:** how to derive vibration for MetroPT-3, how to assign public-dataset rows to machine IDs M-101 to M-112, and the "hero machine" (M-104) degradation story.
3. **Day 2 branch:** `feature/day-02-data-sources`.

Also prepare for tomorrow:

- Create a Kaggle account and accept the M5 competition rules (this can block downloads).
- Check free disk space (MetroPT-3 is about 1.5 million rows and several hundred MB when processed).
