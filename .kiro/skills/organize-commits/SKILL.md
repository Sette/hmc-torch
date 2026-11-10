---
name: organize-commits
description: 'Groups uncommitted files (modified and untracked) into cohesive semantic commits on a new branch, following the `[EMOJI] <type>(<scope>): <description>` format. Never uses `git add .`/`-A`, never commits secrets, never pushes, and only executes after plan confirmation. Usage: /organize-commits [optional context]'
allowed-tools: Bash(git status *), Bash(git diff *), Bash(git log *), Bash(git branch *), Bash(git checkout -b *), Bash(git switch -c *), Bash(git add *), Bash(git commit *), Bash(git rev-parse *), Bash(git tag -l *)
version: 1.0.0
---

# Organize Commits by Responsibility

Groups modified/untracked files into cohesive semantic commits. Every message follows the commit rules below. For a single commit of what is already staged, use `/smart-commit`. `$ARGUMENTS`, if provided, is extra context to understand the intent of the changes and name the branch.

## Message Rules

- Format: `[EMOJI] <type>(<scope>): <description>`
- Required emoji: ✨ `feat` · 🐛 `fix` · 📝 `docs` · 💄 `style` · ♻️ `refactor` · ✅ `test` · 🔧 `chore` · ⚡️ `perf` · 🧹 `clean`
- **English**, imperative (`add`, `fix`, `remove`), all lowercase, **no trailing period**
- No co-authorship/footer lines unless the user requests them

## Step 1 — Diagnosis

```bash
git status --short
git branch --show-current
git diff --stat HEAD
```

- If the repo has no commits yet (`No commits yet` / invalid `HEAD`), use `git diff --cached --stat` and the untracked list instead.
- If something is already staged, warn the user: the current staged content will be folded into the grouping plan (nothing is committed without going through the plan).
- If there is nothing to commit, say so and stop.

## Step 2 — Branch

- If on `main`/`master`/`develop` (or the default branch), create a descriptive branch from context:
  - `feat/<short-description>` — new features
  - `refactor/<short-description>` — refactoring
  - `chore/<short-description>` — configuration/maintenance
- If already on a working branch that looks appropriate, **ask** before creating a new one.
- Command: `git checkout -b <branch-name>` (only after Step 4 confirmation).
- If the context is a **hotfix on a closed version** (e.g. "fix 0.1.4"), follow the [Hotfix from tag](#hotfix-from-tag) section instead of branching from the current branch.

## Step 3 — Logical Grouping

Group by responsibility, in this priority order (also the commit order):

| Group | Typical files | Message |
| :--- | :--- | :--- |
| **Project config** | `.gitignore`, `pyproject.toml`, `Makefile`, `.env.example`, `*.lock`, `sonar-*.properties`, `.python-version`, `.dockerignore` | `🔧 chore(config): ...` |
| **Documentation** | `README.md`, `CHANGELOG.md`, `CLAUDE.md`, `docs/`, `*.md` at root | `📝 docs(<scope>): ...` |
| **Core / entrypoint** | `__init__.py`, `main.py`, `settings.py`, `state.py`, `factory.py`, `service.py`, `arguments.py` | `✨ feat(core): ...` |
| **Domain modules** | Each subfolder of `src/` with its own logic becomes **one separate commit** | `✨ feat(<module>): ...` |
| **Utilities** | `utils/`, `helpers/`, auxiliary functions | `✨ feat(utils): ...` |
| **Data / assets** | `data/`, config `.json`, `.sql`, `.csv` | `✨ feat(data)` or `🔧 chore(data)` |
| **Tests** | `tests/`, `*_test.py`, `test_*.py`, `conftest.py` | `✅ test(<scope>): ...` |
| **CI/CD and infra** | `.github/`, `Dockerfile`, `docker-compose.yml`, `k8s/`, deploy scripts | `🔧 chore(ci): ...` |
| **AI tools** | `.claude/`, `.kiro/` | `🔧 chore(claude)` / `🔧 chore(kiro)` |

- The table type is the default for **new** files. For **modified** files, read the diff and choose the real type (`fix`, `refactor`, `perf`, `style`, `clean`...).
- Each domain module (e.g. `embeddings/`, `pipelines/`, `vectorstore/`) gets **its own commit**.
- **Never** mix files from different domains in the same commit.
- A file that does not clearly fit any group → isolated commit.
- **Secrets are never committed** (`.env`, credential `.json`, keys, `.secrets/`, `*.pem`): leave them out of the plan and warn the user (suggest adding to `.gitignore`).

## Step 4 — Confirm the Plan

Before touching git, show:

```
Branch: feat/<short-description>  (new, from <current-branch>)

| # | Message | Files |
|---|---|---|
| 1 | 🔧 chore(config): add project configuration | .gitignore, pyproject.toml |
| 2 | ✨ feat(embeddings): add embedding generation | src/.../embeddings/* |
...

Out of plan (secrets/ignored): .env
Confirm? (yes / adjust)
```

Only proceed with explicit confirmation. Adjustments requested by the user (merge, split, rename) are applied to the plan and it is shown again.

## Step 5 — Execute

Create the branch (if needed) and, for each group in plan order:

```bash
git add -- <file1> <file2> ...
git commit -m "<message>"
```

- **Never** use `git add .` or `git add -A` — always explicit files.
- If a commit fails (hook, conflict), stop, show the error, and ask how to proceed. Do not use `--no-verify`.
- Never `git push`.

## Step 6 — Final Verification

```bash
git log --oneline <origin-branch>..HEAD
git status --short
```

Report the created branch and a commit summary table (hash, message, number of files). If the working tree is not clean, list what remains and why (e.g. secrets left out).

## Hotfix from tag

Closed versions are recorded by **tag** (`vX.Y.Z`), not by a `release/X.Y.Z` branch. Do not create or maintain release branches just to archive a version: the tag is immutable and release-please already generates it. To fix an old version, create a **temporary** branch from the tag:

```bash
git tag -l 'v*'                        # confirm the tag exists
git switch -c hotfix/<X.Y.Z> v<X.Y.Z>  # temporary branch from tag
```

1. Always start from the tag (a bare `git checkout v<X.Y.Z>` leaves you in *detached HEAD* and commits will be lost).
2. Run Steps 1–6 normally on the `hotfix/*` branch.
3. After the fix, **propose** (and only execute with explicit confirmation) the new tag and push: `git tag v<X.Y.Z+1>` and `git push origin hotfix/<X.Y.Z> v<X.Y.Z+1>`. This skill never pushes by default.
4. Remind the user to bring the fix to `main` (`git cherry-pick <hash>` or merge), otherwise it will be lost in the next release.
5. After the new tag is published, the `hotfix/*` branch can be deleted; the tag preserves the result.

Note: release-please assumes releases from `main`, so a hotfix tag on an old version is created manually. If the hotfix is on the **latest** version, prefer fixing on `main` and letting release-please generate the next one. Also check whether the deploy pipeline triggers on tags or only on branches before relying solely on the tag.
