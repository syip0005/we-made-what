---
name: git-pr
description: Pre-PR checklist — run tests, update docs/CLAUDE.md, then create the PR.
---

# Git PR

Run through the pre-PR checklist before creating the pull request.

## Phase 1: Tests

Run backend tests and fix any failures:

```bash
cd backend && uv run pytest tests/ -v
```

## Phase 2: Update CLAUDE.md

Check if any of the following changed in this branch:

- New architecture patterns or conventions
- New setup steps or environment variables
- New scripts or tools
- Changed project structure

Update `CLAUDE.md` at the repo root if needed.

## Phase 3: Final commit & PR

1. Stage and commit any remaining changes.
2. Push the branch.
3. Create the PR with `gh pr create`, including:
   - A concise title (under 70 chars)
   - A `## Summary` section with bullet points
   - A `## Test plan` section with checklist
   - The `🤖 Generated with Claude Code` footer
