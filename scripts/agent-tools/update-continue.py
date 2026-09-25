#!/usr/bin/env python3
"""update-continue.py — Scaffold or refresh CONTINUE.md for session handoff.

Usage:
    python scripts/agent-tools/update-continue.py --new           # Create a fresh CONTINUE.md from template
    python scripts/agent-tools/update-continue.py --save          # Update status/gotchas/blocks from current state
    python scripts/agent-tools/update-continue.py --verify        # Check that CONTINUE.md exists and bootstrap works

Template: project-agent-interface skill — CONTINUE.md handoff framework
Patchwork: modular model composition runtime
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════
# PROJECT-SPECIFIC CONFIGURATION — CUSTOMIZE THESE
# ═══════════════════════════════════════════════════════════════

# Derive project root from this script's location (scripts/agent-tools/)
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent.parent
CONTINUE_PATH = PROJECT_ROOT / "CONTINUE.md"
PROJECT_NAME = "patchwork"

# (label, command, expectation)
#   "exit"     — the command's exit code decides (used by `test -f`)
#   "empty"    — the command must print NOTHING (a clean working tree)
#   "nonempty" — the command must print something (a readable commit log)
# The expectation is declared, not inferred: polarity differs per check, and
# inferring "non-empty means ok" silently inverted the clean-tree check.
BOOTSTRAP_CHECKS: list[tuple[str, str, str]] = [
    ("Working tree clean", "git status --short", "empty"),
    ("Latest commits", "git log --oneline -5", "nonempty"),
    ("AGENTS.md exists", "test -f AGENTS.md", "exit"),
    ("MEMORY.md exists", "test -f MEMORY.md", "exit"),
    ("CONTINUE.md exists", "test -f CONTINUE.md", "exit"),
    ("plans/index.md exists", "test -f plans/index.md", "exit"),
    ("docs/references.md exists", "test -f docs/references.md", "exit"),
]

EXTERNAL_DEPS: list[tuple[str, str]] = [
    # Profile-local skill path kept generic: no machine-local profile name in-repo.
    ("project-agent-interface skill", "~/.hermes/profiles/<profile>/skills/software-development/project-agent-interface/"),
    ("colibri", "https://github.com/JustVugg/colibri"),
    ("llama.cpp / GGUF", "https://github.com/ggerganov/llama.cpp"),
    ("mergekit", "https://github.com/arcee-ai/mergekit"),
]

INTERNAL_FILES: list[tuple[str, str, str]] = [
    ("Master plan", "plans/index.md", "First planning doc"),
    ("AGENTS.md", "AGENTS.md", "Bootstrap"),
    ("MEMORY.md", "MEMORY.md", "Project state"),
    ("CONTINUE.md", "CONTINUE.md", "Session handoff"),
    ("docs/references.md", "docs/references.md", "External references"),
    ("scripts/agent-tools/", "scripts/agent-tools/", "Maintenance tools"),
    ("research/", "research/", "Investigation findings"),
    ("experiments/", "experiments/", "Prototypes"),
]


# ── Helpers ──

def _run(cmd: str, timeout: int = 15) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, cwd=str(PROJECT_ROOT),
                                        stderr=subprocess.STDOUT, timeout=timeout).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f"[error: {e!r}]"


def _evaluate_check(expect: str, ret: str) -> tuple[bool, str]:
    """Decide one bootstrap check's status, per its declared expectation.

    Do not infer success from "output is non-empty": a clean working tree prints
    nothing at all, so that inference reports a clean tree as a failure and a
    dirty tree as a pass.
    """
    errored = ret.startswith("[error")
    if expect == "exit":
        return (not errored), ("OK" if not errored else ret)
    if expect == "empty":
        ok = not errored and not ret.strip()
        return ok, (ret.strip() or "clean")
    ok = not errored and bool(ret.strip())
    return ok, ret


def _today_str() -> str:
    return date.today().isoformat()


def _git_commits(n: int = 5) -> str:
    result = _run(f"git log --oneline -{n}")
    return result or "(no commits yet)"


def _git_status_short() -> str:
    result = _run("git status --short")
    return result or "(clean)"


def _detect_active_phase() -> str:
    # Check plans/index.md for active phase markers
    plans = PROJECT_ROOT / "plans" / "index.md"
    if plans.exists():
        content = plans.read_text()
        for line in content.splitlines():
            stripped = line.strip()
            if "Phase" in stripped and "[ ]" in stripped:
                return stripped
    result = _run("git log --oneline -20", timeout=10)
    if result:
        for line in result.split("\n"):
            for marker in ["phase", "investigation"]:
                if marker in line.lower():
                    return line.strip()
    return "Phase 0 (Foundations)"


def _detect_active_files() -> list[str]:
    result = _run("git log --oneline -3 --name-only", timeout=10)
    files = [f.strip() for f in result.split("\n") if f.strip() and not f.startswith("[") and "/" in f]
    return list(dict.fromkeys(files))


# ── Template Builder ──

def _build_bootstrap_table() -> str:
    lines = []
    for label, cmd, expect in BOOTSTRAP_CHECKS:
        ok, result = _evaluate_check(expect, _run(cmd))
        status = "✅" if ok else "❌"
        lines.append(f"- [{status}] `{cmd}` — {label}")
        if result and len(result) < 200:
            lines.append(f"  ```\n  {result}\n  ```")
    return "\n".join(lines)


def _build_internal_files_table() -> str:
    lines = []
    for name, path, when in INTERNAL_FILES:
        exists = "✅" if (PROJECT_ROOT / path).exists() else "❌"
        lines.append(f"| `{path}` | {name} | {when} |")
    if not lines:
        lines.append("| *(set at session end)* | *(purpose)* | *(when)* |")
    return "\n".join(lines)


def _build_external_deps_table() -> str:
    lines = []
    for path, purpose in EXTERNAL_DEPS:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / path
        exists = "✅" if resolved.exists() else "❌"
        lines.append(f"| `{path}` {exists} | {purpose} |")
    if not lines:
        lines.append("| *(set at session end)* | *(purpose)* |")
    return "\n".join(lines)


def _build_decision_log() -> str:
    memory_path = PROJECT_ROOT / "MEMORY.md"
    if memory_path.exists():
        content = memory_path.read_text()
        in_decisions = False
        decisions = []
        for line in content.split("\n"):
            if "## Design Decision" in line:
                in_decisions = True
                continue
            if in_decisions:
                if line.startswith("## "):
                    break
                if line.strip().startswith("|") and "|" in line[1:]:
                    decisions.append(line.strip())
        if decisions:
            return "\n".join(decisions)

    result = _run('git log --oneline --merges -5', timeout=10)
    if result and result != "(clean)":
        return "\n".join(f"| *(merge)* | {line} | *(see commit)* |" for line in result.split("\n"))
    return "| D001-D004 | Qwen2.5, IQ4_XS, Stack→MoE path, post-hoc bridge | See plans/index.md §6 |"


def _build_file_manifest() -> str:
    result = _run("find . -maxdepth 3 -type f \\( -name '*.py' -o -name '*.md' -o -name '*.yaml' -o -name '*.toml' \\) | grep -v node_modules | grep -v .venv | grep -v __pycache__ | grep -v .pixi | sort | head -40", timeout=10)
    return result or "(project root listing)"


def build_continue() -> dict[str, str]:
    return {
        "bootstrap_checks": _build_bootstrap_table(),
        "git_status": _git_status_short(),
        "git_commits": _git_commits(),
        "active_phase": _detect_active_phase(),
        "active_files": "\n".join(f"- `{f}`" for f in _detect_active_files()[:10]),
        "internal_files": _build_internal_files_table(),
        "external_deps": _build_external_deps_table(),
        "decision_log": _build_decision_log(),
        "file_manifest": _build_file_manifest(),
        "today": _today_str(),
        "project_name": PROJECT_NAME,
        "project_root": str(PROJECT_ROOT),
    }


TEMPLATE = """# CONTINUE.md — {project_name}

> Session: {today} | Modular model composition runtime — Qwen2.5 family, IQ4_XS, 16 GB target.
> **Rule:** Overwrite at end of session. Never append. This is a snapshot, not a history.

---

## Bootstrap Sequence (Do This First, In Order)

```bash
cd .
git status                              # expected: {git_status}
git log --oneline -5                    # recent work: {git_commits}
```

{bootstrap_checks}

---

## Current Status

<!-- Write at session end. Active phase, what's working, what's broken. -->

Active phase: {active_phase}

Recently touched files:
{active_files}

---

## What Was Last Built / Decided

<!-- Checklist of what was completed. Set at session end. -->

_(Fill in at session end)_

---

## What's Next (Prioritized)

<!-- Actionable next steps. Tier by value/urgency. Set at session end. -->

### Tier 1 — Already Spec'd, Just Needs Building
_(See plans/index.md §2 — investigation checklist. Pick the first unchecked item in Phase 0.)_

### Tier 2 — High Value, Needs Design
_(Fill in at session end)_

### Tier 3 — Data Accumulation / Time-Bound
_(Fill in at session end)_

### Quick Wins (30 min or less)
_(Fill in at session end)_

---

## Open Questions / Blockers

_(Fill in at session end)_

---

## Gotchas

<!-- Earned lessons. Updated as discovered. -->

_(Fill in at session end)_

---

## Key File Paths

### Internal Files

| Path | Purpose | When |
|------|---------|------|
{internal_files}

### External Dependencies

| Path | Purpose |
|------|---------|
{external_deps}

---

## Key Decisions Made (and Why)

{decision_log}

---

## Quick Start Commands

```bash
# Common operations
cd .
python scripts/agent-tools/update-continue.py --verify    # Validate handoff
python scripts/agent-tools/update-agents.py --report      # Validate doc refs
python scripts/agent-tools/update-memory.py --state       # Bump freshness
```

---

## File Manifest

```
{file_manifest}
```

---

## Agent Handoff Checklist

- [ ] Update `What's Next` with the immediate next steps
- [ ] Log completion in `What Was Last Built`
- [ ] Record new gotchas
- [ ] Update decision log if architecture decisions were made
- [ ] Check open questions — resolved removed, new added
- [ ] Run `python scripts/agent-tools/update-memory.py --change "desc"` to log change
- [ ] Run `python scripts/agent-tools/update-agents.py --report` to validate refs
- [ ] **Overwrite this file** — do NOT append
"""


# ── Main ──

def cmd_new() -> None:
    sections = build_continue()
    content = TEMPLATE.format(**sections)
    CONTINUE_PATH.write_text(content)
    print(f"✅ CONTINUE.md created at {CONTINUE_PATH}")
    print(f"   {len(content)} bytes — {len([l for l in content.split(chr(10)) if l.strip()])} non-empty lines")
    failures = len([s for s in sections['bootstrap_checks'].split(chr(10)) if '❌' in s])
    print(f"   {failures} bootstrap check(s) failed — review before trusting snapshot.")


def cmd_save() -> None:
    if not CONTINUE_PATH.exists():
        print(f"❌ No CONTINUE.md at {CONTINUE_PATH}. Run --new first.")
        sys.exit(1)

    sections = build_continue()
    content = CONTINUE_PATH.read_text()
    old_bootstrap = _extract_section(content, "## Bootstrap Sequence")
    new_bootstrap = _build_bootstrap_section(sections)
    if old_bootstrap:
        content = content.replace(old_bootstrap, new_bootstrap)
    CONTINUE_PATH.write_text(content)
    print(f"✅ CONTINUE.md refreshed at {CONTINUE_PATH}")


def cmd_verify() -> None:
    if not CONTINUE_PATH.exists():
        print(f"❌ Handoff broken: no CONTINUE.md at {CONTINUE_PATH}")
        sys.exit(1)

    print(f"✅ CONTINUE.md exists ({CONTINUE_PATH})")
    print()

    failures = 0
    for label, cmd, expect in BOOTSTRAP_CHECKS:
        ok, result = _evaluate_check(expect, _run(cmd))
        status = "✅" if ok else "❌"
        print(f"  {status} {label}")
        if not ok:
            failures += 1
            print(f"     Command: {cmd}")
            print(f"     Error:   {result}")

    print()
    if failures == 0:
        print("✅ All bootstrap checks pass — handoff is valid.")
        sys.exit(0)
    else:
        print(f"❌ {failures} bootstrap check(s) failed — handoff may be stale.")
        sys.exit(1)


def _extract_section(content: str, header: str) -> str | None:
    lines = content.split("\n")
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if start is None and stripped.startswith(header):
            start = i
        elif start is not None and stripped.startswith("## "):
            end = i
            break
    if start is None:
        return None
    return "\n".join(lines[start:end])


def _build_bootstrap_section(sections: dict[str, str]) -> str:
    lines = ["## Bootstrap Sequence (Do This First, In Order)"]
    lines.append("")
    lines.append("```bash")
    lines.append("cd .")
    lines.append(f"git status                              # expected: {sections['git_status']}")
    lines.append(f"git log --oneline -5                    # recent work: {sections['git_commits']}")
    lines.append("```")
    lines.append("")
    if sections.get("bootstrap_checks"):
        # Re-add the header for the check section
        lines.append(sections["bootstrap_checks"])
    return "\n".join(lines)


def cmd_selftest() -> None:
    """Regression checks for the bootstrap-check evaluation. Non-zero exit on failure."""
    cases = [
        # (expect, output the command produced, should it count as a pass?)
        ("empty", "", True),                        # clean tree — prints nothing
        ("empty", " M AGENTS.md", False),           # dirty tree — the file list IS the failure
        ("empty", "[error: CalledProcessError(...)]", False),
        ("nonempty", "abc1234 a commit subject", True),
        ("nonempty", "", False),                    # unreadable repo history
        ("nonempty", "[error: TimeoutExpired(...)]", False),
        ("exit", "OK", True),
        ("exit", "[error: FileNotFoundError(...)]", False),
    ]
    failures = 0
    for expect, ret, want in cases:
        got, _ = _evaluate_check(expect, ret)
        if got != want:
            failures += 1
            print(f"  ❌ expect={expect!r} ret={ret!r} → {got}, wanted {want}")
    # The regression this exists for: an inverted clean-tree check reports a
    # clean tree as a failure and a dirty tree as a pass.
    if not _evaluate_check("empty", "")[0]:
        failures += 1
        print("  ❌ the clean-tree case is inverted again")
    print(f"  {len(cases)} checks, {failures} failure(s)")
    sys.exit(1 if failures else 0)


def print_usage() -> None:
    print(__doc__)
    print("Commands:")
    print("  python scripts/agent-tools/update-continue.py --new       Create a fresh CONTINUE.md")
    print("  python scripts/agent-tools/update-continue.py --save      Refresh dynamic sections")
    print("  python scripts/agent-tools/update-continue.py --verify    Validate handoff state")
    print("  python scripts/agent-tools/update-continue.py --selftest  Check the check logic")


def main() -> None:
    args = sys.argv[1:]

    if "--new" in args:
        cmd_new()
    elif "--save" in args:
        cmd_save()
    elif "--verify" in args:
        cmd_verify()
    elif "--selftest" in args:
        cmd_selftest()
    else:
        if not CONTINUE_PATH.exists():
            print(f"📝 No CONTINUE.md found at {CONTINUE_PATH}")
            print("   Run with --new to create one.")
            print()
        cmd_verify()


if __name__ == "__main__":
    main()
