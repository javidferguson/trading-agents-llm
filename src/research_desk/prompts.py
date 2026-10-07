"""Versioned prompt loading.

``prompt_pack_version`` is a hash of the prompt files' contents, not a number
somebody remembers to bump. Two runs whose prompts differ are visibly
different in ``DecisionState``, which is what makes a Stage 8 comparison
between them honest -- and §4 already stamps it into every record for exactly
that reason.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from .config import REPO_ROOT

PROMPT_DIR = REPO_ROOT / "prompts"


class PromptNotFound(FileNotFoundError):
    """A node asked for a prompt that does not exist."""


@lru_cache(maxsize=None)
def _pack() -> tuple[dict[str, str], str]:
    """Every prompt, plus a hash identifying the set."""
    if not PROMPT_DIR.exists():
        raise PromptNotFound(f"no prompts directory at {PROMPT_DIR}")

    prompts: dict[str, str] = {}
    for path in sorted(PROMPT_DIR.glob("*.md")):
        prompts[path.stem] = path.read_text()

    digest = hashlib.sha256()
    for name in sorted(prompts):
        digest.update(name.encode())
        digest.update(prompts[name].encode())
    return prompts, digest.hexdigest()[:12]


def load_prompt(name: str) -> str:
    prompts, _ = _pack()
    if name not in prompts:
        raise PromptNotFound(
            f"no prompt {name!r} in {PROMPT_DIR}. Found: {sorted(prompts)}"
        )
    return prompts[name]


def prompt_pack_version() -> str:
    """Short hash of every prompt file. Changes when any prompt changes."""
    return _pack()[1]


def reset_cache() -> None:
    """Tests, and editing a prompt in a long-lived process."""
    _pack.cache_clear()
