"""The novelty guard. **Not optional (§5).**

> *"Novelty guard: every turn this round has ``new_information=False``, or a
> speaker's claim has >0.85 Jaccard overlap with its previous round. Local
> models loop and restate; this is not optional."*

Two independent signals, because each catches what the other misses:

* **Facilitator-verified ``new_information``** catches a speaker who changes
  the words and not the point. A self-report would be worthless -- a model
  restating itself believes it is making a new point.
* **Jaccard overlap** catches the reverse: a facilitator too agreeable to call
  a repeat a repeat. It needs no second opinion and cannot be talked out of
  its answer.
"""

from __future__ import annotations

import re

#: Words carrying no argumentative content. Kept short on purpose: an
#: aggressive stop-list makes two different claims look identical, and a
#: false "no novelty" ends a debate that was still working.
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as", "it",
    "its", "has", "have", "had", "will", "would", "could", "should", "may",
    "might", "can", "do", "does", "did", "not", "no", "so", "which", "while",
})

#: Above this, two claims are the same claim wearing different words.
DEFAULT_THRESHOLD = 0.85


def tokens(text: str) -> set[str]:
    """Content words, lowercased. Numbers are kept -- they are the argument."""
    words = re.findall(r"[a-z0-9.]+", text.lower())
    return {w.strip(".") for w in words if w.strip(".") and w not in STOPWORDS}


def jaccard(left: str, right: str) -> float:
    """Overlap of content words, 0 to 1."""
    a, b = tokens(left), tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_restatement(
    claim: str, previous: list[str], threshold: float = DEFAULT_THRESHOLD
) -> tuple[bool, float]:
    """Does ``claim`` repeat anything in ``previous``? Returns (verdict, score)."""
    if not previous:
        return False, 0.0
    best = max(jaccard(claim, earlier) for earlier in previous)
    return best > threshold, best
