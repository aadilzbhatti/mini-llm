"""Sample scoring for labs.decode, vendored verbatim from the owner's mini_llm working tree at fdec5b7 (2026-10-14; the
degeneration metric is not on origin yet): report.py's PROMPTS and samples.py's prompt list, protocol constants and
per-sample metrics (rep4, distinct-n, exact loops, near-loop onset, topic retention). "Degenerate" = ends in an exact
loop or rep4 >= 0.5. The owner's 20 prompts x 5 draws are a holdout here: reported for accepted programs, never used to
select. Protected: candidates never see it.
"""
# ruff: noqa
import re

# the owner's GEN_PROMPTS (samples.py), resolved to literals; frozen: seeds are per index
GEN_PROMPTS: tuple[tuple[str, str], ...] = (
    ('definition', 'Photosynthesis is a process that'),
    ('biography', 'Albert Einstein was a German-born theoretical physicist who'),
    ('science_explainer', 'Oxygen is a chemical element with'),
    ('instructional', 'In this lesson, students will learn how to'),
    ('bullet_list', 'There are several benefits to regular exercise:\n- '),
    ('numbered_list', 'To solve a quadratic equation, follow these steps:\n1.'),
    ('enumeration', 'There are three main types of'),
    ('long_dependency', 'Although the treaty was signed in 1919, it'),
    ('attribution', 'According to a study published in'),
    ('numeric_units', 'The mountain rises to a height of'),
    ('agreement_gap', 'The students who had spent the entire semester preparing for the final examination in organic chemistry'),
    ('history', 'The French Revolution began in 1789, when'),
    ('anatomy', 'The human heart is a muscular organ that'),
    ('geography', 'The Amazon River flows through'),
    ('math_definition', 'In mathematics, a prime number is'),
    ('environment', 'Climate change refers to long-term shifts in'),
    ('recipe', 'To make bread at home, you will need'),
    ('literature', 'William Shakespeare wrote many plays, including'),
    ('technology', 'The internet began as a research project in'),
    ('economics', 'Inflation occurs when'),
)
DRAWS = 5
NEW_TOKENS = 256
TEMPERATURE, TOP_K = 0.7, 40
BASE_SEED = 20260929

_STOP = {
    "that",
    "this",
    "with",
    "from",
    "have",
    "were",
    "will",
    "there",
    "their",
    "which",
    "about",
    "these",
    "those",
    "into",
    "than",
    "then",
    "them",
    "they",
    "what",
    "when",
    "where",
    "while",
    "also",
    "been",
    "being",
    "some",
    "such",
    "several",
    "follow",
    "steps",
    "main",
    "types",
    "according",
    "although",
    "process",
}


def seed_for(prompt_index: int, draw: int) -> int:
    """Distinct per (prompt, draw) -- one shared seed would give every prompt the same uniform draws."""
    return BASE_SEED + 1000 * prompt_index + draw


def rep4(ids: list[int]) -> float:
    grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def distinct(ids: list[int], n: int) -> float:
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 if not grams else len(set(grams)) / len(grams)


def loop_info(ids: list[int], max_period: int = 64, min_len: int = 32) -> dict:
    """Does the sequence end in an exact cycle? -> {looped, period, onset} (earliest-starting cycle)."""
    best = None
    for p in range(1, min(max_period, len(ids) // 2) + 1):
        run = 0
        for i in range(len(ids) - 1, p - 1, -1):
            if ids[i] != ids[i - p]:
                break
            run += 1
        span = run + p  # tokens covered by the cycle, counting its first occurrence
        if span >= max(2 * p, min_len) and (best is None or len(ids) - span < best["onset"]):
            best = {"looped": True, "period": p, "onset": len(ids) - span}
    return best or {"looped": False, "period": None, "onset": None}


# A sample is DEGENERATE if it ends in an exact loop or its rep4 is at least this. The exact-loop test
# alone misses heavy repetition that never settles into a verbatim cycle (e.g. a list alternating
# "Safe Diet" / "Healthy Diet" in varying order: rep4 0.82, not looped).
DEGENERATE_REP4 = 0.5
# Near-loop: from some token on, every NEAR_WINDOW-token window has rep4 >= DEGENERATE_REP4.
NEAR_WINDOW = 64


def degeneration_onset(ids: list[int], window: int = NEAR_WINDOW, threshold: float = DEGENERATE_REP4) -> int | None:
    """Where the sample stops saying new things: the start of the unbroken run of `window`-token windows
    with rep4 >= threshold that reaches the end of the sample. None if the last window is below it.
    Catches near-loops (repetitive but never an exact cycle) that loop_info misses; for an exact loop
    the onset is at or before the cycle's."""
    starts = range(0, len(ids) - window + 1)
    if not starts:
        return None
    onset = None
    for i in reversed(starts):
        if rep4(ids[i : i + window]) < threshold:
            break
        onset = i
    return onset


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP}


def topic_retention(prompt: str, second_half: str) -> float | None:
    """Share of the prompt's content words (matched on a 5-letter stem) used again in the second half."""
    want = {w[:5] for w in content_words(prompt)}
    if not want:
        return None
    have = {w[:5] for w in content_words(second_half)}
    return len(want & have) / len(want)


def score_ids(prompt: str, ids: list[int], tokenizer) -> dict:
    """samples.score() on generated token ids: the owner re-encodes the decoded text first, so do the same."""
    text = tokenizer.decode(ids)
    ids = tokenizer.encode(text) if text else []
    loop = loop_info(ids)
    r4 = rep4(ids)
    onset = degeneration_onset(ids)
    return {
        "text": text,
        "rep4": round(r4, 3),
        "degenerate": bool(loop["looped"] or round(r4, 3) >= DEGENERATE_REP4),
        "near_loop": onset is not None,
        "distinct2": round(distinct(ids, 2), 3),
        "looped": loop["looped"],
        "topic": topic_retention(prompt, tokenizer.decode(ids[len(ids) // 2 :]) if ids else ""),
    }
