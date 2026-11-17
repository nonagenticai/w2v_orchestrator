"""Deterministic text-quality helpers shared by the craft rubrics.

These were extracted verbatim from the retired ``narrative.py`` rubric, whose
``evaluate(artifact_uris, stage_execution)`` protocol had no live caller. The
five metric helpers below *did* have one — ``script_craft.score_script`` reuses
them so readability / TTR / sentence-variance / preamble become gates in the
``script_doctor`` loop rather than passive insights — so they live here now,
independent of the legacy stage-rubric protocol.

Pure and I/O-free: no randomness, no LLM, no DB, no network.
"""

from __future__ import annotations

import re

from src.services.evaluation.models import MetricResult
from src.services.evaluation.rubrics._kernel import metric


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex."""
    if not text.strip():
        return []
    parts = re.split(r"[.!?]+\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _count_syllables(word: str) -> int:
    """Count syllables using vowel-group heuristic."""
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) > 2 and word.endswith("e") and word[-2] not in "aeiou":
        word = word[:-1]
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in "aeiouy"
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(1, count)


def _readability_score(text: str) -> MetricResult:
    """Simple readability estimate using avg sentence and word length."""
    sentences = _split_sentences(text)
    if not sentences:
        return metric(
            "readability_score",
            0.0,
            0.20,
            raw_value={"grade": 0},
            detail="No narration text to evaluate",
            threshold="fail",
        )

    words = text.split()
    avg_sentence_len = len(words) / len(sentences)

    # Flesch-Kincaid grade level using syllable count
    avg_syllables = sum(_count_syllables(w) for w in words) / len(words) if words else 0
    grade = 0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59
    grade = max(0, min(grade, 20))

    # Continuous scoring: linear interpolation from ideal center
    ideal_center = 10.0
    ideal_half_width = 2.0  # 8-12 is perfect
    max_distance = 8.0  # beyond 2-18 = minimum

    distance = abs(grade - ideal_center)
    if distance <= ideal_half_width:
        score = 100.0
    elif distance <= max_distance:
        score = 100.0 - (distance - ideal_half_width) / (max_distance - ideal_half_width) * 70.0
    else:
        score = 30.0

    score = round(score, 1)
    return metric(
        "readability_score",
        score,
        0.20,
        raw_value={"grade": round(grade, 1), "avg_sentence_len": round(avg_sentence_len, 1)},
        detail=f"Readability grade {grade:.1f} (target 8-12)",
    )


def _lexical_diversity(text: str) -> MetricResult:
    """Rolling-window type-token ratio (repetitive vs incoherent-variety gate)."""
    words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
    if len(words) < 10:
        return metric(
            "lexical_diversity",
            50.0,
            0.15,
            raw_value={"ttr": 0},
            detail="Too few words for lexical diversity analysis",
            threshold="warn",
        )

    # Rolling-window average TTR to normalize for multi-section content
    window = min(100, len(words))
    if len(words) <= window:
        ttr = len(set(words)) / len(words)
    else:
        ttrs: list[float] = []
        step = max(1, window // 2)  # 50% overlap
        for start in range(0, len(words) - window + 1, step):
            chunk = words[start : start + window]
            ttrs.append(len(set(chunk)) / len(chunk))
        ttr = sum(ttrs) / len(ttrs)

    if 0.3 <= ttr <= 0.7:
        score = 100.0
        issue = "good diversity"
    elif ttr < 0.3:
        score = max(20.0, ttr / 0.3 * 70)
        issue = "too repetitive"
    else:
        score = max(40.0, 100.0 - (ttr - 0.7) / 0.3 * 60)
        issue = "too varied (may lack cohesion)"

    score = round(score, 1)
    return metric(
        "lexical_diversity",
        score,
        0.15,
        raw_value={"ttr": round(ttr, 3)},
        detail=f"Type-token ratio {ttr:.3f}: {issue}",
    )


_PREAMBLE_PATTERNS = [
    r"^here\s+is",
    r"^sure,?\s+here",
    r"^certainly",
    r"^of\s+course",
    r"^i['']d\s+be\s+happy",
    r"^i['']ll\s+create",
    r"^let\s+me\s+write",
    r"^below\s+is",
    r"^the\s+following",
]


def _preamble_contamination(text: str) -> MetricResult:
    """Flag an LLM's conversational preamble leaking into the deliverable text."""
    first_line = text.strip().split("\n")[0].lower() if text.strip() else ""
    contaminated = any(re.match(pat, first_line) for pat in _PREAMBLE_PATTERNS)

    if contaminated:
        score = 0.0
        detail = f"AI preamble detected: '{first_line[:60]}...'"
    else:
        score = 100.0
        detail = "No preamble contamination"

    return metric(
        "preamble_contamination",
        score,
        0.25,
        raw_value={"contaminated": contaminated, "first_line": first_line[:100]},
        detail=detail,
    )


def _sentence_length_variance(text: str) -> MetricResult:
    """Sentence-length coefficient of variation (monotone vs erratic rhythm)."""
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return metric(
            "sentence_length_variance",
            50.0,
            0.15,
            raw_value={"sentence_count": len(sentences)},
            detail="Too few sentences to assess rhythm",
            threshold="warn",
        )

    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((slen - mean) ** 2 for slen in lengths) / len(lengths)
    std = variance**0.5
    cv = std / mean if mean > 0 else 0

    if 0.3 <= cv <= 0.8:
        score = 100.0
        issue = "good rhythm variation"
    elif cv < 0.3:
        score = max(30.0, cv / 0.3 * 70)
        issue = "too uniform (monotonous)"
    else:
        score = max(40.0, 100.0 - (cv - 0.8) / 0.5 * 60)
        issue = "too erratic"

    score = round(score, 1)
    return metric(
        "sentence_length_variance",
        score,
        0.15,
        raw_value={"cv": round(cv, 3), "mean_length": round(mean, 1), "std": round(std, 1)},
        detail=f"Sentence length CV={cv:.2f}: {issue}",
    )
