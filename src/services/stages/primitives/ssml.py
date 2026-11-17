"""Lower a narration line + its performance directive into SSML (TTSB-3).

This is a pure leaf helper: no I/O, no config, no imports of the narrate
stages. A narration line carries two *structured* directive fields produced by
``voice_director``:

- ``pause_before_ms`` — a leading pause, in milliseconds.
- ``emphasis_indices`` — the ``text.split()`` word positions to stress.

``build_line_ssml`` renders those into a ``<speak>`` document that the TTS
backend can parse. The contract has three load-bearing properties:

1. **None means "plain text".** When there is nothing to express
   (``pause_before_ms <= 0`` AND ``emphasis_indices`` empty), the function
   returns ``None`` so the caller sends the raw ``text`` exactly as today —
   byte-identical, cache-stable behaviour.
2. **Indices are ``text.split()`` positions.** ``voice_director`` computes
   ``word_count = len(text.split())`` and emits indices against that same
   whitespace tokenization, so we MUST tokenize with ``text.split()`` here for
   the indices to line up. Out-of-range indices are ignored.
3. **XML-escape before wrapping.** Every raw token is passed through
   ``xml.sax.saxutils.escape`` first (so ``&``/``<``/``>`` in e.g. ``"R&D"``
   become entities and never corrupt the backend's XML parse). The
   ``<emphasis>`` tags are added *after* escaping, so tags are never escaped.

The freeform ``directive.ssml`` string the LLM may emit is deliberately NOT
consumed here — only the structured fields above.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from src.services.stages.shared.time_budget import VO_RATE_NATURAL_FLOOR

#: The multiplier ``tts_backend`` applies to a ``<emphasis level="strong">`` span.
#: ``tts_backend/src/services/ssml_processor.py`` ``_EMPHASIS_SPEED["strong"]`` — the
#: backend renders each emphasised span as its OWN kokoro utterance at
#: ``request_speed * 0.88``, and clamps only the PRODUCT, at kokoro's 0.5 assertion
#: floor. Mirrored here (not imported — it lives in another repo) so this module can
#: reason about the rate the listener will actually hear.
_EMPHASIS_STRONG_SPEED = 0.88

#: The lowest resolved ``vo_rate`` a line may carry and still be emphasised.
#:
#: DERIVED, not tuned: ``VO_RATE_NATURAL_FLOOR`` is the rate below which the voice
#: stops sounding like narration, and the whole fit solver refuses to go under it. An
#: emphasised span is delivered at ``vo_rate * _EMPHASIS_STRONG_SPEED``, so emphasis on
#: a line already at the floor ships a mid-sentence span at 0.80 x 0.88 = 0.704 — below
#: the floor the rest of the line is held to, in a separate utterance with its own pitch
#: contour and its own join. That is heard as the narration "weirdly slowing down"
#: inside a sentence. Emphasis is therefore allowed only while the emphasised span
#: ITSELF still lands at or above the natural floor.
EMPHASIS_MIN_VO_RATE = VO_RATE_NATURAL_FLOOR / _EMPHASIS_STRONG_SPEED


def emphasis_affordable(vo_rate: float | None) -> bool:
    """Can this line afford a further ``_EMPHASIS_STRONG_SPEED`` drop on a word?

    ``None`` (no resolved rate — no ``vo_rate``, no authored rate, no directive hint)
    means "the backend default, 1.0", which affords it: that is the pre-existing path
    and must stay byte-identical. A non-finite rate is treated the same way rather than
    silently suppressing direction on a coercion bug.
    """
    if vo_rate is None:
        return True
    try:
        rate = float(vo_rate)
    except TypeError, ValueError:
        return True
    if rate != rate:  # NaN — no usable rate, keep today's behaviour
        return True
    return rate >= EMPHASIS_MIN_VO_RATE


def build_line_ssml(
    text: str,
    pause_before_ms: int,
    emphasis_indices: list[int],
    vo_rate: float | None = None,
) -> str | None:
    """Render one narration line to SSML, or ``None`` for the plain-text path.

    Args:
        text: The narration line. Tokenized with ``text.split()`` to match the
            positions ``voice_director`` used when computing ``emphasis_indices``.
        pause_before_ms: Leading pause in milliseconds. ``<= 0`` means no pause.
        emphasis_indices: ``text.split()`` word positions to wrap in
            ``<emphasis level="strong">``. Indices outside ``0 <= i < len(tokens)``
            are ignored.
        vo_rate: This line's resolved TTS rate (``speed_for``'s return value), or
            ``None`` when the line carries no resolved rate (backend default,
            1.0). A beat already fit to a rate that cannot afford a further
            ``_EMPHASIS_STRONG_SPEED`` drop (see :func:`emphasis_affordable`) gets
            NO emphasis at all rather than a mid-sentence rate cliff — bounding
            the defect at the source, belt-and-braces with the backend-side clamp.

    Returns:
        A ``<speak>…</speak>`` string, or ``None`` when there is nothing to
        express (no pause and no emphasis) — the caller then uses plain ``text``.
    """
    emphasized = set(emphasis_indices) if emphasis_affordable(vo_rate) else set()
    if pause_before_ms <= 0 and not emphasized:
        return None

    tokens = text.split()

    rendered: list[str] = []
    for i, token in enumerate(tokens):
        escaped = escape(token)
        if i in emphasized:
            escaped = f'<emphasis level="strong">{escaped}</emphasis>'
        rendered.append(escaped)

    body = " ".join(rendered)
    if pause_before_ms > 0:
        prefix = f'<break time="{pause_before_ms}ms"/>'
        body = f"{prefix}{body}" if body else prefix

    return f"<speak>{body}</speak>"
