"""Shared two-voice dialogue helpers (promo + website_review + lesson + demo_video).

The LLM writes Q&A dialogue CONTENT reliably but does NOT always alternate the
per-line ``speaker`` tag — it sometimes emits every line as "a", collapsing the two
voices back into one narrator. So a draft whose tags carry only ONE role has its
speaker DERIVED from each line's role (a question is the asker "a", anything else
the answerer "b"). A draft that DID tag both roles keeps its tags: the model knows
who says "Right, and the form?" — a trailing ``?`` does not (prod run 01a0a054 cast
a reviewer's opening critique, a statement, to the designer's voice).

Whether a run is a dialogue at all is ONE predicate, :func:`dialogue_cast`, shared
by the script prompt, the speaker derivation and the voice casting. Before it
existed the script stage defaulted dialogue ON while ``voice_for_speaker`` cast
only on an explicit flag — so an unset flag produced a two-speaker conversation
read by a single voice.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast

from src.services.stages.audiobook.voice_catalog import VOICE_CATALOG

# Default two-hander cast: a warm American-female asker (af_heart) trading turns
# with a grounded American-male answerer (am_michael). Distinct gender keeps the
# two unmistakably separate by ear. Overridable via ``config.voice.cast``.
DEFAULT_CAST: dict[str, str] = {"a": "af_heart", "b": "am_michael"}

#: The only speaker tags a two-hander carries.
SPEAKERS: tuple[str, str] = ("a", "b")


class _DialogueLine(Protocol):
    text: str

    def model_copy(self, *, update: Mapping[str, object]) -> object: ...


def _tag(line: object) -> str:
    return str(getattr(line, "speaker", "") or "").strip().lower()


def assign_dialogue_speakers[LineT: _DialogueLine](lines: list[LineT]) -> list[LineT]:
    """Each line's ``speaker``: the model's own tags when usable, else derived.

    * Every line already carries a valid tag AND both roles occur → kept verbatim.
    * Otherwise the role heuristic: a question → "a", else "b".
    * Falls back to strict a/b parity when the heuristic can't find both roles (e.g.
      a script with no question marks), so dialogue mode always yields two voices.
    """
    tags = [_tag(line) for line in lines]
    if all(tag in SPEAKERS for tag in tags) and set(tags) == set(SPEAKERS):
        roles = tags
    else:
        roles = ["a" if line.text.rstrip().endswith("?") else "b" for line in lines]
        if "a" not in roles or "b" not in roles:
            roles = ["a" if i % 2 == 0 else "b" for i in range(len(lines))]
    return [
        cast("LineT", line.model_copy(update={"speaker": role}))
        for line, role in zip(lines, roles, strict=True)
    ]


def _named_cast(configured: object) -> dict[str, str] | None:
    """A NAMED-role cast (``{"Mark": voice, "Lisa": voice}``) keyed by lowercased role.

    The wizard writes speaker names, not ``a``/``b`` slots. Only catalog-valid voices
    are kept, and the map counts only when it yields at least two DISTINCT voices —
    otherwise one voice would read every part, which is what this module prevents.
    ``None`` when the cast carries no role beyond ``a``/``b``.
    """
    if not isinstance(configured, Mapping):
        return None
    named = {
        str(role).strip().lower(): str(v or "").strip()
        for role, v in configured.items()
        if str(role).strip() and str(role).strip().lower() not in SPEAKERS
    }
    if not named:
        return None
    valid = {role: v for role, v in named.items() if v in VOICE_CATALOG}
    if len(set(valid.values())) < 2:
        return None
    return valid


def dialogue_cast(
    voice: Mapping[str, object] | None,
    brief: Mapping[str, object] | None,
    *,
    default_cast: Mapping[str, str] = DEFAULT_CAST,
) -> dict[str, str] | None:
    """The speaker → voice cast when this run IS a multi-voice dialogue, else ``None``.

    Two shapes:

    * **a/b two-hander** — ``brief.dialogue.enabled`` explicitly truthy, and two
      DISTINCT catalog voices resolved for "a" and "b" (``config.voice.cast`` over
      ``default_cast``).
    * **named roles** — ``config.voice.cast`` carries roles beyond a/b (the wizard
      writes ``{"Mark": …, "Lisa": …}`` and tags each line ``speaker="Mark"``). Keys
      are lowercased/stripped, and at least two distinct catalog voices are
      required. Naming the cast IS the opt-in, so ``dialogue.enabled`` need not be
      set — but an explicit ``enabled: false`` still wins. Any a/b slots that also
      resolve are kept so derived a/b tags still cast.

    Anything else is a single narrator: one voice reading both parts is the
    defect this predicate exists to prevent.
    """
    dialogue = (brief or {}).get("dialogue")
    enabled = dialogue.get("enabled") if isinstance(dialogue, Mapping) else None
    configured = (voice or {}).get("cast")
    named = _named_cast(configured)
    if named is not None and (enabled is None or enabled):
        configured_ab = configured if isinstance(configured, Mapping) else {}
        for speaker in SPEAKERS:
            v = str(configured_ab.get(speaker) or default_cast.get(speaker) or "")
            if v in VOICE_CATALOG:
                named.setdefault(speaker, v)
        return named
    if not enabled:
        return None
    cast_map = {**default_cast, **(configured if isinstance(configured, Mapping) else {})}
    resolved = {speaker: str(cast_map.get(speaker) or "") for speaker in SPEAKERS}
    if any(v not in VOICE_CATALOG for v in resolved.values()):
        return None
    if resolved["a"] == resolved["b"]:
        return None
    return resolved


def voice_for_speaker(
    item: dict,
    voice: dict,
    brief: dict,
    *,
    default_cast: dict[str, str] = DEFAULT_CAST,
) -> str | None:
    """Cast a dialogue line to its speaker's voice; else the single voice.

    Gated on :func:`dialogue_cast` so a single-narrator run — including a dialogue
    run whose cast cannot yield two distinct voices — is read entirely in
    ``voice.voice_id``. An unknown/blank speaker also drops back to ``voice_id``.
    """
    cast_map = dialogue_cast(voice, brief, default_cast=default_cast)
    if cast_map is None:
        return voice.get("voice_id")
    speaker = str(item.get("speaker") or "").strip().lower()
    return cast_map.get(speaker) or voice.get("voice_id")
