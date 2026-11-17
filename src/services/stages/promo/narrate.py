"""promo_narrate — TTS each script line; collect audio-clip artifact URLs.

One sync ``TTSClient.synthesize`` per line (via the shared NarrationStageBase),
the WAV bytes persisted via ``store_tts_clip`` so ``promo_render`` can reference
a stable ``artifact_url`` keyed by ``line_id``.

When ``promo_script`` ran in two-hander dialogue mode each line carries a
``speaker`` ("a"/"b"); this stage casts each speaker to a distinct kokoro voice
so the promo reads as a real conversation between two voice actors rather than a
single AI narrator. The cast comes from ``config.voice.cast`` (``{"a": id,
"b": id}``) and falls back to a sensible default pair; every id is validated
against the live ``VOICE_CATALOG`` (the TTS backend 500s on an unknown voice).
A line with no speaker (single-narrator promos) keeps ``voice.voice_id``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar
from urllib.parse import urlsplit

from src.services.stages.audiobook.pronunciation import apply_lexicon, build_lexicon
from src.services.stages.audiobook.voice_catalog import VOICE_CATALOG
from src.services.stages.base import (
    ArtifactUrl,
    SlotUse,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
)
from src.services.stages.primitives.narration import NarrationStageBase

if TYPE_CHECKING:
    from src.services.runs.run_context import RunContext

# Default two-hander cast: a warm American-female host/skeptic (af_heart) trading
# turns with a grounded American-male guide (am_michael). Distinct gender keeps
# the two unmistakably separate by ear. Overridable via config.voice.cast.
_DEFAULT_CAST: dict[str, str] = {"a": "af_heart", "b": "am_michael"}


class PromoNarrateInput(StageInput):
    pass


class PromoAudioClip(StageOutput):
    id: str
    line_id: str
    artifact_url: ArtifactUrl
    seconds: float
    # Speed the kept clip was synthesized at (None = backend default) and its
    # spoken word count, filled by NarrationStageBase. Optional so outputs
    # persisted before these fields existed still validate.
    speed: float | None = None
    words: int | None = None


class PromoNarrateOutput(StageOutputWithNotes):
    # Same spine as the generic narrate: `shot_assembly` keys audio off
    # `clip["line_id"]`, `promo_footage_direction`/`_critique` take their per-line
    # windows from clip `seconds`, and `captions.py` builds the SRT from it.
    # Zero clips ships a SILENT promo that still reports completed.
    produces: ClassVar[tuple[str, ...]] = ("clips",)

    clips: list[PromoAudioClip]


# How a "." inside a name is said, by narration language (ISO 639-1 prefix).
_DOT_WORD = {"en": "dot", "nl": "punt", "de": "Punkt", "fr": "point", "es": "punto"}
# Domain labels said as a word; any other trailing label of <= 3 letters is spelled.
_SPOKEN_LABELS = frozenset({"com", "org", "net", "app", "dev", "info", "shop"})
# Well-known file names that turn up in marketing copy. A closed list on purpose:
# a generic ``word.ext`` rule cannot tell ``robots.txt`` from "daily.no setup".
_SPOKEN_FILES = ("robots.txt", "sitemap.xml", "llms.txt", "security.txt")


def _spell(label: str) -> str:
    return " ".join(label.upper())


def _speak_host(host: str, dot: str) -> str:
    """``app.merqi.co.uk`` -> ``app dot merqi dot C O dot U K``."""
    labels = host.split(".")
    spelled: set[int] = set()
    for index in range(len(labels) - 1, 0, -1):  # trailing TLD labels, never the first
        label = labels[index].lower()
        if len(label) > 3 or label in _SPOKEN_LABELS:
            break
        spelled.add(index)
    return f" {dot} ".join(
        _spell(label) if index in spelled or label.lower() == "www" else label
        for index, label in enumerate(labels)
    )


def brand_domains(config_snapshot: Mapping[str, object]) -> tuple[str, ...]:
    """The brand's own registrable hosts, from the URLs the brief names."""
    brief = config_snapshot.get("brief")
    brief = brief if isinstance(brief, Mapping) else {}
    hosts: list[str] = []
    for block in (brief, brief.get("intent"), config_snapshot.get("intent")):
        if not isinstance(block, Mapping):
            continue
        for key in ("url", "source_url", "website"):
            raw = block.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            host = (urlsplit(raw if "//" in raw else f"//{raw}").hostname or "").lower()
            host = host.removeprefix("www.")
            if "." in host and host not in hosts:
                hosts.append(host)
    return tuple(hosts)


def narration_language(config_snapshot: Mapping[str, object]) -> str:
    """The narration language's ISO prefix (``en`` when the run names none)."""
    brief = config_snapshot.get("brief")
    voice = config_snapshot.get("voice")
    for block in (voice, brief):
        raw = block.get("language") if isinstance(block, Mapping) else None
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()[:2]
    return "en"


def spoken_text(
    text: str,
    pronunciations: dict[str, str] | None = None,
    *,
    domains: Sequence[str] = (),
    language: str = "en",
) -> str:
    """``text`` rewritten for the voice. The caption keeps the script's text.

    Run 01a10fcf (2026-10-06): the TTS engine has no rule for an inner ``.``, so
    "Request access at merqi.nl" was spoken "merqi, N L" (no "dot") and
    "robots.txt" broke into "robots." + a sentence pause + "TXT".

    Three rewrites, each on names the run KNOWS — there is deliberately no
    generic ``word.suffix`` rule (it turned "daily.no setup" into a domain):

    * ``brief.pronunciations`` (``{written: spoken}``, whole-word,
      case-insensitive — the audiobook lexicon), so an author can respell the
      brand (``"merqi": "MER-kee"``) or a whole dotted name;
    * the brand's own ``domains`` (and their subdomains): every ``.`` becomes the
      language's word for it and the short TLD labels are spelled out;
    * the well-known file names in :data:`_SPOKEN_FILES`.
    """
    dot = _DOT_WORD.get(language, _DOT_WORD["en"])
    lexicon = build_lexicon(pronunciations)
    # A respelling of a whole dotted name wins over the rules below; a respelling
    # of a plain word (the brand) is applied AFTER them, so it also lands inside
    # the spoken domain ("MER-kee dot N L").
    text = apply_lexicon(text, {k: v for k, v in lexicon.items() if "." in k})
    for domain in sorted({d.lower() for d in domains if "." in d}, key=len, reverse=True):
        pattern = re.compile(
            r"(?<![\w@/.-])((?:[A-Za-z0-9-]+\.)*)" + re.escape(domain) + r"(?![\w/-]|\.\w)",
            re.IGNORECASE,
        )
        text = pattern.sub(lambda m: _speak_host(m.group(0), dot), text)
    for name in _SPOKEN_FILES:
        stem, _, ext = name.partition(".")
        text = re.sub(
            r"(?<![\w@/.-])(" + re.escape(stem) + r")\." + ext + r"(?![\w/-]|\.\w)",
            lambda m, ext=ext: f"{m.group(1)} {dot} {_spell(ext)}",
            text,
            flags=re.IGNORECASE,
        )
    return apply_lexicon(text, {k: v for k, v in lexicon.items() if "." not in k})


class PromoNarrateStage(NarrationStageBase[PromoNarrateInput, PromoNarrateOutput]):
    name = "promo_narrate"
    input_model = PromoNarrateInput
    output_model = PromoNarrateOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (SlotUse("tts"),)
    source_stage = "promo_script"
    source_list_key = "lines"
    source_id_field = "line_id"
    clip_model = PromoAudioClip

    def synth_text_for(self, item: dict, ctx: RunContext) -> tuple[str, bool]:
        """The line as it must be SPOKEN — the caption keeps the script's text.

        Captions read ``promo_script.lines[].text`` directly, so this rewrite
        never reaches the screen. See :func:`spoken_text`.
        """
        brief = ctx.config_snapshot.get("brief") or {}
        pronunciations = brief.get("pronunciations") if isinstance(brief, dict) else None
        return spoken_text(
            str(item["text"]),
            pronunciations if isinstance(pronunciations, dict) else None,
            domains=brand_domains(ctx.config_snapshot),
            language=narration_language(ctx.config_snapshot),
        ), False

    def voice_for(self, item: dict, voice: dict, ctx: RunContext) -> str | None:
        """Cast a dialogue line to its speaker's voice; else the single voice.

        Casting is gated on ``brief.dialogue.enabled`` so single-narrator promos
        are unaffected regardless of any ``speaker`` the model happened to emit.
        When enabled, resolution for a line is ``config.voice.cast[speaker]`` →
        ``_DEFAULT_CAST[speaker]``. Any resolved id absent from ``VOICE_CATALOG``
        (or an unknown/blank speaker) drops back to ``voice.voice_id`` so a
        typo'd cast can't 500 the whole synth — the line still narrates, just in
        the default voice.
        """
        brief = ctx.config_snapshot.get("brief") or {}
        if not (brief.get("dialogue") or {}).get("enabled"):
            return voice.get("voice_id")
        speaker = str(item.get("speaker") or "").strip().lower()
        cast = {**_DEFAULT_CAST, **(voice.get("cast") or {})}
        candidate = cast.get(speaker)
        if candidate and candidate in VOICE_CATALOG:
            return candidate
        return voice.get("voice_id")
