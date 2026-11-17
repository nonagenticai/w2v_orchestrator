"""Kokoro voice catalog — the authoritative set of voice ids the live TTS
backend accepts, plus prompt-facing descriptions for the cast stage.

The list mirrors ``GET /api/tts/voices`` on ``tts.test:9202`` (kokoro
v1.0, 54 voices) as observed live. ``audiobook_cast`` validates every
LLM-emitted voice id against ``VOICE_CATALOG`` before it can reach
``/api/tts/synthesize`` — the backend 500s on an unknown voice
(see ``narrate.py`` / ``narration.py``), so an unvalidated id is a run-fatal
bug, not a cosmetic one.

First cut: this is a static frozenset (presets-only, no blending). A live
``TTSClient.voices()`` fetch is a deliberate follow-up, not this change.
"""

from __future__ import annotations

from src.utils import coerce_float

#: The kokoro / TTS-gateway speed range (gateway ``ChunkIn`` 422s outside it).
SPEED_MIN, SPEED_MAX = 0.5, 2.0


def clamp_speed(value: object, default: float) -> float:
    """``value`` as a speed inside [``SPEED_MIN``, ``SPEED_MAX``]; ``default`` if unusable."""
    speed = coerce_float(value)
    if speed is None:
        return default
    return max(SPEED_MIN, min(SPEED_MAX, speed))


# kokoro id prefix → (language, gender) for human-readable casting hints.
# af/am = American English, bf/bm = British English, then other languages.
_PREFIX_META: dict[str, tuple[str, str]] = {
    "af": ("American English", "female"),
    "am": ("American English", "male"),
    "bf": ("British English", "female"),
    "bm": ("British English", "male"),
    "ef": ("Spanish", "female"),
    "em": ("Spanish", "male"),
    "ff": ("French", "female"),
    "hf": ("Hindi", "female"),
    "hm": ("Hindi", "male"),
    "if": ("Italian", "female"),
    "im": ("Italian", "male"),
    "jf": ("Japanese", "female"),
    "jm": ("Japanese", "male"),
    "pf": ("Portuguese", "female"),
    "pm": ("Portuguese", "male"),
    "zf": ("Mandarin Chinese", "female"),
    "zm": ("Mandarin Chinese", "male"),
}

# The 54 live kokoro voice ids (sorted). Authoritative validation set.
VOICE_CATALOG: frozenset[str] = frozenset(
    {
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
        "ef_dora",
        "em_alex",
        "em_santa",
        "ff_siwis",
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
        "if_sara",
        "im_nicola",
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
        "pf_dora",
        "pm_alex",
        "pm_santa",
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    }
)


def describe_voice(voice_id: str) -> str:
    """Human-readable '<id> (<language> <gender>)' hint for casting prompts."""
    lang, gender = _PREFIX_META.get(voice_id[:2], ("English", "unspecified"))
    return f"{voice_id} ({lang} {gender})"


def voice_catalog_prompt_block() -> str:
    """A compact, language-grouped listing of the allowed voices for the LLM."""
    by_lang: dict[str, list[str]] = {}
    for vid in sorted(VOICE_CATALOG):
        lang, gender = _PREFIX_META.get(vid[:2], ("English", "unspecified"))
        by_lang.setdefault(lang, []).append(f"{vid} ({gender})")
    return "\n".join(f"- {lang}: {', '.join(ids)}" for lang, ids in by_lang.items())
