"""Typed responses for the TTS synthesis seam (tts_backend, port 6194 / mini2 :9202).

Two models, deliberately distinct:

- ``SynthesizeWire`` mirrors the raw ``SynthesizeResponse`` JSON tts_backend
  returns when called with ``Accept: application/json`` (base64 WAV + duration
  metadata). ``extra="ignore"`` (pydantic's default) drops any field the
  orchestrator does not read.
- ``TTSSynthesis`` is the CLIENT's OWN transformed return: the base64 is decoded
  to raw ``audio_bytes`` and the measured post-premaster loudness
  (``integrated_lufs``, read from the ``X-TTS-LUFS`` header — not on the wire)
  is folded in. Consumers read this model, never the wire model.

``speech_seconds`` / ``pause_seconds`` (A/V quality wave 8, plan item 2) are the
producer's OWN field names — ``tts_backend``'s ``SynthesizeResponse`` defines
them and this seam consumes them verbatim. They report how the returned clip
splits::

    duration_sec == speech_seconds + pause_seconds

``pause_seconds`` is the silence tts_backend deliberately spliced in — the
inter-piece pacing gap between sentences/clauses PLUS any authored SSML
``<break>``. It is FIXED: rendered as literal zeros, never scaled by ``speed``.
⚠️ Because the authored break is ALREADY INSIDE it, this value is a TOTAL and
must never be ADDED to the orchestrator's own ``pause_seconds_for`` — see
``narration._fit_speed``, which takes the larger of the two, never the sum.

Both are OPTIONAL here (``float | None``) even though the producer declares them
with a ``0.0`` default, because a tts_backend deployed before the companion
change omits them entirely. ``None`` therefore means "this backend does not
report the split — fall back to the authored-break-only behaviour", and MUST
NOT be read as "zero silence": zero would tell ``_fit_speed`` the whole clip is
compressible speech, which is exactly the bug being fixed.
"""

from __future__ import annotations

from pydantic import BaseModel


class WordTiming(BaseModel):
    """One word's span in the FINAL audio timebase, as tts_backend measured it.

    Emitted by the PyTorch kokoro engine, whose duration predictor gives true
    per-word ``start``/``end`` — the ONNX engine returns audio only and cannot
    produce these, so the field is optional and absent on that path.
    """

    word: str
    start: float
    end: float


class SynthesizeWire(BaseModel):
    """Raw JSON from tts_backend ``POST /api/tts/synthesize`` (Accept: application/json)."""

    audio_base64: str
    sample_rate: int
    duration_sec: float
    provider: str
    voice: str
    format: str = "wav"
    # Absent whenever tts_backend is on the ONNX engine (which structurally cannot
    # emit timings) or on a non-kokoro provider. Consumers MUST treat None as
    # "fall back to ASR", never as "this line has no words".
    word_timings: list[WordTiming] | None = None
    # ⚠️ Names fixed by the PRODUCER (tts_backend SynthesizeResponse) — a typo
    # here does not fail, it silently leaves both None forever and the rate fix
    # never connects. None = "backend predates the field", NOT "zero silence".
    speech_seconds: float | None = None
    pause_seconds: float | None = None


class TTSSynthesis(BaseModel):
    """The client's transformed synthesize result (decoded bytes + measured LUFS)."""

    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    provider: str
    voice: str
    format: str = "wav"
    integrated_lufs: float | None = None
    word_timings: list[WordTiming] | None = None
    # See the module docstring — passed through verbatim by TTSClient.synthesize.
    # None means "unknown", not "zero".
    speech_seconds: float | None = None
    pause_seconds: float | None = None
