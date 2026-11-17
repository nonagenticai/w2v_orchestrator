"""Live connectivity test for a tenant provider connection (P3 WP4).

``PUT /api/tenant/connections/{slot}`` tests a connection BEFORE persisting it and
``POST …/{slot}/test`` re-tests a draft or the current version. The route codes
against :class:`ProviderTester` so tests inject a fake; :class:`HttpProviderTester`
is the production implementation:

* ``llm`` ``openai-compatible`` — ``GET {base_url}/models``; a listing that names the
  configured model passes without further requests. A listing that does NOT name it
  (Azure deployment names, Gemini's ``models/`` prefix, Ollama's ``:latest``,
  paginated listings) or a server that cannot list (404/405/501, or a 2xx body that
  is not a listing) falls through to a 1-token ``POST {base_url}/chat/completions``
  against the configured model — only that call's failure fails the test.
* ``llm`` ``anthropic`` — a minimal ``POST /v1/messages`` with ``max_tokens=1``.
* ``tts`` / ``image`` ``openai``, ``video_gen`` ``sora`` — ``GET {base_url}/models``
  (bearer). Auth-only, nothing is generated.
* ``tts`` ``openai-compatible`` — ``GET {base_url}/models``; when that is 404 (many
  self-hosted speech servers implement only ``/audio/speech``) fall back to a
  minimal ``POST {base_url}/audio/speech`` speaking the word "test". A 404 on both is
  a failure, never a pass.
* ``tts`` ``elevenlabs`` — ``GET /v1/user`` (``xi-api-key``).
* ``tts`` ``nonagentic_endpoint`` — a one-word ``POST {base_url}/api/tts/synthesize``
  (the platform body, ``Accept: application/json``, bearer when set).
* ``asr`` ``nonagentic_endpoint`` — an AUTHENTICATED ``POST {base_url}/api/asr/transcribe``
  whose multipart body deliberately has NO ``file`` part, so the server rejects it
  before any work (see "authenticated probes" below).
* ``asr`` ``openai-compatible`` — ``GET {base_url}/models``; on 404/405/501 the same
  silent WAV goes to ``{base_url}/audio/transcriptions``.
* ``image`` ``nonagentic_endpoint`` — an AUTHENTICATED ``POST {base_url}/api/image/generate``
  with ``{}`` (no ``prompt``), rejected by request validation before any generation.
* ``vlm`` ``openai-compatible`` — the ``llm`` probe (``/models``, else a 1-token chat),
  whose chat fallback carries a 1x1 PNG ``image_url`` part so it proves vision.
* ``video_gen`` ``veo`` — ``GET {base_url}/models`` (``x-goog-api-key``, Gemini API).
* ``video_gen`` ``runway`` / ``kling`` / ``seedance`` — **not probed**: none has an
  auth-only endpoint we can call without generating (and paying for) a video.
  The shape is validated by the route; the result says ``probed=False`` so the
  connection is stored with an honest "not probed" ``last_test``, never a pass.
* ``device_farm`` ``browserstack`` — ``GET https://api.browserstack.com/automate/plan.json``
  with HTTP basic auth ``options.username:secret``.

Every request goes through :func:`w2v_common.net.safe_request`: the host is resolved
once and the connection pinned to the validated IP, and redirects are followed
manually with each hop re-validated, so a public endpoint cannot bounce the test
(and the tenant's key) into the estate. ``hosted`` mode passes NO allow-list;
``private_ok`` passes the operator egress allow-list — the same rule as
:func:`~src.services.providers.policy.check_egress`.

Authenticated probes (``asr`` / ``image`` ``nonagentic_endpoint``): ``GET /health`` is
unauthenticated, so a wrong Bearer token passed it. Instead the real route is called
with the token and a body it must refuse: 401/403 ⇒ "token rejected" (fail), 400/422
⇒ the token got past auth and only the deliberately empty body was refused (pass),
2xx ⇒ pass, 404 ⇒ wrong base URL (fail), 5xx ⇒ endpoint unavailable (fail). Auth is
checked before body validation on the platform servers (gateway middleware /
``Depends`` run before FastAPI body parsing), so a bad token cannot 422.

The whole test has one 10 s deadline. The result message never carries the secret,
URL userinfo/query or the upstream body (an upstream 401 body can echo part of the
key): only the redacted endpoint, the HTTP status and a fixed explanation.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol

import httpx
from w2v_common.net import ResponseTooLargeError, SSRFError, safe_request

from src.services.providers.policy import redact_url
from src.services.providers.registry import (
    ASR_DEFAULT_MODEL,
    KIND_ANTHROPIC,
    KIND_BROWSERSTACK,
    KIND_ELEVENLABS,
    KIND_KLING,
    KIND_NONAGENTIC_ENDPOINT,
    KIND_OPENAI,
    KIND_OPENAI_COMPATIBLE,
    KIND_RUNWAY,
    KIND_SEEDANCE,
    KIND_SORA,
    KIND_VEO,
)
from src.services.providers.settings import ProviderPolicySettings
from src.services.providers.tts_inprocess import (
    OPENAI_DEFAULT_MODEL as OPENAI_TTS_DEFAULT_MODEL,
)
from src.services.providers.tts_inprocess import (
    OPENAI_DEFAULT_VOICE as OPENAI_TTS_DEFAULT_VOICE,
)

#: Whole-test deadline (the /models → chat fallback shares it).
TEST_TIMEOUT_SECONDS: Final = 10.0
#: A test response is tiny; never buffer more than this.
_MAX_TEST_RESPONSE_BYTES: Final = 1024 * 1024
ANTHROPIC_DEFAULT_BASE_URL: Final = "https://api.anthropic.com"
ANTHROPIC_VERSION: Final = "2023-06-01"
OPENAI_DEFAULT_BASE_URL: Final = "https://api.openai.com/v1"
ELEVENLABS_DEFAULT_BASE_URL: Final = "https://api.elevenlabs.io"
VEO_DEFAULT_BASE_URL: Final = "https://generativelanguage.googleapis.com/v1beta"
#: Fixed: a BrowserStack credential is only ever sent to BrowserStack.
BROWSERSTACK_PLAN_URL: Final = "https://api.browserstack.com/automate/plan.json"
#: Kinds with no auth-only endpoint — stored as "not probed", never as a pass.
UNPROBED_KINDS: Final = frozenset({KIND_RUNWAY, KIND_KLING, KIND_SEEDANCE})
_PING: Final[list[dict[str, object]]] = [{"role": "user", "content": "ping"}]
#: A 1x1 transparent PNG — the image part of the ``vlm`` chat probe.
_PIXEL_PNG: Final = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII="
)
_VISION_PING: Final[list[dict[str, object]]] = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "ping"},
            {"type": "image_url", "image_url": {"url": _PIXEL_PNG}},
        ],
    }
]
_MULTIPART_BOUNDARY: Final = "w2v-provider-test-boundary"


def silent_wav(seconds: float = 0.25, rate: int = 16_000) -> bytes:
    """A tiny mono 16-bit PCM WAV of silence (the ASR probe's audio)."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def multipart_body(fields: Mapping[str, str], file_bytes: bytes | None) -> tuple[bytes, str]:
    """``multipart/form-data`` with ``fields`` + one ``file`` part → (body, content type).

    ``file_bytes=None`` omits the ``file`` part (the authenticated ASR probe).
    """
    b = _MULTIPART_BOUNDARY
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{b}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    if file_bytes is not None:
        parts.append(
            f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="probe.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n".encode()
            + file_bytes
            + b"\r\n"
        )
    parts.append(f"--{b}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={b}"


@dataclass(frozen=True, slots=True)
class ProviderTestResult:
    """Outcome of one live test. ``message`` is safe to show and to persist."""

    ok: bool
    message: str
    upstream_status: int | None = None
    #: ``False`` when no request was made (kind has no auth-only endpoint): ``ok`` then
    #: only means "the shape is acceptable", and the stored ``last_test`` says so.
    probed: bool = True


@dataclass(frozen=True, slots=True)
class _Plan:
    """The one request that proves a credential (``mode`` picks the shape)."""

    mode: str
    endpoint: str
    headers: Mapping[str, str]
    #: The voice a synthesize-fallback probe speaks with (tts only).
    voice: str | None = None


class ProviderTester(Protocol):
    async def test(
        self,
        *,
        slot: str,
        kind: str,
        base_url: str | None,
        model: str | None,
        options: Mapping[str, object],
        secret: str | None,
    ) -> ProviderTestResult: ...


class ModelLister(Protocol):
    """``POST /api/tenant/connections/{slot}/models`` — a provider's model listing."""

    async def list_models(
        self, *, slot: str, kind: str, base_url: str | None, secret: str | None
    ) -> ModelListResult: ...


@dataclass(frozen=True, slots=True)
class ModelListResult:
    """``models`` on success; otherwise ``None`` and a ``message`` safe to show
    (redacted endpoint + status only — never the secret or the upstream body)."""

    models: list[str] | None
    message: str | None = None
    upstream_status: int | None = None


#: Model ids returned at most (a listing is a picker, not a dump).
MAX_LISTED_MODELS: Final = 500


def _join(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


#: ``GET /models`` statuses meaning "this server does not implement model listing".
_MODELS_UNSUPPORTED_STATUSES: Final = frozenset({404, 405, 501})
_MAX_MODEL_NAME_IN_MESSAGE: Final = 120


def _safe_model(model: str) -> str:
    return model[:_MAX_MODEL_NAME_IN_MESSAGE]


def _entry_ids(entries: list[object], id_keys: tuple[str, ...]) -> frozenset[str]:
    ids: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            ids.add(entry)
        elif isinstance(entry, dict):
            ids.update(v for k in id_keys if isinstance(v := entry.get(k), str))
    return frozenset(ids)


def _listed_model_ids(
    resp: httpx.Response, *, id_keys: tuple[str, ...] = ("id", "name")
) -> frozenset[str] | None:
    """Model ids from an OpenAI-style listing, or ``None`` if the body is not one.

    OpenAI shape is ``{"data": [{"id": ...}]}``; some compatible servers answer
    ``{"models": [{"id"|"name": ...}]}``, ElevenLabs a bare list of
    ``{"model_id": ...}`` (``id_keys=("model_id",)``). Anything else (non-JSON, no
    list) is "cannot list" — the caller then proves the model with a chat completion.
    """
    try:
        body = resp.json()
    except ValueError:
        return None
    if isinstance(body, list):
        entries: object = body
    elif isinstance(body, dict):
        entries = body.get("data")
        if not isinstance(entries, list):
            entries = body.get("models")
    else:
        return None
    if not isinstance(entries, list):
        return None
    return _entry_ids(entries, id_keys)


def _status_message(endpoint: str, status: int) -> str:
    hint = {
        400: "the request was rejected (check the model name)",
        401: "the credential was rejected",
        403: "the credential is not permitted to use this endpoint",
        404: "endpoint or model not found",
        429: "rate limited by the provider",
    }.get(status, "the provider returned an error" if status >= 400 else "unexpected response")
    return f"{endpoint} answered HTTP {status}: {hint}"


#: (slot, kind) → (request mode, default base URL or None = required, auth header).
#: ``Authorization`` is sent as ``Bearer <secret>``; any other header carries the raw key.
_MODALITY_PROBES: Final[dict[tuple[str, str], tuple[str, str | None, str]]] = {
    ("tts", KIND_OPENAI): ("models", OPENAI_DEFAULT_BASE_URL, "Authorization"),
    ("tts", KIND_OPENAI_COMPATIBLE): ("tts-openai-compatible", None, "Authorization"),
    ("tts", KIND_ELEVENLABS): ("elevenlabs-user", ELEVENLABS_DEFAULT_BASE_URL, "xi-api-key"),
    ("tts", KIND_NONAGENTIC_ENDPOINT): ("nonagentic-tts", None, "Authorization"),
    ("asr", KIND_NONAGENTIC_ENDPOINT): ("nonagentic-asr", None, "Authorization"),
    ("asr", KIND_OPENAI_COMPATIBLE): ("asr-openai-compatible", None, "Authorization"),
    ("image", KIND_OPENAI): ("models", OPENAI_DEFAULT_BASE_URL, "Authorization"),
    ("image", KIND_NONAGENTIC_ENDPOINT): ("nonagentic-image", None, "Authorization"),
    ("image", KIND_OPENAI_COMPATIBLE): ("models", None, "Authorization"),
    ("video_gen", KIND_SORA): ("models", OPENAI_DEFAULT_BASE_URL, "Authorization"),
    ("video_gen", KIND_VEO): ("models", VEO_DEFAULT_BASE_URL, "x-goog-api-key"),
    # OCR (Mistral-OCR wire): the authenticated model listing proves the token —
    # ``{base}/models`` for a /v1 base URL, ``{base}/v1/models`` for a platform copy.
    ("ocr", KIND_OPENAI_COMPATIBLE): ("models", None, "Authorization"),
    ("ocr", KIND_NONAGENTIC_ENDPOINT): ("models-v1", None, "Authorization"),
}


def _llm_plan(kind: str, base_url: str | None, model: str | None) -> _Plan | ProviderTestResult:
    if kind == KIND_OPENAI_COMPATIBLE:
        if not base_url:
            return ProviderTestResult(False, "an OpenAI-compatible provider needs a base URL")
        return _Plan("llm-openai", base_url, {})
    if kind == KIND_ANTHROPIC:
        if not model:
            return ProviderTestResult(False, "an Anthropic connection needs a model")
        return _Plan("llm-anthropic", base_url or ANTHROPIC_DEFAULT_BASE_URL, {})
    return ProviderTestResult(False, f"kind {kind!r} cannot be tested")


def _browserstack_plan(
    options: Mapping[str, object], secret: str | None
) -> _Plan | ProviderTestResult:
    username = options.get("username")
    if not isinstance(username, str) or not username or not secret:
        return ProviderTestResult(
            False, "a BrowserStack connection needs options.username and an access key"
        )
    token = base64.b64encode(f"{username}:{secret}".encode()).decode("ascii")
    return _Plan("get", BROWSERSTACK_PLAN_URL, {"Authorization": f"Basic {token}"})


def _models_plan(
    kind: str, base_url: str | None, secret: str | None
) -> tuple[str, dict[str, str]] | ModelListResult:
    """``(listing URL, auth headers)`` for ``kind``, or why it cannot be listed."""
    if kind == KIND_ANTHROPIC:
        base = (base_url or ANTHROPIC_DEFAULT_BASE_URL).rstrip("/")
        url = base + "/models" if base.endswith("/v1") else base + "/v1/models"
        headers = {"anthropic-version": ANTHROPIC_VERSION}
        if secret:
            headers["x-api-key"] = secret
        return url, headers
    if kind == KIND_ELEVENLABS:
        base = (base_url or ELEVENLABS_DEFAULT_BASE_URL).rstrip("/")
        url = base + "/models" if base.endswith("/v1") else base + "/v1/models"
        return url, ({"xi-api-key": secret} if secret else {})
    if kind in (KIND_OPENAI, KIND_OPENAI_COMPATIBLE, KIND_NONAGENTIC_ENDPOINT):
        base = base_url or (OPENAI_DEFAULT_BASE_URL if kind == KIND_OPENAI else None)
        if base is None:
            return ModelListResult(None, f"a {kind} provider needs a base URL")
        return _join(base, "models"), ({"Authorization": f"Bearer {secret}"} if secret else {})
    return ModelListResult(None, f"kind {kind!r} has no model listing")


#: Probe modes served by :meth:`HttpProviderTester._gpu_probe`.
_GPU_MODES: Final = frozenset(
    {"nonagentic-tts", "nonagentic-asr", "asr-openai-compatible", "nonagentic-image"}
)


class HttpProviderTester:
    """:class:`ProviderTester` over SSRF-safe httpx (see module doc)."""

    def __init__(
        self,
        policy: ProviderPolicySettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = TEST_TIMEOUT_SECONDS,
    ) -> None:
        self._policy = policy
        self._transport = transport
        self._timeout = timeout

    async def list_models(
        self, *, slot: str, kind: str, base_url: str | None, secret: str | None
    ) -> ModelListResult:
        """``GET {base}/models`` with the kind's auth header (module doc), same SSRF
        pinning and deadline as :meth:`test`. Kinds: ``openai-compatible`` /
        ``openai`` / ``nonagentic_endpoint`` (Bearer), ``anthropic``
        (``x-api-key`` + ``anthropic-version``, ``/v1/models``), ``elevenlabs``
        (``xi-api-key``, ``/v1/models``, ``model_id``)."""
        plan = _models_plan(kind, base_url, secret)
        if isinstance(plan, ModelListResult):
            return plan
        url, headers = plan
        safe_endpoint = redact_url(url)
        try:
            async with (
                asyncio.timeout(self._timeout),
                httpx.AsyncClient(
                    timeout=self._timeout, follow_redirects=False, transport=self._transport
                ) as client,
            ):
                resp = await safe_request(
                    client,
                    "GET",
                    url,
                    headers=headers,
                    allow_hosts=self._allow_hosts(),
                    max_fetch_bytes=_MAX_TEST_RESPONSE_BYTES,
                )
        except TimeoutError:
            return ModelListResult(
                None, f"{safe_endpoint} did not answer within {self._timeout:g}s"
            )
        except SSRFError:
            return ModelListResult(None, f"{safe_endpoint} is refused by egress policy")
        except ResponseTooLargeError:
            return ModelListResult(None, f"{safe_endpoint} sent an oversized response")
        except (httpx.HTTPError, ValueError) as exc:
            return ModelListResult(None, f"could not reach {safe_endpoint} ({type(exc).__name__})")
        if not 200 <= resp.status_code < 300:
            return ModelListResult(
                None, _status_message(safe_endpoint, resp.status_code), resp.status_code
            )
        listed = _listed_model_ids(
            resp, id_keys=("model_id",) if kind == KIND_ELEVENLABS else ("id", "name")
        )
        if listed is None:
            return ModelListResult(
                None, f"{safe_endpoint} did not return a model listing", resp.status_code
            )
        return ModelListResult(sorted(listed)[:MAX_LISTED_MODELS], None, resp.status_code)

    def _allow_hosts(self) -> frozenset[str] | None:
        return self._policy.egress_allowlist if self._policy.ssrf_mode == "private_ok" else None

    async def test(
        self,
        *,
        slot: str,
        kind: str,
        base_url: str | None,
        model: str | None,
        options: Mapping[str, object],
        secret: str | None,
    ) -> ProviderTestResult:
        if kind in UNPROBED_KINDS:
            return ProviderTestResult(
                True,
                f"not probed: {kind} has no auth-only endpoint, so the credential was not "
                "verified (configuration shape only); the first generation will verify it",
                probed=False,
            )
        plan = self._plan(slot, kind, base_url, model, options, secret)
        if isinstance(plan, ProviderTestResult):
            return plan

        safe_endpoint = redact_url(plan.endpoint)
        try:
            async with asyncio.timeout(self._timeout):
                return await self._run(plan, model, secret, safe_endpoint)
        except TimeoutError:
            return ProviderTestResult(
                False, f"{safe_endpoint} did not answer within {self._timeout:g}s"
            )
        except SSRFError:
            # The message quotes the raw URL (may carry userinfo) — never relay it.
            return ProviderTestResult(False, f"{safe_endpoint} is refused by egress policy")
        except ResponseTooLargeError:
            return ProviderTestResult(False, f"{safe_endpoint} sent an oversized response")
        except (httpx.HTTPError, ValueError) as exc:
            return ProviderTestResult(
                False, f"could not reach {safe_endpoint} ({type(exc).__name__})"
            )

    @staticmethod
    def _plan(
        slot: str,
        kind: str,
        base_url: str | None,
        model: str | None,
        options: Mapping[str, object],
        secret: str | None,
    ) -> _Plan | ProviderTestResult:
        """Which request proves the credential — or the reason it cannot be tested."""
        if slot == "llm":
            return _llm_plan(kind, base_url, model)
        if slot == "vlm":
            if kind != KIND_OPENAI_COMPATIBLE:
                return ProviderTestResult(False, f"kind {kind!r} cannot be tested for 'vlm'")
            if not base_url:
                return ProviderTestResult(False, "an OpenAI-compatible provider needs a base URL")
            return _Plan("vlm-openai", base_url, {})
        if kind == KIND_BROWSERSTACK and slot == "device_farm":
            return _browserstack_plan(options, secret)
        route = _MODALITY_PROBES.get((slot, kind))
        if route is None:
            return ProviderTestResult(False, f"kind {kind!r} cannot be tested for {slot!r}")
        mode, default_base, header = route
        endpoint = base_url or default_base
        if endpoint is None:
            return ProviderTestResult(False, f"a {kind} provider needs a base URL")
        headers = (
            {}
            if not secret
            else {header: f"Bearer {secret}" if header == "Authorization" else secret}
        )
        voice = options.get("voice")
        return _Plan(mode, endpoint, headers, voice if isinstance(voice, str) else None)

    async def _run(
        self,
        plan: _Plan,
        model: str | None,
        secret: str | None,
        safe_endpoint: str,
    ) -> ProviderTestResult:
        allow = self._allow_hosts()
        endpoint = plan.endpoint
        async with httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=False, transport=self._transport
        ) as client:
            if plan.mode == "llm-anthropic":
                base = endpoint.rstrip("/")
                url = base + "/messages" if base.endswith("/v1") else base + "/v1/messages"
                headers = {"anthropic-version": ANTHROPIC_VERSION}
                if secret:
                    headers["x-api-key"] = secret
                resp = await safe_request(
                    client,
                    "POST",
                    url,
                    json={"model": model, "max_tokens": 1, "messages": _PING},
                    headers=headers,
                    allow_hosts=allow,
                    max_fetch_bytes=_MAX_TEST_RESPONSE_BYTES,
                )
                return self._verdict(resp.status_code, safe_endpoint)

            if plan.mode in _GPU_MODES:
                return await self._gpu_probe(client, plan, model, allow, safe_endpoint)

            if plan.mode == "elevenlabs-user":
                base = endpoint.rstrip("/")
                url = base + "/user" if base.endswith("/v1") else base + "/v1/user"
            elif plan.mode == "get":
                url = endpoint
            elif plan.mode == "models-v1":
                url = _join(endpoint, "v1/models")
            else:  # "models" / "llm-openai" / "tts-openai-compatible"
                url = _join(endpoint, "models")
            if plan.mode == "tts-openai-compatible":
                return await self._tts_openai_compatible(
                    client, plan, url, model, allow, safe_endpoint
                )
            if plan.mode not in {"llm-openai", "vlm-openai"}:
                resp = await safe_request(
                    client,
                    "GET",
                    url,
                    headers=plan.headers,
                    allow_hosts=allow,
                    max_fetch_bytes=_MAX_TEST_RESPONSE_BYTES,
                )
                return self._verdict(resp.status_code, safe_endpoint)

            return await self._llm_openai_compatible(
                client,
                endpoint,
                url,
                model,
                secret,
                allow,
                safe_endpoint,
                messages=_VISION_PING if plan.mode == "vlm-openai" else _PING,
            )

    async def _llm_openai_compatible(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        models_url: str,
        model: str | None,
        secret: str | None,
        allow: frozenset[str] | None,
        safe_endpoint: str,
        *,
        messages: list[dict[str, object]] = _PING,
    ) -> ProviderTestResult:
        """``GET /models``; a listing naming ``model`` passes, otherwise a chat call decides.

        A reachable key is not enough: a typo'd or unserved model passes an
        auth-only probe and then fails every run. But a listing that LACKS the model
        is not proof it is unserved either — Azure lists deployments under other
        names, Gemini prefixes ``models/``, Ollama serves ``llama3`` as
        ``llama3:latest``, and listings paginate. So: listed ⇒ pass (fast path);
        unlisted, or the target cannot list (404/405/501, or a 2xx body that is not
        a listing) ⇒ a 1-token chat completion AGAINST the model is the proof, and
        only its failure fails the test.
        """
        auth = {"Authorization": f"Bearer {secret}"} if secret else {}
        resp = await safe_request(
            client,
            "GET",
            models_url,
            headers=auth,
            allow_hosts=allow,
            max_fetch_bytes=_MAX_TEST_RESPONSE_BYTES,
        )
        listing_unsupported = resp.status_code in _MODELS_UNSUPPORTED_STATUSES
        if not listing_unsupported and not 200 <= resp.status_code < 300:
            return self._verdict(resp.status_code, safe_endpoint)
        if not listing_unsupported:
            if not model:
                # Nothing to check against; the listing proves the credential.
                return self._verdict(resp.status_code, safe_endpoint)
            listed = _listed_model_ids(resp)
            if listed is not None and model in listed:
                return ProviderTestResult(
                    True,
                    f"{safe_endpoint} answered HTTP {resp.status_code} and lists "
                    f"model {_safe_model(model)!r}",
                    resp.status_code,
                )
            # Unlisted or not a listing: fall through to the chat proof.
        if not model:
            return ProviderTestResult(
                False,
                f"{safe_endpoint} has no /models endpoint; set a model so the "
                "connection can be tested with a chat completion",
                upstream_status=resp.status_code,
            )
        resp = await safe_request(
            client,
            "POST",
            _join(endpoint, "chat/completions"),
            json={"model": model, "max_tokens": 1, "messages": messages},
            headers=auth,
            allow_hosts=allow,
            max_fetch_bytes=_MAX_TEST_RESPONSE_BYTES,
        )
        if 200 <= resp.status_code < 300:
            return ProviderTestResult(
                True,
                f"{safe_endpoint} answered a chat completion for model "
                f"{_safe_model(model)!r} (HTTP {resp.status_code})",
                resp.status_code,
            )
        return ProviderTestResult(
            False,
            f"model {_safe_model(model)!r}: chat completion failed — "
            + _status_message(safe_endpoint, resp.status_code),
            resp.status_code,
        )

    async def _gpu_probe(
        self,
        client: httpx.AsyncClient,
        plan: _Plan,
        model: str | None,
        allow: frozenset[str] | None,
        safe_endpoint: str,
    ) -> ProviderTestResult:
        """The ``nonagentic_endpoint`` probes and the ``asr`` ``openai-compatible`` one."""

        async def get(path: str) -> httpx.Response:
            return await safe_request(
                client,
                "GET",
                _join(plan.endpoint, path),
                headers=plan.headers,
                allow_hosts=allow,
                max_fetch_bytes=_MAX_TEST_RESPONSE_BYTES,
            )

        async def post_wav(
            path: str, fields: Mapping[str, str], *, audio: bytes | None
        ) -> httpx.Response:
            body, content_type = multipart_body(fields, audio)
            return await safe_request(
                client,
                "POST",
                _join(plan.endpoint, path),
                content=body,
                headers={**plan.headers, "Content-Type": content_type},
                allow_hosts=allow,
                max_fetch_bytes=_MAX_TEST_RESPONSE_BYTES,
            )

        if plan.mode == "nonagentic-tts":
            payload: dict[str, object] = {"text": "test"}
            if plan.voice:
                payload["voice"] = plan.voice
            resp = await safe_request(
                client,
                "POST",
                _join(plan.endpoint, "api/tts/synthesize"),
                json=payload,
                headers={**plan.headers, "Accept": "application/json"},
                allow_hosts=allow,
                max_fetch_bytes=_MAX_TEST_RESPONSE_BYTES,
            )
            return self._verdict(resp.status_code, safe_endpoint)
        if plan.mode == "nonagentic-image":
            # ``{}`` has no ``prompt`` — refused by validation, nothing is generated.
            resp = await safe_request(
                client,
                "POST",
                _join(plan.endpoint, "api/image/generate"),
                json={},
                headers={**plan.headers, "Accept": "application/json"},
                allow_hosts=allow,
                max_fetch_bytes=_MAX_TEST_RESPONSE_BYTES,
            )
            return self._auth_probe_verdict(resp.status_code, safe_endpoint)
        if plan.mode == "nonagentic-asr":
            # No ``file`` part — refused before any transcription.
            resp = await post_wav(
                "api/asr/transcribe",
                {"word_timestamps": "false", "return_srt": "false"},
                audio=None,
            )
            return self._auth_probe_verdict(resp.status_code, safe_endpoint)
        # "asr-openai-compatible"
        resp = await get("models")
        if resp.status_code not in _MODELS_UNSUPPORTED_STATUSES:
            return self._verdict(resp.status_code, safe_endpoint)
        resp = await post_wav(
            "audio/transcriptions",
            {"model": model or ASR_DEFAULT_MODEL, "response_format": "json"},
            audio=silent_wav(),
        )
        return self._verdict(resp.status_code, safe_endpoint)

    async def _tts_openai_compatible(
        self,
        client: httpx.AsyncClient,
        plan: _Plan,
        models_url: str,
        model: str | None,
        allow: frozenset[str] | None,
        safe_endpoint: str,
    ) -> ProviderTestResult:
        """``GET /models``; on 404 a one-word ``POST /audio/speech`` (same safe client)."""
        resp = await safe_request(
            client,
            "GET",
            models_url,
            headers=plan.headers,
            allow_hosts=allow,
            max_fetch_bytes=_MAX_TEST_RESPONSE_BYTES,
        )
        if resp.status_code != 404:
            return self._verdict(resp.status_code, safe_endpoint)
        resp = await safe_request(
            client,
            "POST",
            _join(plan.endpoint, "audio/speech"),
            json={
                "model": model or OPENAI_TTS_DEFAULT_MODEL,
                "input": "test",
                "voice": plan.voice or OPENAI_TTS_DEFAULT_VOICE,
                "response_format": "wav",
            },
            headers=plan.headers,
            allow_hosts=allow,
            max_fetch_bytes=_MAX_TEST_RESPONSE_BYTES,
        )
        return self._verdict(resp.status_code, safe_endpoint)

    @staticmethod
    def _auth_probe_verdict(status: int, safe_endpoint: str) -> ProviderTestResult:
        """Verdict for a deliberately invalid request sent WITH the token (module doc)."""
        if status in (401, 403):
            return ProviderTestResult(
                False, f"{safe_endpoint} answered HTTP {status}: token rejected", status
            )
        if status in (400, 422) or 200 <= status < 300:
            return ProviderTestResult(
                True,
                f"{safe_endpoint} accepted the token (HTTP {status} on an empty probe request)",
                status,
            )
        if status == 404:
            return ProviderTestResult(
                False,
                f"{safe_endpoint} answered HTTP 404: route not found (check the base URL)",
                status,
            )
        if status >= 500:
            return ProviderTestResult(
                False, f"{safe_endpoint} answered HTTP {status}: endpoint unavailable", status
            )
        return ProviderTestResult(False, _status_message(safe_endpoint, status), status)

    @staticmethod
    def _verdict(status: int, safe_endpoint: str) -> ProviderTestResult:
        if 200 <= status < 300:
            return ProviderTestResult(True, f"{safe_endpoint} answered HTTP {status}", status)
        return ProviderTestResult(False, _status_message(safe_endpoint, status), status)
