"""TranscriptionWorkerClient — a bad response body must surface as RuntimeError."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from src.clients.transcription_worker_client import TranscriptionWorkerClient


class _Resp:
    def __init__(self, *, status: int = 200, raise_json: bool = False, body: Any = None) -> None:
        self.status_code = status
        self._raise = raise_json
        self._body = body or {}
        self.text = "body"

    def json(self) -> Any:
        if self._raise:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")  # JSONDecodeError-like
        return self._body


class _Client:
    def __init__(self, *, resp: _Resp | None = None, exc: Exception | None = None) -> None:
        self._resp = resp
        self._exc = exc

    async def post(self, *_a: Any, **_k: Any) -> _Resp:
        if self._exc is not None:
            raise self._exc
        assert self._resp is not None
        return self._resp


def _wav(tmp_path: Path) -> Path:
    p = tmp_path / "clip.wav"
    p.write_bytes(b"RIFF....")
    return p


async def test_malformed_2xx_body_raises_runtimeerror(tmp_path: Path, monkeypatch) -> None:
    # A 2xx with an unparseable body previously raised JSONDecodeError (a ValueError)
    # OUTSIDE the except → a raw 500; callers expect the client's RuntimeError.
    client = TranscriptionWorkerClient()

    async def _gc() -> _Client:
        return _Client(resp=_Resp(status=200, raise_json=True))

    monkeypatch.setattr(client, "_get_client", _gc)
    with pytest.raises(RuntimeError):
        await client.transcribe_file(file_path=_wav(tmp_path))


async def test_schema_drift_raises_runtimeerror(tmp_path: Path, monkeypatch) -> None:
    # A 2xx whose JSON is missing required fields → pydantic ValidationError, which
    # is likewise not an httpx.HTTPError; it must surface as RuntimeError.
    client = TranscriptionWorkerClient()

    async def _gc() -> _Client:
        return _Client(resp=_Resp(status=200, body={"unexpected": "shape"}))

    monkeypatch.setattr(client, "_get_client", _gc)
    with pytest.raises(RuntimeError):
        await client.transcribe_file(file_path=_wav(tmp_path))


async def test_invalid_url_raises_runtimeerror(tmp_path: Path, monkeypatch) -> None:
    # httpx.InvalidURL (misconfigured discovery base_url) is NOT an httpx.HTTPError,
    # so it escaped the request except → must be mapped to RuntimeError.
    client = TranscriptionWorkerClient()

    async def _gc() -> _Client:
        return _Client(exc=httpx.InvalidURL("bad base url"))

    monkeypatch.setattr(client, "_get_client", _gc)
    with pytest.raises(RuntimeError):
        await client.transcribe_file(file_path=_wav(tmp_path))
