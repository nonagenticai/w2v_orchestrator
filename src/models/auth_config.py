"""Pydantic schema for authenticated-scrape configuration.

Matches webscraper_service's ``src/schemas/scrape.py::AuthConfig`` field-for-field
(re-verified 2026-10-18; the retired KAG gateway forwarded the same model verbatim).
When a pipeline request carries ``user_context.auth``, w2v_orchestrator validates
it against this model before forwarding it to webscraper (``WebscraperClient``). The raw credential
is never stored or transmitted here — only the ``credential_id`` issued by the
secrets app at storage time.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class AuthType(StrEnum):
    """Authentication flow kind. Values map 1:1 to webscraper's AuthType."""

    FORM = "form"
    STORAGE_STATE = "storage_state"
    BEARER = "bearer"
    BASIC = "basic"


class AuthConfig(BaseModel):
    """Reference to a stored credential plus Playwright login hints.

    The credential itself is stored by the caller (fr, n8n, portal) against
    the secrets app; the returned ``credential_id`` is what w2v receives.
    Downstream webscraper resolves the id at scrape time — raw
    credentials are never passed inline.
    """

    type: AuthType = Field(default=AuthType.FORM)
    credential_id: str = Field(min_length=1, max_length=256)
    login_url: HttpUrl | None = None
    username_selector: str | None = Field(default=None, max_length=1000)
    password_selector: str | None = Field(default=None, max_length=1000)
    submit_selector: str | None = Field(default=None, max_length=1000)
    wait_for_url: str | None = Field(default=None, max_length=1000)
    wait_for_selector: str | None = Field(default=None, max_length=1000)
    wait_timeout_ms: int = Field(default=15_000, ge=1000, le=120_000)
