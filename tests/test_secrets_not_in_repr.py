"""Secret-valued settings must never appear in a Settings repr.

Why this exists — a real leak, not a hypothetical
-------------------------------------------------
On 2026-08-04 a CI run failed inside ``tests/test_config.py``. pytest renders
the offending object in the assertion diff, so it printed the whole ``Settings``
repr into the GitHub Actions log — including the live shared credential, in
full and in plain text, as the ``license_token`` field of the nested
``DownstreamServiceSettings``.

Nothing about that failure was security-related. Any future assertion that
happens to render ``Settings`` republishes every plain-``str`` secret it holds,
and CI logs are retained. The type is the control: :class:`pydantic.SecretStr`
reprs as ``SecretStr('**********')`` and only yields the value through an
explicit ``.get_secret_value()`` call, which is greppable in review.

These tests assert the *rendering*, not the type, on purpose: swapping the
annotation back to ``str`` must fail here, and so must any new secret field that
is introduced as a bare ``str``.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from src.config import Settings, clear_settings_cache

# (dotted path from Settings, env var that populates it, a sentinel value).
# Add a row when a new secret-valued setting appears — that is the whole
# maintenance cost of this guard.
SECRET_FIELDS: list[tuple[str, str, str]] = [
    ("downstream.license_token", "DOWNSTREAM_LICENSE_TOKEN", "s3nt1nel-license-token-VALUE"),
    # ProvisioningSettings declares no env_prefix and is case-insensitive, so the
    # env var is the flat field name — it deliberately matches the vars already
    # wired on the orchestrator Deployment (KEYCLOAK_URL etc.).
    (
        "provisioning.keycloak_client_secret",
        "KEYCLOAK_CLIENT_SECRET",
        "s3nt1nel-client-secret-VALUE",
    ),
]


def _resolve(settings: Settings, dotted: str) -> object:
    obj: object = settings
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize(
    ("dotted", "env_var", "sentinel"),
    [("downstream.license_token", "DOWNSTREAM_LICENSE_TOKEN", "s3nt1nel-license-token-VALUE")],
)
def test_secret_value_absent_from_repr(
    dotted: str, env_var: str, sentinel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The raw secret must not survive into repr() or str() of Settings."""
    monkeypatch.setenv(env_var, sentinel)
    clear_settings_cache()
    settings = Settings()

    # Precondition: the sentinel actually landed, so a pass cannot be vacuous.
    resolved = _resolve(settings, dotted)
    actual = resolved.get_secret_value() if isinstance(resolved, SecretStr) else resolved
    assert actual == sentinel, (
        f"{dotted} did not pick up {env_var}; the assertions below would be vacuous"
    )

    assert sentinel not in repr(settings), (
        f"{dotted} leaked its raw value into repr(Settings). Annotate it as "
        f"pydantic.SecretStr and read it via .get_secret_value() at the call site."
    )
    assert sentinel not in str(settings), (
        f"{dotted} leaked its raw value into str(Settings). Annotate it as "
        f"pydantic.SecretStr and read it via .get_secret_value() at the call site."
    )


@pytest.mark.parametrize(
    ("dotted", "env_var", "sentinel"),
    [("downstream.license_token", "DOWNSTREAM_LICENSE_TOKEN", "s3nt1nel-license-token-VALUE")],
)
def test_secret_field_is_secretstr(
    dotted: str, env_var: str, sentinel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SecretStr is the mechanism; assert it directly so regressions name themselves."""
    monkeypatch.setenv(env_var, sentinel)
    clear_settings_cache()
    resolved = _resolve(Settings(), dotted)
    assert isinstance(resolved, SecretStr), (
        f"{dotted} is {type(resolved).__name__}, not SecretStr — a plain str is "
        f"rendered verbatim by pydantic's repr and will leak into CI logs."
    )
