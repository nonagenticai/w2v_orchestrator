"""Tests for configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import (
    APISettings,
    DownstreamServiceSettings,
    LoggingSettings,
    Settings,
    clear_settings_cache,
    get_settings,
)

# Ambient variables that shadow a DECLARED default. Settings reads the real
# process environment, so a test asserting a default has to clear these first or
# it asserts whatever the runner happens to export rather than what the code
# declares. The CI archetype exports ENVIRONMENT=test at the job level
# (gitops #2432 — rmt-secrets-lib 0.1.7 treats an ABSENT ENVIRONMENT as
# production, which needs Vault), so leaning on the unset value went red.
_DEFAULT_SHADOWING_ENV = ("ENVIRONMENT",)


def test_default_settings(monkeypatch: pytest.MonkeyPatch):
    """Settings() with no arguments produces sensible defaults."""
    for name in _DEFAULT_SHADOWING_ENV:
        monkeypatch.delenv(name, raising=False)
    clear_settings_cache()
    settings = Settings()
    assert settings.app_name == "W2V Orchestrator Service"
    assert settings.app_version == "0.1.0"
    assert settings.environment == "development"


def test_api_defaults():
    """APISettings defaults."""
    settings = Settings()
    assert settings.api.port == 6300
    assert settings.api.debug is False


def test_pipeline_defaults():
    """PipelineSettings defaults — for the fields something actually READS.

    This test used to assert the defaults of `max_concurrent_pipelines`,
    `retry_max_attempts`, `retry_delay_seconds` and `poll_interval_seconds`, none of which
    had a single reader in `src/`. That is worse than no coverage: it made a dead block look
    load-bearing and kept CI green while an operator setting those knobs got silence. The
    fields were deleted 2026-08-20; `tests/test_no_dead_settings_surface.py` now fails on any
    field with no reader, so this cannot come back.
    """
    settings = Settings()
    assert settings.pipeline.mode == "gateway"
    assert settings.pipeline.default_timeout_seconds == 7200
    assert settings.pipeline.max_poll_attempts == 500
    # (d) Hourly, per its own docstring. An audited scope-expansion bug on this
    # PR's branch briefly wired this to ``compile_budget.PRESIGN_TTL_SECONDS``
    # (7800s) instead — an accidental substitution of the literal ``3600``,
    # unrelated to either defect this PR fixes. MUTATION PROOF: reintroducing
    # that wiring turns this red (7800 != 3600).
    assert settings.pipeline.e2e_project_prune_interval_seconds == 3600


def test_database_url(monkeypatch):
    """DatabaseSettings has the expected default URL components."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    clear_settings_cache()
    settings = Settings()
    assert "5450" in settings.database.url
    assert "w2v_orchestrator" in settings.database.url
    assert "postgresql+psycopg" in settings.database.url


def test_database_pool_defaults():
    """DatabaseSettings pool defaults."""
    settings = Settings()
    assert settings.database.pool_size == 10
    assert settings.database.max_overflow == 20
    assert settings.database.pool_timeout == 30
    assert settings.database.echo is False


def test_redis_defaults():
    """RedisSettings field defaults — url is required env (no localhost fallback)."""
    settings = Settings()
    # Orchestrator overrides the shared base default (10): each live SSE stream
    # holds a pool connection for its lifetime, so 10 starves the pool. See
    # RedisSettings.max_connections in src/config.py.
    assert settings.redis.max_connections == 100
    assert settings.redis.decode_responses is True


def test_redis_url_required(monkeypatch):
    """REDIS_URL is required (orchestrator + w2v_common stripped localhost defaults)."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    clear_settings_cache()
    with pytest.raises(ValidationError):
        Settings()


def test_license_gate_defaults_on(monkeypatch: pytest.MonkeyPatch):
    """The w2v license gate is ON unless someone explicitly turns it off.

    Owner ruling: all w2v usage is licence-gated and an unsanctioned request is
    blocked. A default of False made `require_license` return before it looked at
    anything, so the gate was mounted on every tenant-scoped data router and
    enforced nothing. Both env aliases are cleared because Settings reads the real
    process environment — otherwise this asserts the runner's export, not the code.
    """
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)
    monkeypatch.delenv("LICENSE_GATE_ENABLED", raising=False)
    clear_settings_cache()
    assert Settings().license_gate_enabled is True


def test_license_gate_is_break_glass_off(monkeypatch: pytest.MonkeyPatch):
    """...and the ConfigMap switch still turns it off without a rebuild."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    clear_settings_cache()
    assert Settings().license_gate_enabled is False


def test_downstream_license_token(monkeypatch):
    """DownstreamServiceSettings exposes the license token read from DOWNSTREAM_LICENSE_TOKEN."""
    monkeypatch.setenv("DOWNSTREAM_LICENSE_TOKEN", "explicit-fake-token")
    clear_settings_cache()
    settings = Settings()
    token = settings.downstream.license_token
    assert token is not None
    assert token.get_secret_value() == "explicit-fake-token"


def test_downstream_license_token_unset_fails_loudly(monkeypatch, tmp_path):
    """No code default: an unset token must refuse to build Settings, never call peers unauthenticated.

    Hermetic against a dev box whose ``.env`` carries the token: ``_env_file=None``
    stops the root Settings reading it, but ``downstream`` is built by its own
    ``default_factory`` (``DownstreamServiceSettings`` has its own
    ``env_file=".env"``), which ``_env_file`` does not reach — so also run from an
    empty cwd where no relative ``.env`` exists.
    """
    monkeypatch.delenv("DOWNSTREAM_LICENSE_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)
    clear_settings_cache()
    with pytest.raises(ValidationError, match="license_token"):
        Settings(_env_file=None)
    with pytest.raises(ValidationError, match="license_token"):
        DownstreamServiceSettings(_env_file=None)


@pytest.mark.parametrize("blank", [""])
def test_downstream_license_token_blank_fails_loudly(monkeypatch, blank):
    """A set-but-blank token is as unauthenticated as an unset one and must also fail."""
    monkeypatch.setenv("DOWNSTREAM_LICENSE_TOKEN", blank)
    clear_settings_cache()
    with pytest.raises(ValidationError, match="DOWNSTREAM_LICENSE_TOKEN"):
        Settings()


def test_discovery_resolves_peer_services(monkeypatch):
    """Peer-service URLs are produced by w2v_common.discovery, not config fields."""
    from w2v_common.discovery import service_url

    # Clean any local-dev or per-peer overrides before exercising defaults.
    monkeypatch.delenv("W2V_LOCAL", raising=False)
    for name in ("ANALYSIS", "CONTENT", "TTS", "VIDEO_PRODUCTION", "WEBSCRAPER", "MEDIA_DISCOVERY"):
        monkeypatch.delenv(f"W2V_{name}_URL", raising=False)

    assert service_url("analysis") == "http://analysis-service.w2v:6303"
    assert service_url("content") == "http://content-service.w2v:6305"
    # tts replaced by mlx-Kokoro launchd on mini2 (Tailnet split-DNS, port 9202).
    assert service_url("tts") == "http://tts:9202"
    assert service_url("video_production") == "http://video-production-service.w2v:6308"
    # KAG is retired: the orchestrator calls webscraper + media-discovery directly.
    assert service_url("webscraper") == "http://webscraper.webscraper:9119"
    assert service_url("media_discovery") == "http://media-discovery.media-discovery:8809"


def test_discovery_honors_per_peer_override(monkeypatch):
    """W2V_<NAME>_URL env var overrides the discovery default for that peer."""
    from w2v_common.discovery import service_url

    monkeypatch.setenv("W2V_ANALYSIS_URL", "http://analysis.test:9999")
    assert service_url("analysis") == "http://analysis.test:9999"


def test_storage_defaults(monkeypatch):
    """StorageSettings defaults."""
    monkeypatch.delenv("STORAGE_LOCAL_PATH", raising=False)
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("STORAGE_PRESIGN_EXPIRY_SECONDS", raising=False)
    clear_settings_cache()
    settings = Settings()
    assert settings.storage.backend == "local"
    assert settings.storage.local_path == "/app/data"
    assert settings.storage.minio_bucket == "w2v-orchestrator"
    # (a) The GLOBAL default must stay at the inherited 3600s (1h) — NOT the long
    # compile-input TTL. Browser-facing presigns (uploads_video original_url,
    # storyboard panel thumbnails, the final-artifact rewrite path) all read this
    # setting directly with no per-call override, so raising it here would
    # silently double how long a leaked link stays valid. The long TTL is scoped
    # to ONLY the compile-input presign path — see
    # tests/services/test_shot_presign.py::test_default_ttl_is_the_compile_budget_ttl
    # and compile_budget.py for where it actually lives.
    #
    # MUTATION PROOF: this line was, for one commit on this PR's branch, wired to
    # ``compile_budget.PRESIGN_TTL_SECONDS`` (a scope-expansion bug an audit
    # caught) — reintroducing that wiring turns this red (7800 != 3600).
    assert settings.storage.presign_expiry_seconds == 3600


def test_auth_defaults():
    """AuthSettings defaults."""
    settings = Settings()
    assert settings.auth.enabled is True


def test_auth_settings_has_no_default_tenant_fallback():
    """There is no configured tenant to fall back to. Absence must RAISE.

    ``auth.default_tenant_id`` was removed in w2v-common 0.67.0. It held the nil
    uuid as a "fail-safe sentinel", but a sentinel tenant is not fail-safe: it is
    a real, truthy value that flows into storage prefixes, cache keys and
    outbound tenant headers, silently merging every unattributed caller into one
    shared bucket. 0.67.0 replaced substitution with refusal
    (``MissingTenantError`` / 401).

    This asserts the attribute is GONE rather than deleting the old test,
    so that re-introducing any config-level tenant fallback fails HERE.
    """
    settings = Settings()
    assert not hasattr(settings.auth, "default_tenant_id")


def test_logging_defaults():
    """LoggingSettings defaults."""
    settings = Settings()
    assert settings.logging.level == "INFO"
    assert settings.logging.json_logs is False


def test_logging_level_validation():
    """LoggingSettings rejects invalid log levels."""
    with pytest.raises(ValidationError):
        LoggingSettings(level="INVALID")


def test_logging_level_case_insensitive():
    """LoggingSettings accepts lowercase level and uppercases it."""
    ls = LoggingSettings(level="debug")
    assert ls.level == "DEBUG"


def test_helper_methods(monkeypatch: pytest.MonkeyPatch):
    """Settings helper methods work correctly."""
    for name in _DEFAULT_SHADOWING_ENV:
        monkeypatch.delenv(name, raising=False)
    clear_settings_cache()
    settings = Settings()
    assert settings.get_database_url() == settings.database.url
    assert settings.get_redis_url() == settings.redis.url
    assert settings.is_development() is True
    assert settings.is_production() is False


def test_is_production():
    """is_production returns True for production environment."""
    settings = Settings(environment="production")
    assert settings.is_production() is True
    assert settings.is_development() is False


def test_get_settings_caching():
    """get_settings returns the same instance (lru_cache)."""
    clear_settings_cache()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_clear_settings_cache():
    """clear_settings_cache resets the cached settings."""
    clear_settings_cache()
    s1 = get_settings()
    clear_settings_cache()
    s2 = get_settings()
    # After clearing, a new instance should be returned
    assert s1 is not s2


def test_api_cors_parsing():
    """APISettings accepts a list of CORS origins."""
    api = APISettings(cors_origins=["http://localhost:3000", "http://example.com"])
    assert api.cors_origins == ["http://localhost:3000", "http://example.com"]


def test_api_cors_empty_list():
    """APISettings handles empty CORS list."""
    api = APISettings(cors_origins=[])
    assert api.cors_origins == []


def test_api_cors_list():
    """APISettings accepts a list directly."""
    api = APISettings(cors_origins=["http://a.com", "http://b.com"])
    assert api.cors_origins == ["http://a.com", "http://b.com"]


def test_redis_settings_inherits_health_check_interval(monkeypatch) -> None:
    """Regression: idle pool connections must be health-checked so that a
    Redis restart doesn't leave the pool full of stale connections that fail
    every ping (seen in prod 2026-04-18).

    The field now lives on w2v_common.config.RedisSettings (default 0); the
    orchestrator opts in to a non-zero interval via REDIS_HEALTH_CHECK_INTERVAL
    in its K8s manifest.
    """
    from src.config import RedisSettings

    # Default (no env var) — inherited from w2v_common, value 0 (disabled).
    # url is required (REDIS_URL), seeded by conftest._clear_settings.
    monkeypatch.delenv("REDIS_HEALTH_CHECK_INTERVAL", raising=False)
    s_default = RedisSettings()
    assert hasattr(s_default, "health_check_interval"), (
        "RedisSettings must expose health_check_interval (inherited from w2v_common)"
    )
    assert s_default.health_check_interval == 0, (
        f"expected default 0 (inherited from w2v_common), got {s_default.health_check_interval}"
    )

    # Env var override — orchestrator's deploy sets this to 30.
    monkeypatch.setenv("REDIS_HEALTH_CHECK_INTERVAL", "30")
    s_env = RedisSettings()
    assert s_env.health_check_interval == 30, (
        f"expected 30 from REDIS_HEALTH_CHECK_INTERVAL env, got {s_env.health_check_interval}"
    )


def test_environment_is_read_from_the_ambient_env(monkeypatch: pytest.MonkeyPatch):
    """The other half: clearing the var above must not disable the override."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    clear_settings_cache()

    assert Settings().environment == "production"
