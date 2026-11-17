"""A login run only ever runs an orchestrator-uploaded, tenant-bound, hash-pinned package."""

from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.api.schemas.projects import ProjectCreate
from src.services import package_refs
from src.services.package_refs import (
    PackageRefError,
    make_package_ref,
    parse_package_ref,
    require_uploaded_package_for_logins,
    resolve_package_ref,
)

_UPLOAD_ID = "01a0fc54-f365-77e7-88ff-6e0835d36093"
_DATA = b"PK\x03\x04 recording package"
_LOGIN = {"login_url": "https://twenty.example/", "username": "u"}


class _Store:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str, str], bytes] = {}

    async def retrieve_stage_artifact(self, job_id: str, stage: str, filename: str) -> bytes | None:
        return self.objects.get((job_id, stage, filename))

    async def stage_artifact_url(self, job_id: str, stage: str, filename: str) -> str:
        return f"http://minio.test:8389/artifacts/{job_id}/{stage}/{filename}?sig=x"


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _Store:
    s = _Store()

    async def _get() -> _Store:
        return s

    monkeypatch.setattr(package_refs, "storage_for_packages", _get)
    return s


def _brief(package_url: str) -> dict:
    return {
        "target_url": "https://twenty.example/",
        "package_url": package_url,
        "auth_config": _LOGIN,
    }


def test_ref_shape_pins_id_and_sha() -> None:
    ref = make_package_ref(_UPLOAD_ID, _DATA)
    assert parse_package_ref(ref) == (_UPLOAD_ID, hashlib.sha256(_DATA).hexdigest())
    assert parse_package_ref("https://minio.test/pkg.zip") is None
    assert parse_package_ref(ref.replace("sha256:", "sha1:")) is None


@pytest.mark.parametrize(
    "package_url",
    ["https://attacker.example/pkg.zip"],
)
def test_login_brief_with_an_external_package_is_rejected_at_the_door(package_url: str) -> None:
    with pytest.raises(ValueError, match="package_ref"):
        require_uploaded_package_for_logins(_brief(package_url))
    with pytest.raises(ValidationError):
        ProjectCreate.model_validate(
            {
                "type": "showcase",
                "title": "t",
                "brief": _brief(package_url),
                "deliverables": [{"name": "d", "format": "16x9"}],
            }
        )


def test_login_brief_with_a_ref_and_login_free_external_package_pass() -> None:
    require_uploaded_package_for_logins(_brief(make_package_ref(_UPLOAD_ID, _DATA)))
    no_login = {"target_url": "https://x.example/", "package_url": "https://cdn.example/p.zip"}
    assert require_uploaded_package_for_logins(no_login) is None


async def test_resolve_needs_the_same_tenant_and_the_pinned_content(store: _Store) -> None:
    tenant, other = uuid4(), uuid4()
    ns = f"video-packages/{tenant}"
    store.objects[(ns, "recording_package", f"{_UPLOAD_ID}.zip")] = _DATA
    ref = make_package_ref(_UPLOAD_ID, _DATA)

    url = await resolve_package_ref(ref, tenant)
    assert url.startswith("http://minio.test:8389/artifacts/video-packages/")

    with pytest.raises(PackageRefError, match="not uploaded by this tenant"):
        await resolve_package_ref(ref, other)
    store.objects[(ns, "recording_package", f"{_UPLOAD_ID}.zip")] = _DATA + b"tampered"
    with pytest.raises(PackageRefError, match="sha256"):
        await resolve_package_ref(ref, tenant)
