"""``public_media_ref`` must decide ownership BEFORE anything mints a signature.

WHY THIS FILE EXISTS
--------------------
The read path used to be composed as::

    proxied_media_ref(row.id, await refresh_panel_media_url(row.media_ref), ...)

``refresh_panel_media_url`` re-signs on BUCKET ownership alone --
``StorageService.resign_presigned_url`` accepts "own bucket or any ``w2v-*``" and
applies no project/tenant key confinement. So it minted a fresh, valid, 1-hour
presigned GET for a key ``panel_object_ref`` was about to refuse, and
``proxied_media_ref`` then returned that media_ref *unchanged* -- handing the
caller a working pointer to the object the guard exists to withhold.
``media_ref`` is client-writable (a field on both ``StoryboardPanelCreate`` and
``StoryboardPanelUpdate``), so a tenant could PATCH its panel's url to another
tenant's key and read back a live signature for it.

Why the existing suite did not catch it: every storage double in
``test_storyboard_panel_image_proxy.py`` returns ``None`` from
``resign_presigned_url`` -- deliberately, so those tests prove the proxy path
comes from the rewrite rather than from a re-signer. Declining the re-sign is
also exactly what makes the unsafe composition look safe. The fake here RE-SIGNS,
like production does.
"""

from __future__ import annotations

from uuid import uuid4

from src.api.panel_image_url import panel_image_proxy_path, public_media_ref
from src.services.storyboard.panel_presign import MediaRefJson

_BUCKET = "w2v-orchestrator"
_HOST = f"http://minio.test:8389/{_BUCKET}/"


def _key(project_id, name: str = "frame-1") -> str:
    """The key a real panel generation writes for this project."""
    return f"artifacts/storyboard/{project_id}/storyboard_panel_image/{name}.png"


def _url(project_id, name: str = "frame-1", sig: str = "expired") -> str:
    return f"{_HOST}{_key(project_id, name)}?X-Amz-Signature={sig}"


def test_our_own_object_becomes_the_proxy_path_with_no_signature():
    panel_id, project_id, tenant_id = uuid4(), uuid4(), uuid4()
    ref: MediaRefJson = {"type": "image", "url": _url(project_id)}

    out = public_media_ref(panel_id, ref, project_id=project_id, tenant_id=tenant_id)

    assert out is not None
    assert out.get("url") == panel_image_proxy_path(panel_id)
    # The proxy reads with the orchestrator's own credentials, so the emitted url
    # must carry no signature at all -- re-signing this path was pure waste.
    assert "X-Amz-Signature" not in (out.get("url") or "")
    assert out.get("type") == "image"


def test_another_tenants_object_is_withheld_not_returned_unchanged():
    """The regression guard. Old behaviour returned the (re-signed) foreign url."""
    panel_id, project_id, tenant_id = uuid4(), uuid4(), uuid4()
    victim_project = uuid4()
    poisoned = _url(victim_project, sig="freshly-minted")
    ref: MediaRefJson = {"type": "image", "url": poisoned}

    out = public_media_ref(panel_id, ref, project_id=project_id, tenant_id=tenant_id)

    assert out is not None
    assert out.get("url") is None, "a foreign object's url must never be emitted"
    # Both halves matter: not the signature (which grants READ on the object),
    # and not the bare key (which discloses another tenant's object identity).
    assert "freshly-minted" not in str(out)
    assert str(victim_project) not in str(out)


def test_a_foreign_key_is_not_merely_de_signed_but_absent():
    """Stripping the query would still disclose the key. It must be None."""
    panel_id, project_id, tenant_id = uuid4(), uuid4(), uuid4()
    ref: MediaRefJson = {"url": _url(uuid4())}

    out = public_media_ref(panel_id, ref, project_id=project_id, tenant_id=tenant_id)

    assert out is not None
    assert out.get("url") is None
    assert "artifacts/storyboard" not in str(out)


def test_a_sibling_prefix_that_merely_shares_a_string_prefix_is_refused():
    """Containment is segment-wise -- ``{id}evil`` must not pass as ``{id}``."""
    panel_id, project_id, tenant_id = uuid4(), uuid4(), uuid4()
    evil = f"{_HOST}artifacts/storyboard/{project_id}evil/storyboard_panel_image/x.png"
    ref: MediaRefJson = {"url": evil}

    out = public_media_ref(panel_id, ref, project_id=project_id, tenant_id=tenant_id)

    assert out is not None
    assert out.get("url") is None


def test_an_operator_pasted_external_image_passes_through_untouched():
    """Not ours, not internal -- nothing to proxy, nothing to withhold."""
    panel_id, project_id, tenant_id = uuid4(), uuid4(), uuid4()
    external = "https://images.example.com/reference/hero.jpg"
    ref: MediaRefJson = {"type": "image", "url": external}

    out = public_media_ref(panel_id, ref, project_id=project_id, tenant_id=tenant_id)

    assert out == {"type": "image", "url": external}


def test_an_empty_or_absent_media_ref_is_returned_as_is():
    panel_id, project_id, tenant_id = uuid4(), uuid4(), uuid4()

    assert public_media_ref(panel_id, None, project_id=project_id, tenant_id=tenant_id) is None
    empty: MediaRefJson = {}
    assert public_media_ref(panel_id, empty, project_id=project_id, tenant_id=tenant_id) == {}


def test_the_callers_dict_is_never_mutated():
    """The argument is the SQLAlchemy row's JSONB value -- mutating it would mark
    the ORM object dirty and persist a url the DB must not hold."""
    panel_id, project_id, tenant_id = uuid4(), uuid4(), uuid4()
    stored = _url(project_id)
    ref: MediaRefJson = {"type": "image", "url": stored}

    public_media_ref(panel_id, ref, project_id=project_id, tenant_id=tenant_id)
    assert ref.get("url") == stored

    poisoned_url = _url(uuid4())
    poisoned: MediaRefJson = {"url": poisoned_url}
    public_media_ref(panel_id, poisoned, project_id=project_id, tenant_id=tenant_id)
    assert poisoned.get("url") == poisoned_url


def test_the_media_cache_prefix_is_accepted_for_the_bound_tenant():
    """Branch (b) of the guard -- the cross-run content-addressed cache."""
    panel_id, project_id, tenant_id = uuid4(), uuid4(), uuid4()
    cached = f"{_HOST}media-cache/tenants/{tenant_id}/ab/abcd.png?X-Amz-Signature=x"
    ref: MediaRefJson = {"url": cached}

    out = public_media_ref(panel_id, ref, project_id=project_id, tenant_id=tenant_id)

    assert out is not None
    assert out.get("url") == panel_image_proxy_path(panel_id)


def test_another_tenants_media_cache_entry_is_withheld():
    panel_id, project_id, tenant_id = uuid4(), uuid4(), uuid4()
    other = f"{_HOST}media-cache/tenants/{uuid4()}/ab/abcd.png?X-Amz-Signature=x"
    ref: MediaRefJson = {"url": other}

    out = public_media_ref(panel_id, ref, project_id=project_id, tenant_id=tenant_id)

    assert out is not None
    assert out.get("url") is None
