"""``captions_mode`` accepts ``burned`` as an alias of the canonical ``burned_in``.

The alias is folded at the request boundary, so the value the route hands to the
repository (and therefore the DB CHECK, the compile stages and every response) is
always the canonical ``burned_in`` — never a second spelling downstream.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas.projects import (
    DeliverableCreate,
    DeliverableResponse,
    DeliverableUpdate,
    FromIntentRequest,
)


@pytest.mark.parametrize("value", ["burned"])
def test_create_folds_burned_alias_to_canonical(value: str) -> None:
    body = DeliverableCreate.model_validate({"name": "d", "format": "16x9", "captions_mode": value})
    assert body.captions_mode == "burned_in"
    assert body.model_dump()["captions_mode"] == "burned_in"


def test_update_folds_burned_alias_to_canonical() -> None:
    assert DeliverableUpdate.model_validate({"captions_mode": "burned"}).captions_mode == (
        "burned_in"
    )
    assert DeliverableUpdate.model_validate({}).captions_mode is None


def test_from_intent_folds_burned_alias() -> None:
    body = FromIntentRequest.model_validate(
        {
            "intent": {"type": "promo", "title": "t", "url": "https://example.com"},
            "captions_mode": "burned",
        }
    )
    assert body.captions_mode == "burned_in"


@pytest.mark.parametrize("value", ["burnt"])
def test_unknown_captions_mode_still_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        DeliverableCreate.model_validate({"name": "d", "format": "16x9", "captions_mode": value})


def test_response_does_not_accept_alias() -> None:
    """Responses are canonical-only: the alias is an INPUT convenience."""
    assert "burned" not in DeliverableResponse.model_fields["captions_mode"].annotation.__args__
