"""The tenant audit ``action`` filter accepts exactly the stored vocabulary.

``AuditAction`` (``schemas/tenant_audit.py``) must equal the ``ck_admin_audit_action``
CHECK set on ``AdminAuditLog``: a value missing from the Literal makes that action
unfilterable (422), an extra one is a filter that can never match.
"""

from __future__ import annotations

import re
from typing import get_args

from sqlalchemy import CheckConstraint

from src.api.schemas.tenant_audit import AuditAction
from src.database.models import AdminAuditLog
from src.database.repositories.admin_audit_repository import PLATFORM_ONLY_ACTIONS


def _check_values() -> set[str]:
    [ck] = [
        c
        for c in AdminAuditLog.__table_args__
        if isinstance(c, CheckConstraint) and c.name == "ck_admin_audit_action"
    ]
    return set(re.findall(r"'([a-z_.]+)'", str(ck.sqltext)))


def test_filter_vocabulary_equals_the_check_constraint() -> None:
    values = _check_values()
    assert len(values) > 20  # the regex really parsed the constraint
    assert set(get_args(AuditAction)) == values


def test_platform_only_actions_are_real_actions() -> None:
    assert set(PLATFORM_ONLY_ACTIONS) <= _check_values()
