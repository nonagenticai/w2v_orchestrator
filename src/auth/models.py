"""User model for authentication — delegates to w2v_common.

This module re-exports the shared ``User`` model so that existing imports
(``from src.auth.models import User``) continue to work without modification.
"""

from w2v_common.auth.models import User

__all__ = ["User"]
