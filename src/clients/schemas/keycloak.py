"""Typed responses for the Keycloak Admin REST client.

Keycloak's user representation carries many fields; the admin client mirrors
only what w2v reads. pydantic's default ``extra="ignore"`` drops the
rest without re-introducing ``Any``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class KeycloakUser(BaseModel):
    """Focused mirror of a Keycloak user representation (``GET /users/{id}``).

    Only what w2v reads: identity, the names the early-access identity sets,
    whether the address is verified, whether the account is enabled, and the
    attributes the tenancy backfill and the signup ownership stamp inspect.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    username: str | None = None
    email: str | None = None
    first_name: str | None = Field(default=None, alias="firstName")
    last_name: str | None = Field(default=None, alias="lastName")
    email_verified: bool = Field(default=False, alias="emailVerified")
    enabled: bool = True
    #: Set when the user comes from a user-storage federation (LDAP, …).
    federation_link: str | None = Field(default=None, alias="federationLink")
    attributes: dict[str, list[str]] | None = None
    #: When the realm user was created (epoch MILLISECONDS) — the early-access
    #: removal's legacy-identity evidence window reads it.
    created_timestamp: int | None = Field(default=None, alias="createdTimestamp")


class KeycloakRoleMappings(BaseModel):
    """A user's DIRECT role mappings (``GET /users/{id}/role-mappings``): realm role
    names and, per client (clientId), its role names. Composites are NOT expanded."""

    realm: list[str] = Field(default_factory=list)
    clients: dict[str, list[str]] = Field(default_factory=dict)
