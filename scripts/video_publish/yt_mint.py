#!/usr/bin/env python3
"""Mint a DURABLE YouTube refresh token via the user's real browser (run ONCE).

The in-cluster youtube-upload-service is blocked by a GCP project mismatch (its
OAuth client lives in a project the user doesn't control) and gcloud is out of
scope. The proven, sanctioned path is to upload directly with the user's OWN
*desktop* OAuth client + a refresh token consented in their real browser
(Google blocks OAuth inside automation/CDP Chrome).

Run this on the user's Mac (NOT headless):
    python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
    python yt_mint.py
It opens the default browser → consent as a test user on project
api-project-75907155958 → writes a reusable token to TOKEN_PATH.

Scopes: youtube.upload (videos.insert) + youtube.force-ssl (captions.insert for
soft-CC). force-ssl is the superset that also authorizes uploads.
"""

from __future__ import annotations

import os
from pathlib import Path

# Mac-side publish dep (see requirements.txt); not in the orchestrator venv.
from google_auth_oauthlib.flow import InstalledAppFlow  # ty: ignore[unresolved-import]

# Point YT_CLIENT_SECRET at the operator's downloaded OAuth client file
# (client_secret_*.json from the Google Cloud console). No client file ships
# with this repository.
CLIENT_SECRET = os.environ.get(
    "YT_CLIENT_SECRET",
    str(Path.home() / "Downloads" / "client_secret.json"),
)
TOKEN_PATH = Path(
    os.environ.get("YT_TOKEN_PATH", str(Path.home() / ".config" / "w2v-video" / "yt_token.json"))
)
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


def main() -> int:
    if not Path(CLIENT_SECRET).is_file():
        print(f"FATAL: desktop client secret not found at {CLIENT_SECRET}")
        print("Set YT_CLIENT_SECRET=/path/to/client_secret_*.json")
        return 2
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent")  # opens the real browser
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())
    TOKEN_PATH.chmod(0o600)
    print(
        f"OK — durable token written to {TOKEN_PATH} (refresh token present: {bool(creds.refresh_token)})"
    )
    print("Keep this file private — it grants upload access to the channel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
