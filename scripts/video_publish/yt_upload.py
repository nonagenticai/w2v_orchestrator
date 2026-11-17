#!/usr/bin/env python3
"""Upload a local video to YouTube (unlisted by default) + optional soft-CC SRT.

Reusable across apps. Uses the durable token from yt_mint.py and uploads directly
via the YouTube Data API (bypasses the broken in-cluster youtube-upload-service).

    python yt_upload.py --file out.mp4 --title "..." [--description ...] \
        [--privacy unlisted] [--srt captions.srt] [--lang en] [--tags a,b]

Prints the resulting watch URL. The token auto-refreshes from its refresh_token;
re-run yt_mint.py only if scopes change or the grant is revoked.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Mac-side publish deps (see requirements.txt); not in the orchestrator venv.
from google.auth.transport.requests import Request  # ty: ignore[unresolved-import]
from google.oauth2.credentials import Credentials  # ty: ignore[unresolved-import]
from googleapiclient.discovery import build  # ty: ignore[unresolved-import]
from googleapiclient.http import MediaFileUpload  # ty: ignore[unresolved-import]

TOKEN_PATH = Path(
    os.environ.get("YT_TOKEN_PATH", str(Path.home() / ".config" / "w2v-video" / "yt_token.json"))
)
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


def _creds() -> Credentials:
    if not TOKEN_PATH.is_file():
        raise SystemExit(f"FATAL: no token at {TOKEN_PATH}; run yt_mint.py first")
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds.valid and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
    return creds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--privacy", default="unlisted", choices=["public", "unlisted", "private"])
    ap.add_argument("--tags", default="")
    ap.add_argument("--category", default="27", help="27=Education, 28=Sci/Tech, 22=People")
    ap.add_argument("--srt", default="", help="optional .srt soft-CC track to attach")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    if not Path(args.file).is_file():
        raise SystemExit(f"FATAL: video not found: {args.file}")
    yt = build("youtube", "v3", credentials=_creds())

    body = {
        "snippet": {
            "title": args.title,
            "description": args.description,
            "tags": [t for t in args.tags.split(",") if t],
            "categoryId": args.category,
        },
        "status": {"privacyStatus": args.privacy, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(args.file, chunksize=-1, resumable=True, mimetype="video/mp4")
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            print(f"  upload {int(status.progress() * 100)}%")
    vid = resp["id"]
    url = f"https://youtu.be/{vid}"
    print(f"VIDEO_ID={vid}")
    print(f"WATCH_URL={url}")

    if args.srt and Path(args.srt).is_file():
        cap = (
            yt.captions()
            .insert(
                part="snippet",
                body={
                    "snippet": {
                        "videoId": vid,
                        "language": args.lang,
                        "name": "captions",
                        "isDraft": False,
                    }
                },
                media_body=MediaFileUpload(
                    args.srt, mimetype="application/octet-stream", resumable=False
                ),
            )
            .execute()
        )
        print(f"CAPTION_ID={cap.get('id')} (lang={args.lang})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
