# video_publish — reusable "create video → upload unlisted" capability

End-to-end toolkit for generating a video through the w2v orchestrator and
publishing it (unlisted) to YouTube. Built for repeated use across apps.

## Why this exists (the two hard constraints)

1. **Render must run in-cluster.** The w2v namespace is ambient-mesh STRICT and
   the public ingress strips license headers, so project/run API calls must be
   made from inside the `w2v` namespace. The render auth is the **perpetual**
   license token already in every w2v pod's env (`$DOWNSTREAM_LICENSE_TOKEN`,
   tenant `01a1113b` = "George Dekker"). We never extract it — we exec into the
   orchestrator pod and let it use its own env token.

2. **In-cluster YouTube upload is blocked.** `youtube-upload-service`'s OAuth
   client is in a GCP project the user doesn't control (Testing mode → 403), and
   gcloud is out of scope. The proven path is a **direct** YouTube Data API
   upload using the user's own *desktop* OAuth client + a refresh token consented
   in a real browser (Google blocks OAuth in automation Chrome).

## Pieces

| File | Runs where | What |
|---|---|---|
| `render_video.py` | inside an orchestrator pod | create project → start run (autopilot) → poll → print terminal `artifact_url` |
| `host_package.py` | inside an orchestrator pod | put a Playwright recording-package zip on MinIO + presign (env-configurable namespace/name/TTL) for `brief.package_url` |
| `run_in_pod.py` | user's Mac | codifies the `kubectl exec` glue: base64-embeds stdin, resolves the pod, pipes a helper into it (`--print` to dry-run) |
| `yt_mint.py` | user's Mac (real browser, ONCE) | mint a durable YouTube refresh token (scopes: upload + force-ssl) |
| `yt_upload.py` | anywhere with the token | upload a local MP4 unlisted + optional `.srt` soft-CC |
| `requirements.txt` | for the yt_* venv | google-api-python-client, google-auth-oauthlib |

## Which entry point? (this toolkit vs the parent `submit.py`)

`app_video_docs/` has TWO render paths sharing one publish tail — pick by how the
brief is authored:

| Render path | Brief source | Auth | Footage | Use when |
|---|---|---|---|---|
| **`../submit.py`** | `knowledge.yaml` → `generate.py` payloads (`promo` / `demo` / `instructional`) | KC self-mint (public keycloak) | `site_capture` (goto/click/hold per screen) | the app walk is a simple ordered screen list |
| **`render_video.py`** (here) | a hand-authored **brief JSON** (`showcase` / footage-driven) | the pod's own `$DOWNSTREAM_LICENSE_TOKEN` | **FTR re-runs a Playwright recording-package** (multi-step authed navigation) → needs `host_package.py` first | the walk is a stateful flow `site_capture` can't do (wizards, drag, canvas) |

Both then **publish the same way** — the direct-from-Mac path below
(`yt_mint.py` + `yt_upload.py`). The in-cluster `../_upload.py` is a **fallback
only**: its `youtube-upload-service` OAuth client sits in a GCP project the user
doesn't control (403 Testing mode), so prefer `yt_upload.py`.

## One-time setup (upload side)

```bash
cd app_video_docs/video_publish
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python yt_mint.py          # opens your browser; consent as a test user on api-project-75907155958
# → token at ~/.config/w2v-video/yt_token.json (private; reused by every upload)
```

## Run a video end-to-end

`run_in_pod.py` codifies the exec glue (pod resolution + the base64-embed pattern
below), so each step is one command. Add `--print` to any of them to dry-run
(assemble + show the exec, touch nothing).

```bash
# (a) FOOTAGE videos only: host the recording-package zip → presigned package_url.
#     Put the printed PACKAGE_URL into brief.package_url.
python run_in_pod.py --program host_package.py --stdin recording-package.zip \
    --env VIDEO_PKG_FILENAME=my-app-recording.zip
#     → prints KEY=… / PACKAGE_URL=…

# (b) render: the brief JSON is base64-embedded as the program's stdin.
python run_in_pod.py --program render_video.py --stdin brief.json
#     → prints {"run_id","artifact_url","status":"completed"}

# (c) pull the MP4 out for QC (presign the compiled artifact, curl it down), verify, then:
python yt_upload.py --file out.mp4 --title "..." --privacy unlisted --srt captions.srt
```

> **Gotcha (binary over `kubectl exec`), now codified in `run_in_pod.py`:** never
> `python - <<HEREDOC` *and* pipe data — the heredoc and the data pipe both claim
> stdin and the data is lost. `run_in_pod.py` embeds the stdin data as a base64
> literal inside a wrapper and pipes the whole wrapper to `python -`. Pods are
> multi-replica (HPA) so `/tmp` is not shared across execs — each op is a single
> self-contained exec (no staging step to race). Pass NON-secret config via
> `--env`; a demo password goes in the brief JSON (stdin), never in `--env`.

## Key facts

- Render tenant / channel owner: `01a1113b-674f-72b0-a84b-dc746d39afca`.
- YouTube channel: **George Dekker** `UCCswvznb9fLCzlaetPPkhLg`.
- Desktop OAuth client: `~/Downloads/client_secret_75907155958-*.json` (project `api-project-75907155958`).
- Orchestrator API: `http://w2v-orchestrator.w2v:6300` (container port 6300).
- MinIO: `minio.test:8389`, bucket `w2v-orchestrator`; presigned URLs pass FTR's SSRF guard (DNS host).
- Captions: soft-CC `.srt` attach needs `youtube.force-ssl` (force-ssl also authorizes upload).
