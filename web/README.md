# Granite Studios — Web App

The browser version of Granite Studios, built to live at
`marufhoque.com/tools/granite-studios`.

It imports the desktop app's engines from `src/` unchanged, so the website and
the desktop app always run the exact same image and video pipeline.

```
web/
├── backend/          FastAPI service (Docker → Render)
│   ├── app.py        HTTP endpoints wrapping src/ engines
│   ├── storage.py    Session-scoped temp files, TTL-swept
│   ├── Dockerfile    Build context is the REPO ROOT, not this folder
│   └── requirements.txt
└── frontend/         React + Vite + Tailwind (→ Vercel)
    ├── src/api.ts    Typed API client
    ├── src/App.tsx   Shell: upload, file grid, results
    └── src/components/
```

## Architecture

The portfolio site stays a static Vercel deploy. This tool is a **separate**
Vercel project plus a **separate** Render service; the portfolio only gains a
rewrite rule pointing at it. Editing this repo redeploys the tool and never
touches the portfolio.

```
Browser
  └─ marufhoque.com/tools/granite-studios
       └─ (Vercel rewrite) → granite-studios-web.vercel.app   [ frontend, static ]
            └─ (fetch)     → granite-studios-api.onrender.com [ backend, Docker ]
                                 └─ imports src/ engines (FFmpeg, OpenCV, rembg)
```

Vercel can't host the backend: video encoding needs a long-running process with
FFmpeg and a real filesystem, which serverless functions don't provide. Hence
Render for the API, Vercel for the static frontend.

## Deploy

### 1. Backend → Render

1. Push this repo to GitHub.
2. Render → **New → Blueprint**, select the repo. It reads `render.yaml` at the
   repo root and builds `web/backend/Dockerfile`.
3. Wait for the first build (~10 min — it installs OpenCV, FFmpeg, and
   pre-downloads the background-removal model).
4. Note the service URL, e.g. `https://granite-studios-api.onrender.com`.
5. Verify: `curl https://granite-studios-api.onrender.com/api/health` → `{"status":"ok"}`

**Free plan caveat:** the service sleeps after ~15 minutes idle, so the first
request afterwards takes 30–60s. The frontend shows a "server may be waking up"
message rather than a bare error. Upgrade to Starter ($7/mo) to keep it warm.

### 2. Frontend → Vercel

1. Vercel → **New Project** → same repo → set **Root Directory** to `web/frontend`.
2. Add an environment variable:
   - `VITE_API_URL` = your Render URL (no trailing slash)
3. Deploy. Note the URL, e.g. `https://granite-studios-web.vercel.app`.

### 3. Allow the frontend's origin on the backend

In Render → your service → **Environment**, set:

```
GRANITE_ALLOWED_ORIGINS=https://marufhoque.com,https://www.marufhoque.com,https://granite-studios-web.vercel.app
```

Then redeploy. (Any `localhost` port is always allowed for local development.)

### 4. Wire it into the portfolio site

Two edits in the **HoquePortfolio** repo — nothing else there changes.

**`vercel.json`** — add the rewrite pair (replace the `example-tool-two`
placeholder):

```json
{ "source": "/tools/granite-studios", "destination": "https://granite-studios-web.vercel.app" },
{ "source": "/tools/granite-studios/:path*", "destination": "https://granite-studios-web.vercel.app/:path*" }
```

**`data.ts`** — replace the `example-tool-two` entry in `aiTools`:

```ts
{
  id: "granite-studios",
  title: "Granite Studios",
  description: "AI photo and video suite — background removal, cinematic grading, video trimming, reframing and stabilization, all in the browser.",
  path: "/tools/granite-studios",
  tags: ["Computer Vision", "Video Processing", "Creative Tools"],
},
```

## Local development

```bash
# Terminal 1 — backend (from the repo root)
pip install -r web/backend/requirements.txt
cd web/backend && uvicorn app:app --reload --port 8000
```

```bash
# Terminal 2 — frontend
cd web/frontend && npm install && npm run dev
```

Open http://localhost:5173/tools/granite-studios/. Vite proxies `/api` to
port 8000, so no environment variable is needed locally.

> **Note:** `npm install` fails with `EBADF` if the repo lives on a Google Drive
> (File Stream) path — Drive's virtual filesystem can't handle npm's write
> pattern. Clone to a local disk for frontend work; Vercel's CI is unaffected.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness probe |
| `GET` | `/api/capabilities` | Filters, colors, video ops, presets, limits |
| `POST` | `/api/session` | Start a session (also sweeps expired ones) |
| `POST` | `/api/upload` | Upload a photo or video (multipart) |
| `GET` | `/api/preview/{id}` | Downscaled JPEG preview (renders RAW/HEIC) |
| `GET` | `/api/file/{id}` | Fetch a file; `?download=true` to attach |
| `POST` | `/api/photo/edit` | Background removal, filters, selective color, auto-fix, adjustments |
| `POST` | `/api/photo/instagram` | Format to 4:5 feed or 9:16 story |
| `POST` | `/api/video/edit` | Queue a chained video edit → job id |
| `POST` | `/api/video/reel` | Queue a panoramic reel from 3–7 photos → job id |
| `GET` | `/api/job/{id}` | Poll job status/result |

Photo edits answer inline; video work returns a job id the client polls, since
encoding outlasts a normal request.

## Privacy

Uploads land in a per-session directory and are deleted after one hour
(`SESSION_TTL_SECONDS`). Stored filenames are generated ids, never the uploaded
name, so a hostile filename can't influence where bytes are written. Nothing is
retained or shared.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GRANITE_STORAGE_ROOT` | `/tmp/granite-web` | Where session files live |
| `GRANITE_MAX_UPLOAD_MB` | `200` | Per-file upload cap |
| `GRANITE_ALLOWED_ORIGINS` | portfolio + localhost | Comma-separated CORS allowlist |
| `VITE_API_URL` | *(proxy in dev)* | Frontend → backend base URL |
