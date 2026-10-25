# Volleyball Tactical Analysis website: run and deploy

Full-stack app in `webapp/`:

| Part | Tech | Where |
|---|---|---|
| Frontend | React 18 + Vite (documentation portal + upload / results UI) | `webapp/frontend` |
| Backend API | FastAPI + Uvicorn | `webapp/backend/app/main.py` |
| Database | MongoDB 7 (collection `analyses`); JSON-file fallback if MongoDB is down | `webapp/backend/app/db.py` |
| Analysis engine | the project's own code: `OUTPUTS/bicausal_extract.py`, `BASEFRAMES/*`, `NEW_OUTPUTS/code/new_render.py` + `team_tactics.py`, `tactical_ml_v6.joblib` | `webapp/backend/app/engine.py` |
| Job queue | one GPU worker, one analysis at a time, survives restarts | `webapp/backend/app/worker.py` |
| Public access | Cloudflare Tunnel (HTTPS, no open ports) | `docker-compose.yml`, `public_link.bat` |

The analysis needs an NVIDIA GPU. About 9 frames/s for detection plus 9 frames/s for
rendering on an RTX 3050 laptop GPU, so 5 minutes of 30 fps video takes roughly 35 minutes.

## 0. Live public deployment (current)

| Part | Where |
|---|---|
| Website (all devices, 24/7) | https://volleyballanalyzer.dpdns.org (Vercel, also volleyballtactics.vercel.app) |
| Analysis server (uploads, 24/7) | Azure for Students VM `vta-server` (resource group `vta-rg`, Central India, Standard_B2als_v2: 2 vCPU, 4 GB + 4 GB swap, Ubuntu 24.04) |
| Server address | https://20-197-61-218.sslip.io (Caddy auto-HTTPS, docker compose: app + MongoDB 7 + Caddy, in `/srv/vta`) |
| Config | `webapp/azure/docker-compose.yml`, `webapp/azure/Caddyfile`; image built from `webapp/hfspace/Dockerfile` (CPU PyTorch) |
| Update everything | `powershell -ExecutionPolicy Bypass -File webappzure\deploy_azure.ps1` |
| SSH | `ssh -i %USERPROFILE%\.sshta_azure azureuser@20.197.61.218` |

The server has no GPU: about 1 minute of processing per second of video in Fast mode, so the cloud
limits analysis to 90 s of video per upload. Cost: about $27/month from the $100 student credit.

## 1. Run on this PC (Windows)

1. MongoDB: start Docker Desktop, then once
   `docker run -d --name vta-mongo --restart unless-stopped -p 127.0.0.1:27017:27017 -v vta-mongo-data:/data/db mongo:7`
   (or install MongoDB Community Server as a Windows service). Without MongoDB the site still works and stores
   jobs in a JSON file; `/api/health` shows which store is live.
2. Python packages (once): `pip install -r webapp/backend/requirements.txt`
3. Double-click `webapp/run_website.bat` and open http://localhost:8000

## 1b. Desktop app

- **Windows desktop app (this PC, full analysis on the local GPU):** double-click the
  "Volleyball Tactical Analysis" icon on the Desktop or in the Start menu. It opens the site in its own
  window (Edge WebView2 via `pywebview`) and starts MongoDB and the server itself if they are not running.
  Closing the window lets a running analysis finish in the background, then stops the server.
  Recreate the shortcuts with `powershell -ExecutionPolicy Bypass -File webapp\desktop\create_shortcut.ps1`
  (needs `pip install pywebview`).
- **Install from the website (any PC, PWA):** open the site in Chrome or Edge and click the install icon in
  the address bar (or menu -> "Install Volleyball Tactical Analysis"). It gets its own window and Start-menu
  entry; analysis still runs on the server that hosts the site.

## 1c. Windows installer for other PCs

`powershell -ExecutionPolicy Bypass -File webapp\installer\build_installer.ps1` builds
`%USERPROFILE%\vta_build\output\VolleyballTacticalAnalysis-Setup.exe` (~170 MB, Inno Setup 6).
It contains the app, the models, the docs and demos, a portable MongoDB and `uv`. During setup it downloads
Python 3.12 and the pinned libraries (`installer\requirements-desktop.txt`; CUDA PyTorch if the PC has an
NVIDIA GPU, CPU PyTorch otherwise) into the install folder. Per-user install, no admin rights; Start-menu
entries for the app, "Repair" (re-runs the download) and uninstall. User instructions: `installer\README.txt`.

## 2. Get a public link

### Quick (temporary) link, no account
Install cloudflared once: `winget install --id Cloudflare.cloudflared`.
With the site running, run `webapp/public_link.bat`. It prints a link like
`https://something-random.trycloudflare.com`. It works while the PC is on and both windows stay open, and
changes every time.

### Your own domain: volleyballtacticalanalyze.com
1. Buy the domain from any registrar (Cloudflare Registrar, Namecheap, GoDaddy...; about USD 10-15 per year).
2. Create a free Cloudflare account, "Add a site" -> `volleyballtacticalanalyze.com`, and at the registrar change
   the domain's nameservers to the two Cloudflare gives you.
3. Cloudflare dashboard -> Zero Trust -> Networks -> Tunnels -> Create a tunnel (cloudflared) -> copy the token.
4. In the tunnel, add Public Hostnames: `volleyballtacticalanalyze.com` and `www.volleyballtacticalanalyze.com`
   -> Service `http://localhost:8000` (or `http://web:8000` when using docker compose).
5. Run either
   - Windows: `webapp/public_link.bat <TOKEN>` (or `cloudflared service install <TOKEN>` to run it at startup), or
   - Docker: put the token in `webapp/.env` and run
     `docker compose -f webapp/docker-compose.yml --profile public up -d --build`.

Uploads go through Cloudflare in 32 MB chunks, so the 100 MB per-request limit of the free plan does not
block 500 MB videos.

## 3. Cloud GPU server (always on)
Any Linux VM with an NVIDIA GPU (e.g. AWS g4dn.xlarge, GCP T4, RunPod, Lambda) with Docker and the
NVIDIA Container Toolkit:

```bash
git clone <this repo> && cd finalyearproject     # models and videos are not in git; copy
                                                  # OUTPUTS/yolo11m-pose.pt, fyp/volleyball_best.pt,
                                                  # tactical_ml_v6.joblib, NEW_OUTPUTS/code/ as well
cp webapp/.env.example webapp/.env               # add CLOUDFLARE_TUNNEL_TOKEN
docker compose -f webapp/docker-compose.yml --profile public up -d --build
```

## 4. Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection |
| `VTA_STORAGE_DIR` | `webapp/storage` | uploads and results (keep outside OneDrive) |
| `VTA_MAX_UPLOAD_MB` | `500` | upload limit |
| `VTA_MAX_VIDEO_SECONDS` | `3600` | longest part of a video that can be analysed |
| `VTA_RETENTION_DAYS` | `14` | analyses older than this are deleted |

## 5. API
OpenAPI docs: `/api/docs-ui`. Main endpoints: `POST /api/uploads`, `PUT /api/uploads/{id}/chunks/{n}`,
`POST /api/analyses`, `GET /api/analyses/{id}`, `GET /api/analyses/{id}/files/{name}`, `GET /api/health`.

## 6. Refresh demo videos
`python webapp/scripts/make_demos.py` (add `--extra <mp4> "<title>" "<description>"` for a website output).
