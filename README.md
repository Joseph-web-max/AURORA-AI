# AURORA backend

FastAPI server for `index.html`: accounts (email code + Google), password reset,
encrypted AI-provider key vault, AI gateway (streaming), projects, private file uploads, regression API.

## Run locally
    pip install -r requirements.txt
    cp .env.example .env   # then export the variables, or set them in your shell
    uvicorn main:app --reload --port 8000
    # open http://127.0.0.1:8000/health  -> {"status":"ok"}
Without email settings, verification codes are printed in the server log (or returned on screen if DEV_SHOW_CODE=true).

## Deploy on Render (free)
1. Push this folder to a GitHub repo.
2. Render -> New -> Blueprint -> pick the repo (uses render.yaml). Wait for the build.
3. Open https://YOUR-SERVICE.onrender.com/health. It should show {"status":"ok"}.
4. In index.html set: const AURORA_API_URL='https://YOUR-SERVICE.onrender.com';  then re-upload to GitHub Pages.

## Settings (Render -> Environment)
- ALLOWED_ORIGINS: exact site origin, no trailing slash (default https://joseph-web-max.github.io).
- DATABASE_URL: free Render disk is erased on each deploy/restart, so accounts vanish with SQLite.
  Create a free Postgres (Neon or Supabase) and paste its connection string.
- GOOGLE_CLIENT_ID: Google Cloud Console -> Credentials -> OAuth client (Web). Add your site origin under
  "Authorized JavaScript origins".
- Email: create a free Brevo account, verify a sender address, set BREVO_API_KEY and EMAIL_FROM.
  (Render's free tier blocks SMTP, so use Brevo's HTTPS API.)
- NATIVE_AI_URL: optional, powers "AURORA AI · Built-in". Needs your own OpenAI-compatible model server.
- Free instances sleep after ~15 minutes idle; the first request may take about a minute.
