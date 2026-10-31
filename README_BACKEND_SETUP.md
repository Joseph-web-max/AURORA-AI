# AURORA AI backend starter

This backend matches the AURORA frontend's native routes:
- `POST /auth/register` (also `/api/auth/register`, `/api/v1/auth/register`, `/auth/signup`, `/register`, `/signup`)
- `POST /auth/login` (also `/api/auth/login`, `/api/v1/auth/login`, `/auth/token`, `/login`, `/token`)
- `GET /health`
- `GET /ai/native/models` (JWT required)
- `POST /ai/native/chat/completions` (JWT required)

## Deploy on Render
1. Add this folder and `render.yaml` to the root of the existing GitHub repository. Keep your existing `index.html`.
2. In Render, choose **New + → Blueprint** and connect the GitHub repository.
3. Render reads `render.yaml`; review the service and deploy.
4. After deployment, copy the service's `https://...onrender.com` URL.
5. Set a PostgreSQL `DATABASE_URL` in the service environment. Render's free web service filesystem is ephemeral; do not rely on local SQLite for permanent user accounts.
6. Set the inference environment variables only after you have a real OpenAI-compatible inference endpoint:
   - `AURORA_MODEL_BASE_URL`: base URL ending before `/models` and `/chat/completions`
   - `AURORA_MODEL_NAME`: exact model ID served by that endpoint
   - `AURORA_MODEL_API_KEY`: optional server-side credential for that inference endpoint
7. Redeploy after changing environment variables.

## Important
- This is the backend/API foundation, not a trained or hosted language model.
- Until a real inference service is configured, the native model routes intentionally return HTTP 503 rather than inventing model responses.
- Use a managed PostgreSQL database for durable user accounts. Do not expose `JWT_SECRET` or model credentials in `index.html`.
- Before public production launch, restrict CORS to your actual frontend domain, add rate limiting, email verification, password reset, abuse monitoring, backups, and security review.
- Render free services may sleep when idle and are not GPU inference hosts.
