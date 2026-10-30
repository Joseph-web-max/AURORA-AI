"""AURORA AI backend (FastAPI).

Endpoints used by index.html:
  /health, /auth/config, /auth/register/request-code, /auth/register/verify-code,
  /auth/login, /auth/google, /auth/password/request-reset, /auth/password/reset,
  /settings/provider, /ai/models, /ai/chat/completions,
  /ai/native/models, /ai/native/chat/completions,
  /projects, /files, /files/upload, /files/{id}, /ml/linear-regression
"""
import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import smtplib
import socket
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import urlparse

import bcrypt
import httpx
import jwt
import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import Float, ForeignKey, Integer, String, Text, create_engine, or_, select, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

log = logging.getLogger("aurora")
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- config
def env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()

SECRET_KEY = env("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(48)
    log.warning("SECRET_KEY is not set: using a random key. Everyone is signed out on every restart. Set SECRET_KEY.")
ENCRYPTION_KEY = env("ENCRYPTION_KEY") or SECRET_KEY
FERNET = Fernet(base64.urlsafe_b64encode(hashlib.sha256(ENCRYPTION_KEY.encode()).digest()))

TOKEN_HOURS = int(env("TOKEN_HOURS", "168"))
CODE_TTL_SECONDS = 600
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in env(
    "ALLOWED_ORIGINS",
    "https://joseph-web-max.github.io,http://localhost:8080,http://localhost:5500,http://127.0.0.1:5500",
).split(",") if o.strip()]
GOOGLE_CLIENT_ID = env("GOOGLE_CLIENT_ID")
NATIVE_AI_URL = env("NATIVE_AI_URL").rstrip("/")      # OpenAI-compatible base, e.g. http://gpu-host:8001/v1
NATIVE_AI_KEY = env("NATIVE_AI_KEY")
DEV_SHOW_CODE = env("DEV_SHOW_CODE").lower() in ("1", "true", "yes")  # testing only: returns the code in the API response
ALLOW_INSECURE_PROVIDERS = env("ALLOW_INSECURE_PROVIDERS").lower() in ("1", "true", "yes")
UPLOAD_DIR = Path(env("UPLOAD_DIR", "./data/uploads"))
MAX_UPLOAD_BYTES = int(env("MAX_UPLOAD_MB", "10")) * 1024 * 1024

# --------------------------------------------------------------------------- database
db_url = env("DATABASE_URL", "sqlite:///./data/aurora.db")
if db_url.startswith("postgres://"):
    db_url = "postgresql+psycopg2://" + db_url[len("postgres://"):]
elif db_url.startswith("postgresql://"):
    db_url = "postgresql+psycopg2://" + db_url[len("postgresql://"):]
if db_url.startswith("sqlite"):
    Path("./data").mkdir(parents=True, exist_ok=True)
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
else:
    engine = create_engine(db_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def now() -> float:
    return time.time()


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    full_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str | None] = mapped_column(String(100), nullable=True)
    google_sub: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[float] = mapped_column(Float, default=now)


class EmailCode(Base):
    __tablename__ = "email_codes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(254), index=True)
    purpose: Mapped[str] = mapped_column(String(20))
    code_hash: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[float] = mapped_column(Float)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[float] = mapped_column(Float, default=now)


class ProviderSetting(Base):
    __tablename__ = "provider_settings"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)
    base_url: Mapped[str] = mapped_column(String(500))
    model: Mapped[str] = mapped_column(String(200), default="")
    api_key_enc: Mapped[str] = mapped_column(Text, default="")


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(String(1000), default="")
    content: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[float] = mapped_column(Float, default=now)


class FileRecord(Base):
    __tablename__ = "files"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(String(500))
    size: Mapped[int] = mapped_column(Integer, default=0)
    content_type: Mapped[str] = mapped_column(String(120), default="application/octet-stream")
    created_at: Mapped[float] = mapped_column(Float, default=now)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    log.info("AURORA backend ready. CORS origins: %s | Google: %s | Email: %s",
             ALLOWED_ORIGINS, "on" if GOOGLE_CLIENT_ID else "off",
             "brevo" if env("BREVO_API_KEY") else "smtp" if env("SMTP_HOST") else "NOT CONFIGURED")
    yield


app = FastAPI(title="AURORA API", lifespan=lifespan)


@app.middleware("http")
async def catch_all(request: Request, call_next):
    # Registered before CORS so that even error responses carry CORS headers
    # (otherwise the browser reports a 500 as "Failed to fetch").
    try:
        return await call_next(request)
    except Exception as exc:  # noqa: BLE001
        log.exception("Unhandled error: %s", exc)
        return JSONResponse({"detail": "Internal server error"}, status_code=500)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    return JSONResponse({"detail": f"Invalid input{(' (' + field + ')') if field else ''}: {first.get('msg', 'bad request')}"}, status_code=422)


# --------------------------------------------------------------------------- helpers
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request, bucket: str, limit: int, window: int = 60):
    ip = request.client.host if request.client else "unknown"
    q = _hits[f"{bucket}:{ip}"]
    t = now()
    while q and q[0] < t - window:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Too many attempts. Please wait a minute and try again.")
    q.append(t)


def norm_email(e: str) -> str:
    e = (e or "").strip().lower()
    if not EMAIL_RE.match(e):
        raise HTTPException(400, "Enter a valid email address.")
    return e


def check_name(n: str) -> str:
    n = re.sub(r"\s+", " ", (n or "").strip())
    if not 2 <= len(n) <= 100:
        raise HTTPException(400, "Enter your full name (2-100 characters).")
    return n


def check_password(p: str) -> str:
    if len(p or "") < 10:
        raise HTTPException(400, "Password must be at least 10 characters.")
    if len(p) > 200:
        raise HTTPException(400, "Password is too long.")
    return p


def _pre(pw: str) -> bytes:  # pre-hash so bcrypt's 72-byte limit never truncates silently
    return base64.b64encode(hashlib.sha256(pw.encode()).digest())


def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(_pre(pw), bcrypt.gensalt()).decode()


def check_pw(pw: str, hashed: str | None) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(_pre(pw), hashed.encode())
    except ValueError:
        return False


def make_token(user_id: str) -> str:
    t = int(now())
    return jwt.encode({"sub": user_id, "iat": t, "exp": t + TOKEN_HOURS * 3600}, SECRET_KEY, algorithm="HS256")


def auth_response(u: User) -> dict:
    return {"access_token": make_token(u.id), "token_type": "bearer",
            "user": {"id": u.id, "full_name": u.full_name, "email": u.email}}


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    h = request.headers.get("authorization", "")
    if not h.lower().startswith("bearer "):
        raise HTTPException(401, "Not signed in.")
    try:
        data = jwt.decode(h[7:].strip(), SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "Your session has expired. Please sign in again.")
    u = db.get(User, data.get("sub"))
    if not u:
        raise HTTPException(401, "Account not found. Please sign in again.")
    return u


# --------------------------------------------------------------------------- email
def send_email(to: str, subject: str, text: str):
    sender = env("EMAIL_FROM")
    brevo = env("BREVO_API_KEY")
    try:
        if brevo and sender:
            r = httpx.post("https://api.brevo.com/v3/smtp/email", timeout=20,
                           headers={"api-key": brevo, "accept": "application/json"},
                           json={"sender": {"email": sender, "name": "AURORA AI"}, "to": [{"email": to}],
                                 "subject": subject, "textContent": text})
            if r.status_code >= 300:
                raise RuntimeError(f"Brevo HTTP {r.status_code}: {r.text[:200]}")
        elif env("SMTP_HOST") and sender:
            msg = EmailMessage()
            msg["From"], msg["To"], msg["Subject"] = sender, to, subject
            msg.set_content(text)
            host, port = env("SMTP_HOST"), int(env("SMTP_PORT", "587"))
            smtp = smtplib.SMTP_SSL(host, port, timeout=20) if port == 465 else smtplib.SMTP(host, port, timeout=20)
            with smtp:
                if port != 465:
                    smtp.starttls()
                if env("SMTP_USER"):
                    smtp.login(env("SMTP_USER"), env("SMTP_PASSWORD"))
                smtp.send_message(msg)
        else:
            log.warning("EMAIL NOT CONFIGURED. Message for %s -> %s | %s", to, subject, text)
    except Exception as e:  # noqa: BLE001
        log.error("Email send failed: %s", e)
        raise HTTPException(502, "Could not send the verification email. Please try again later.")


def issue_code(db: Session, email: str, purpose: str) -> str:
    old = db.scalars(select(EmailCode).where(EmailCode.email == email, EmailCode.purpose == purpose)).all()
    if old and now() - max(o.created_at for o in old) < 30:
        raise HTTPException(429, "Please wait 30 seconds before requesting another code.")
    for o in old:
        db.delete(o)
    code = f"{secrets.randbelow(10**6):06d}"
    db.add(EmailCode(email=email, purpose=purpose, expires_at=now() + CODE_TTL_SECONDS,
                     code_hash=hashlib.sha256((code + SECRET_KEY).encode()).hexdigest()))
    db.commit()
    return code


def consume_code(db: Session, email: str, purpose: str, code: str):
    row = db.scalars(select(EmailCode).where(EmailCode.email == email, EmailCode.purpose == purpose)).first()
    if not row or row.expires_at < now():
        raise HTTPException(400, "That code has expired. Request a new one.")
    if row.attempts >= 5:
        db.delete(row); db.commit()
        raise HTTPException(400, "Too many wrong attempts. Request a new code.")
    good = hmac.compare_digest(row.code_hash, hashlib.sha256((code.strip() + SECRET_KEY).encode()).hexdigest())
    if not good:
        row.attempts += 1; db.commit()
        raise HTTPException(400, "Incorrect code.")
    db.delete(row); db.commit()


# --------------------------------------------------------------------------- schemas
class RegisterReq(BaseModel):
    full_name: str
    email: str
    password: str

class VerifyReq(RegisterReq):
    code: str

class LoginReq(BaseModel):
    identifier: str
    password: str

class EmailReq(BaseModel):
    email: str

class ResetReq(BaseModel):
    email: str
    code: str
    new_password: str

class GoogleReq(BaseModel):
    credential: str

class ProviderReq(BaseModel):
    base_url: str
    model: str = ""
    api_key: str = ""

class ProjectReq(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    content: dict = {}

class MLReq(BaseModel):
    rows: list[dict]
    target: str
    features: list[str]


# --------------------------------------------------------------------------- basic + auth
@app.get("/")
def root():
    return {"name": "AURORA API", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/auth/config")
def auth_config():
    return {"google_client_id": GOOGLE_CLIENT_ID or None}


@app.post("/auth/register/request-code")
def register_request(body: RegisterReq, request: Request, db: Session = Depends(get_db)):
    rate_limit(request, "reg", 8)
    check_name(body.full_name); check_password(body.password)
    email = norm_email(body.email)
    if db.scalars(select(User).where(User.email == email)).first():
        raise HTTPException(409, "An account with this email already exists. Please sign in.")
    code = issue_code(db, email, "register")
    send_email(email, "Your AURORA verification code",
               f"Your AURORA verification code is {code}. It expires in 10 minutes.\nIf you did not request it, ignore this email.")
    out = {"message": "Verification code sent."}
    if DEV_SHOW_CODE:
        out["dev_code"] = code
    return out


@app.post("/auth/register/verify-code")
def register_verify(body: VerifyReq, request: Request, db: Session = Depends(get_db)):
    rate_limit(request, "regv", 15)
    name, pw, email = check_name(body.full_name), check_password(body.password), norm_email(body.email)
    if db.scalars(select(User).where(User.email == email)).first():
        raise HTTPException(409, "An account with this email already exists. Please sign in.")
    consume_code(db, email, "register", body.code)
    u = User(full_name=name, email=email, password_hash=hash_pw(pw))
    db.add(u); db.commit()
    return auth_response(u)


@app.post("/auth/login")
def login(body: LoginReq, request: Request, db: Session = Depends(get_db)):
    rate_limit(request, "login", 10)
    ident = (body.identifier or "").strip().lower()
    cands = db.scalars(select(User).where(or_(User.email == ident, func.lower(User.full_name) == ident)).limit(5)).all()
    if not cands:
        check_pw(body.password, hash_pw("dummy-password"))  # equalise timing
    for u in cands:
        if check_pw(body.password, u.password_hash):
            return auth_response(u)
    if cands and all(not u.password_hash for u in cands):
        raise HTTPException(400, "This account uses Google sign-in. Use 'Continue with Google'.")
    raise HTTPException(401, "Incorrect name/email or password.")


@app.post("/auth/google")
def google_login(body: GoogleReq, request: Request, db: Session = Depends(get_db)):
    rate_limit(request, "google", 15)
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(503, "Google sign-in is not configured on this server.")
    try:
        from google.auth.transport import requests as g_requests
        from google.oauth2 import id_token
        info = id_token.verify_oauth2_token(body.credential, g_requests.Request(), GOOGLE_CLIENT_ID)
    except Exception as e:  # noqa: BLE001
        log.warning("Google token rejected: %s", e)
        raise HTTPException(401, "Google sign-in could not be verified. Please try again.")
    if not info.get("email") or not info.get("email_verified"):
        raise HTTPException(401, "Your Google email is not verified.")
    email, sub = info["email"].lower(), info["sub"]
    u = db.scalars(select(User).where(or_(User.google_sub == sub, User.email == email))).first()
    if u:
        if not u.google_sub:
            u.google_sub = sub; db.commit()
    else:
        u = User(full_name=(info.get("name") or email.split("@")[0])[:100], email=email, google_sub=sub)
        db.add(u); db.commit()
    return auth_response(u)


@app.post("/auth/password/request-reset")
def reset_request(body: EmailReq, request: Request, db: Session = Depends(get_db)):
    rate_limit(request, "rst", 6)
    email = norm_email(body.email)
    out = {"message": "If the account exists, a code has been sent."}
    if db.scalars(select(User).where(User.email == email)).first():
        try:
            code = issue_code(db, email, "reset")
        except HTTPException:
            return out
        send_email(email, "Your AURORA password reset code",
                   f"Your AURORA password reset code is {code}. It expires in 10 minutes.\nIf you did not request it, ignore this email.")
        if DEV_SHOW_CODE:
            out["dev_code"] = code
    return out


@app.post("/auth/password/reset")
def reset_password(body: ResetReq, request: Request, db: Session = Depends(get_db)):
    rate_limit(request, "rstv", 10)
    email, pw = norm_email(body.email), check_password(body.new_password)
    u = db.scalars(select(User).where(User.email == email)).first()
    if not u:
        raise HTTPException(400, "Incorrect code.")
    consume_code(db, email, "reset", body.code)
    u.password_hash = hash_pw(pw); db.commit()
    return {"message": "Password changed."}


# --------------------------------------------------------------------------- provider vault + AI gateway
def check_provider_url(url: str) -> str:
    """Block SSRF: only public https hosts (http allowed only if ALLOW_INSECURE_PROVIDERS)."""
    u = urlparse((url or "").strip())
    if u.scheme not in (("https", "http") if ALLOW_INSECURE_PROVIDERS else ("https",)) or not u.hostname:
        raise HTTPException(400, "Provider base URL must be a valid https:// address.")
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except OSError:
        raise HTTPException(400, "Could not resolve the provider host name.")
    if not ALLOW_INSECURE_PROVIDERS and any(not ipaddress.ip_address(i[4][0]).is_global for i in infos):
        raise HTTPException(400, "Provider URL must point to a public internet address.")
    return url.strip().rstrip("/")


@app.put("/settings/provider")
async def save_provider(body: ProviderReq, user: User = Depends(current_user), db: Session = Depends(get_db)):
    base = await run_in_threadpool(check_provider_url, body.base_url)
    row = db.get(ProviderSetting, user.id)
    if not row:
        row = ProviderSetting(user_id=user.id, base_url=base, model=body.model.strip(), api_key_enc="")
        db.add(row)
    row.base_url, row.model = base, body.model.strip()
    if body.api_key.strip():  # blank keeps the previously stored key
        row.api_key_enc = FERNET.encrypt(body.api_key.strip().encode()).decode()
    db.commit()
    return {"configured": True, "base_url": row.base_url, "model": row.model}


@app.get("/settings/provider")
def load_provider(user: User = Depends(current_user), db: Session = Depends(get_db)):
    row = db.get(ProviderSetting, user.id)
    if not row:
        return {"configured": False}
    return {"configured": True, "base_url": row.base_url, "model": row.model, "has_key": bool(row.api_key_enc)}


async def user_provider(user: User, db: Session):
    row = db.get(ProviderSetting, user.id)
    if not row:
        raise HTTPException(400, "No AI provider saved yet. Save your provider key on the Backend page first.")
    base = await run_in_threadpool(check_provider_url, row.base_url)
    try:
        key = FERNET.decrypt(row.api_key_enc.encode()).decode() if row.api_key_enc else ""
    except InvalidToken:
        raise HTTPException(400, "Stored provider key can no longer be read. Please save it again.")
    return base, key, row.model


def native_provider():
    if not NATIVE_AI_URL:
        raise HTTPException(503, "The built-in AURORA model runtime is not configured on this server (set NATIVE_AI_URL).")
    return NATIVE_AI_URL, NATIVE_AI_KEY, ""


async def list_models(base: str, key: str):
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(base + "/models", headers={"Authorization": f"Bearer {key}"} if key else {})
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Could not reach the AI provider: {e.__class__.__name__}")
    if r.status_code >= 300:
        raise HTTPException(502, f"AI provider error (HTTP {r.status_code}): {r.text[:200]}")
    j = r.json()
    items = j.get("data") or j.get("models") or []
    return {"data": [{"id": str(m.get("id") or m.get("name"))} for m in items if isinstance(m, dict) and (m.get("id") or m.get("name"))]}


async def proxy_chat(base: str, key: str, body: dict, default_model: str):
    body = dict(body)
    if not body.get("model") and default_model:
        body["model"] = default_model
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    url = base + "/chat/completions"
    client = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=15))
    try:
        if body.get("stream"):
            r = await client.send(client.build_request("POST", url, json=body, headers=headers), stream=True)
            if r.status_code >= 300:
                msg = (await r.aread()).decode(errors="replace")[:300]
                await r.aclose(); await client.aclose()
                raise HTTPException(502, f"AI provider error (HTTP {r.status_code}): {msg}")

            async def gen():
                try:
                    async for chunk in r.aiter_raw():
                        yield chunk
                finally:
                    await r.aclose(); await client.aclose()
            return StreamingResponse(gen(), media_type=r.headers.get("content-type", "text/event-stream"),
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        r = await client.post(url, json=body, headers=headers)
        await client.aclose()
        if r.status_code >= 300:
            raise HTTPException(502, f"AI provider error (HTTP {r.status_code}): {r.text[:300]}")
        return JSONResponse(r.json())
    except httpx.HTTPError as e:
        await client.aclose()
        raise HTTPException(502, f"Could not reach the AI provider: {e.__class__.__name__}")


@app.get("/ai/models")
async def ai_models(user: User = Depends(current_user), db: Session = Depends(get_db)):
    base, key, _ = await user_provider(user, db)
    return await list_models(base, key)


@app.post("/ai/chat/completions")
async def ai_chat(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    rate_limit(request, "chat", 60)
    base, key, model = await user_provider(user, db)
    return await proxy_chat(base, key, await request.json(), model)


@app.get("/ai/native/models")
async def native_models(user: User = Depends(current_user)):
    base, key, _ = native_provider()
    return await list_models(base, key)


@app.post("/ai/native/chat/completions")
async def native_chat(request: Request, user: User = Depends(current_user)):
    rate_limit(request, "nchat", 60)
    base, key, _ = native_provider()
    return await proxy_chat(base, key, await request.json(), "")


# --------------------------------------------------------------------------- projects
def project_dict(p: Project, full=False):
    d = {"id": p.id, "name": p.name, "description": p.description, "created_at": p.created_at}
    if full:
        d["content"] = json.loads(p.content or "{}")
    return d


@app.get("/projects")
def projects(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(Project).where(Project.user_id == user.id).order_by(Project.id.desc())).all()
    return [project_dict(p) for p in rows]


@app.post("/projects")
def create_project(body: ProjectReq, user: User = Depends(current_user), db: Session = Depends(get_db)):
    raw = json.dumps(body.content)
    if len(raw) > 2_000_000:
        raise HTTPException(413, "Project is too large (limit 2 MB).")
    p = Project(user_id=user.id, name=body.name.strip(), description=body.description, content=raw)
    db.add(p); db.commit()
    return project_dict(p)


@app.get("/projects/{pid}")
def get_project(pid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    p = db.get(Project, pid)
    if not p or p.user_id != user.id:
        raise HTTPException(404, "Project not found.")
    return project_dict(p, full=True)


@app.delete("/projects/{pid}")
def delete_project(pid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    p = db.get(Project, pid)
    if not p or p.user_id != user.id:
        raise HTTPException(404, "Project not found.")
    db.delete(p); db.commit()
    return {"deleted": True}


# --------------------------------------------------------------------------- files
@app.get("/files")
def list_files(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(FileRecord).where(FileRecord.user_id == user.id).order_by(FileRecord.created_at.desc())).all()
    return [{"id": f.id, "filename": f.filename, "size": f.size, "content_type": f.content_type} for f in rows]


@app.post("/files/upload")
async def upload_file(file: UploadFile = File(...), user: User = Depends(current_user), db: Session = Depends(get_db)):
    name = re.sub(r"[^\w.\- ]", "_", os.path.basename(file.filename or "upload"))[:200] or "upload"
    fid = str(uuid.uuid4())
    folder = UPLOAD_DIR / user.id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / fid
    size = 0
    try:
        with open(path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"File too large (limit {MAX_UPLOAD_BYTES // 1024 // 1024} MB).")
                out.write(chunk)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    rec = FileRecord(id=fid, user_id=user.id, filename=name, path=str(path), size=size,
                     content_type=file.content_type or "application/octet-stream")
    db.add(rec); db.commit()
    return {"id": fid, "filename": name, "size": size}


@app.get("/files/{fid}")
def download_file(fid: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    f = db.get(FileRecord, fid)
    if not f or f.user_id != user.id or not Path(f.path).exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(f.path, filename=f.filename, media_type=f.content_type)


@app.delete("/files/{fid}")
def delete_file(fid: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    f = db.get(FileRecord, fid)
    if not f or f.user_id != user.id:
        raise HTTPException(404, "File not found.")
    Path(f.path).unlink(missing_ok=True)
    db.delete(f); db.commit()
    return {"deleted": True}


# --------------------------------------------------------------------------- machine learning
def fit_linear_regression(rows: list[dict], target: str, features: list[str]) -> dict:
    if not target or not features:
        raise HTTPException(400, "Enter a target column and at least one predictor column.")
    if target in features:
        raise HTTPException(400, "The target column cannot also be a predictor.")
    if len(rows) < 12:
        raise HTTPException(400, "Provide at least 12 data rows.")
    X, y = [], []
    for i, r in enumerate(rows):
        try:
            X.append([float(r[f]) for f in features]); y.append(float(r[target]))
        except KeyError as e:
            raise HTTPException(400, f"Column {e} not found. Check the names against your CSV header.")
        except (TypeError, ValueError):
            raise HTTPException(400, f"Row {i + 1} has a non-numeric value in the target or predictor columns.")
    X, y = np.array(X), np.array(y)
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise HTTPException(400, "Data contains missing or infinite values.")
    idx = np.random.default_rng(42).permutation(len(y))
    n_test = max(2, int(round(len(y) * 0.2)))
    te, tr = idx[:n_test], idx[n_test:]
    A = np.hstack([np.ones((len(tr), 1)), X[tr]])
    coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    pred = np.hstack([np.ones((len(te), 1)), X[te]]) @ coef
    err = y[te] - pred
    ss_tot = float(((y[te] - y[te].mean()) ** 2).sum())
    r2 = 1 - float((err ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    clean = lambda v: float(v) if math.isfinite(v) else 0.0  # noqa: E731
    return {
        "observations": len(y),
        "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "metrics": {"r2": clean(r2), "rmse": clean(math.sqrt(float((err ** 2).mean()))), "mae": clean(float(np.abs(err).mean()))},
        "intercept": clean(coef[0]),
        "coefficients": {f: clean(c) for f, c in zip(features, coef[1:])},
        "note": "Ordinary least squares with an 80/20 train/test split (fixed seed). Metrics are computed on the held-out test rows; small datasets give unstable results.",
    }


@app.post("/ml/linear-regression")
def linear_regression(body: MLReq, user: User = Depends(current_user)):
    if len(body.rows) > 50_000:
        raise HTTPException(413, "Too many rows (limit 50,000).")
    return fit_linear_regression(body.rows, body.target.strip(), [f.strip() for f in body.features if f.strip()])
