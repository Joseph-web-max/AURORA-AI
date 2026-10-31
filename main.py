import os
import time
import secrets
import hashlib
import hmac
from typing import Optional

import jwt
import httpx
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

APP_NAME = "AURORA AI Backend"
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    # For local smoke tests only. Set a strong secret in Render environment.
    JWT_SECRET = secrets.token_urlsafe(48)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./aurora.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

engine_kwargs = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, **engine_kwargs)

app = FastAPI(title=APP_NAME, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your published frontend domain before public production use.
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL DEFAULT '',
                password_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
        """))
init_db()

class RegisterIn(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=40)
    name: Optional[str] = Field(default="", max_length=100)
    full_name: Optional[str] = Field(default="", max_length=100)
    password: str = Field(min_length=8, max_length=200)

class LoginIn(BaseModel):
    email: Optional[str] = None
    username: Optional[str] = None
    password: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatIn(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7

def password_hash(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return salt.hex() + "$" + digest.hex()

def password_verify(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False

def issue_token(user_id: int, email: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": str(user_id), "email": email, "iat": now, "exp": now + 60 * 60 * 24 * 7},
        JWT_SECRET,
        algorithm="HS256",
    )

def current_user(authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in required")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return {"id": int(payload["sub"]), "email": payload["email"]}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

@app.get("/")
def root():
    return {"name": APP_NAME, "status": "online", "docs": "/docs"}

@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "service": APP_NAME, "database": "connected",
                "inference_configured": bool(os.getenv("AURORA_MODEL_BASE_URL"))}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")

@app.post("/auth/register")
@app.post("/api/auth/register")
@app.post("/api/v1/auth/register")
@app.post("/auth/signup")
@app.post("/register")
@app.post("/signup")
def register(data: RegisterIn):
    email = data.email.lower().strip()
    username = data.username.lower().strip()
    if not username.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Username may contain letters, numbers, underscores and hyphens only")
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                INSERT INTO users(email, username, full_name, password_hash, created_at)
                VALUES (:email, :username, :full_name, :password_hash, :created_at)
                RETURNING id
            """), {
                "email": email, "username": username,
                "full_name": (data.full_name or data.name or "").strip(),
                "password_hash": password_hash(data.password),
                "created_at": int(time.time())
            })
            user_id = result.scalar_one()
    except IntegrityError:
        raise HTTPException(status_code=409, detail="That email or username is already registered")
    token = issue_token(int(user_id), email)
    return {"access_token": token, "token_type": "bearer",
            "user": {"id": user_id, "email": email, "username": username}}

@app.post("/auth/login")
@app.post("/api/auth/login")
@app.post("/api/v1/auth/login")
@app.post("/auth/token")
@app.post("/login")
@app.post("/token")
def login(data: LoginIn):
    identity = (data.email or data.username or "").lower().strip()
    if not identity:
        raise HTTPException(status_code=400, detail="Enter your email or username")
    with engine.connect() as conn:
        user = conn.execute(text("""
            SELECT id, email, username, password_hash FROM users
            WHERE lower(email)=:identity OR lower(username)=:identity
        """), {"identity": identity}).mappings().first()
    if not user or not password_verify(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email/username or password")
    token = issue_token(int(user["id"]), user["email"])
    return {"access_token": token, "token_type": "bearer",
            "user": {"id": user["id"], "email": user["email"], "username": user["username"]}}

def inference_config():
    base = os.getenv("AURORA_MODEL_BASE_URL", "").rstrip("/")
    key = os.getenv("AURORA_MODEL_API_KEY", "")
    model = os.getenv("AURORA_MODEL_NAME", "")
    return base, key, model

@app.get("/ai/native/models")
def native_models(user=Depends(current_user)):
    base, key, model = inference_config()
    if not base or not model:
        raise HTTPException(status_code=503, detail="AURORA inference engine is not configured yet")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        with httpx.Client(timeout=20) as client:
            response = client.get(base + "/models", headers=headers)
            response.raise_for_status()
            data = response.json()
        items = data.get("data", data.get("models", []))
        if not items:
            items = [{"id": model, "object": "model"}]
        return {"object": "list", "data": items}
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Model service unavailable: {str(exc)[:250]}")

@app.post("/ai/native/chat/completions")
def native_chat(data: ChatIn, user=Depends(current_user)):
    base, key, model = inference_config()
    if not base or not model:
        raise HTTPException(status_code=503, detail="AURORA inference engine is not configured yet")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = data.model_dump()
    payload["model"] = model
    payload["stream"] = False
    try:
        with httpx.Client(timeout=180) as client:
            response = client.post(base + "/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Inference error {exc.response.status_code}: {exc.response.text[:400]}")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Inference service unavailable: {str(exc)[:250]}")
