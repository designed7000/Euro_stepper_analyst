"""
NBA Shot DNA — FastAPI backend.
Replaces the Streamlit app with a proper REST API.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import home, leaders, mvp, players, similarity
from config import SEASONS

app = FastAPI(
    title="NBA Shot DNA API",
    description="NBA analytics API powering the Euro Stepper Analyst platform",
    version="2.0.0"
)

# CORS — update ALLOWED_ORIGINS env var in production
_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [o.strip() for o in _origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(home.router, prefix="/api")
app.include_router(leaders.router, prefix="/api")
app.include_router(mvp.router, prefix="/api")
app.include_router(players.router, prefix="/api")
app.include_router(similarity.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/seasons")
async def get_seasons():
    return {"seasons": SEASONS}
