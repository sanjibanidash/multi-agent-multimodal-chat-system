"""
backend/main.py
----------------
FastAPI application entry point.
Registers all routes and starts the database seeding on startup.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.database.sample_data import seed_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before serving requests."""
    # Seed the SQLite database with sample data (skips if already seeded)
    seed_database()
    yield
    # Cleanup tasks on shutdown (nothing needed for in-memory resources)


app = FastAPI(
    title="Multi-Agent Multimodal Chat System",
    description=(
        "A beginner-friendly multi-agent chat API with a Multimodal Agent, "
        "SQL Agent, and RAG Agent. Built with FastAPI + LangChain + ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Streamlit frontend (running on a different port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes under the /api prefix
app.include_router(router, prefix="/api")


@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "ok", "message": "Multi-Agent Chat API is running."}
