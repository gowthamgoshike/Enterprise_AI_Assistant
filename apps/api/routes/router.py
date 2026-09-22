"""One intentional versioned API surface."""

from fastapi import APIRouter

from apps.api.routes import auth, chat, chat_stream, health, jobs


api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(chat.router)
api_router.include_router(chat_stream.router)
api_router.include_router(jobs.router)
api_router.include_router(health.router)
