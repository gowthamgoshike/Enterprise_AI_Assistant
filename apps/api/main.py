from fastapi import FastAPI
from fastapi.middleware.cors import (
    CORSMiddleware,
)
import apps.api.logging_config
from apps.api.routes.chat import (
    router as chat_router,
)

from apps.api.routes.health import (
    router as health_router,
)


app = FastAPI(
    title="Enterprise AI Assistant API",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():

    return {
        "service":
            "Enterprise AI Assistant API",
        "version":
            "0.1.0",
    }


app.include_router(
    health_router
)

app.include_router(
    chat_router
)