"""Bound expensive local-model inference without locking whole workflows."""

import asyncio

from core.config import settings


model_semaphore = asyncio.Semaphore(settings.model_max_concurrency)
