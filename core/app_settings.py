"""Backward-compatible import path for application settings."""

from core.config import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
