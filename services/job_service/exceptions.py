class UnknownJobTypeError(Exception):
    """Raised when no executor is registered for a persisted job type."""
