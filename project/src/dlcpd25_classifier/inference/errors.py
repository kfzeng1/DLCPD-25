"""Public inference errors with messages safe to display in the Web UI."""

from __future__ import annotations


class InferenceError(RuntimeError):
    """Base error carrying a stable code and a user-facing message."""

    def __init__(self, code: str, user_message: str) -> None:
        super().__init__(user_message)
        self.code = code
        self.user_message = user_message


class ImageValidationError(InferenceError):
    """Raised when an uploaded image cannot be accepted safely."""


class BundleValidationError(InferenceError):
    """Raised when a model bundle violates the frozen package contract."""


class PredictionError(InferenceError):
    """Raised when a backend returns an invalid classification result."""
