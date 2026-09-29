"""Backward-compatible alias for the active verification module."""

from app.verification import (
    ClaimVerification,
    VerificationResult,
    decide_result,
    verify_answer,
)

__all__ = [
    "ClaimVerification",
    "VerificationResult",
    "verify_answer",
    "decide_result",
]