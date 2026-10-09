
import pytest
from pydantic import ValidationError

from app.config import Settings


def test_valid_configuration():
    config = Settings(
        _env_file=None,
        openai_model="gpt-4o",
        openai_timeout=10,
        openai_max_retries=2,
    )

    assert config.openai_model == "gpt-4o"
    assert config.openai_timeout == 10
    assert config.openai_max_retries == 2


def test_timeout_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            openai_timeout=-5,
        )


def test_retry_count_cannot_be_negative():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            openai_max_retries=-1,
        )
