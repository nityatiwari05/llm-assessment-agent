import os
from unittest.mock import patch

import pytest

from app.llm_client import LLMError, call_agent, call_llm


@patch("app.llm_client.requests.post")
def test_call_llm_uses_ollama_when_configured(mock_post):
    mock_post.return_value.raise_for_status.return_value = None
    mock_post.return_value.json.return_value = {"message": {"content": "{}"}}

    with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=False):
        result = call_llm([{"role": "user", "content": "hello"}])

    assert result == "{}"
    mock_post.assert_called_once()


def test_call_agent_rejects_unsupported_provider():
    with patch.dict(os.environ, {"LLM_PROVIDER": "unsupported"}, clear=False):
        with pytest.raises(LLMError):
            call_agent("system", [{"role": "user", "content": "hello"}])
