"""Tests for the agent router."""
import pytest
from unittest.mock import patch, MagicMock
from backend.orchestration.router import route


def test_forced_agent_is_honoured():
    """If the user explicitly selects an agent, always use it."""
    assert route("anything", forced_agent="sql") == "sql"
    assert route("anything", forced_agent="rag") == "rag"
    assert route("anything", forced_agent="multimodal") == "multimodal"


def test_image_upload_routes_to_multimodal():
    result = route("What is in this image?", has_file=True, file_is_image=True)
    assert result == "multimodal"


def test_sql_keyword_routes_to_sql():
    # Should match via keyword regex without calling LLM
    result = route("What were the total sales last month?")
    assert result == "sql"


def test_rag_keyword_routes_to_rag_when_docs_loaded():
    result = route("What does the document say about refunds?", has_rag_documents=True)
    assert result == "rag"


@patch("backend.orchestration.router.client")
def test_llm_fallback_for_general_question(mock_client):
    """General question with no keywords → falls through to LLM classifier."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "multimodal"
    mock_client.chat.completions.create.return_value = mock_response

    result = route("What is the weather like today?")
    assert result == "multimodal"
    mock_client.chat.completions.create.assert_called_once()
