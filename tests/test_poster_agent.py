"""Tests for agents/poster_agent.py.

No real network calls or API key needed — requests.get is mocked. What's
tested: the function returns a real poster URL on a successful TMDB
response, and degrades to None (never raises) on every failure mode a
live deployment could actually hit — missing API key, no results, a
result with no poster_path, and a network/HTTP error.
"""

from unittest.mock import MagicMock, patch

import pytest

from agents.poster_agent import _fetch_poster_sync, fetch_poster


class TestFetchPosterSync:
    def test_returns_none_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("TMDB_API_KEY", raising=False)
        assert _fetch_poster_sync("RRR", 2022) is None

    def test_returns_none_when_title_empty(self, monkeypatch):
        monkeypatch.setenv("TMDB_API_KEY", "fake-key")
        assert _fetch_poster_sync("", 2022) is None

    @patch("agents.poster_agent.requests.get")
    def test_returns_full_poster_url_on_success(self, mock_get, monkeypatch):
        monkeypatch.setenv("TMDB_API_KEY", "fake-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"poster_path": "/abc123.jpg"}]}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = _fetch_poster_sync("RRR", 2022)

        assert result == "https://image.tmdb.org/t/p/w500/abc123.jpg"

    @patch("agents.poster_agent.requests.get")
    def test_passes_year_when_provided(self, mock_get, monkeypatch):
        monkeypatch.setenv("TMDB_API_KEY", "fake-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        _fetch_poster_sync("RRR", 2022)

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["year"] == 2022

    @patch("agents.poster_agent.requests.get")
    def test_returns_none_on_empty_results(self, mock_get, monkeypatch):
        monkeypatch.setenv("TMDB_API_KEY", "fake-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        assert _fetch_poster_sync("Some Obscure Title", None) is None

    @patch("agents.poster_agent.requests.get")
    def test_returns_none_when_result_has_no_poster_path(self, mock_get, monkeypatch):
        monkeypatch.setenv("TMDB_API_KEY", "fake-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"poster_path": None}]}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        assert _fetch_poster_sync("RRR", 2022) is None

    @patch("agents.poster_agent.requests.get")
    def test_degrades_to_none_on_network_failure_never_raises(self, mock_get, monkeypatch):
        monkeypatch.setenv("TMDB_API_KEY", "fake-key")
        mock_get.side_effect = ConnectionError("network down")

        assert _fetch_poster_sync("RRR", 2022) is None

    @patch("agents.poster_agent.requests.get")
    def test_degrades_to_none_on_http_error_never_raises(self, mock_get, monkeypatch):
        monkeypatch.setenv("TMDB_API_KEY", "fake-key")
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")
        mock_get.return_value = mock_response

        assert _fetch_poster_sync("RRR", 2022) is None


class TestFetchPosterAsync:
    def test_async_wrapper_delegates_to_sync(self, monkeypatch):
        import asyncio
        monkeypatch.delenv("TMDB_API_KEY", raising=False)
        result = asyncio.run(fetch_poster("RRR", 2022))
        assert result is None
