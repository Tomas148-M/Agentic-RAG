from typing import Any

import httpx

from app.core.config import (
    WEB_SEARCH_API_KEY_FIELD,
    WEB_SEARCH_CONTENT_KEY,
    WEB_SEARCH_DISABLED_ERROR,
    WEB_SEARCH_FAILED_ERROR,
    WEB_SEARCH_INCLUDE_ANSWER_FIELD,
    WEB_SEARCH_KEY_MISSING_ERROR,
    WEB_SEARCH_MAX_RESULTS_FIELD,
    WEB_SEARCH_QUERY_KEY,
    WEB_SEARCH_RESULTS_KEY,
    WEB_SEARCH_TITLE_KEY,
    WEB_SEARCH_URL_KEY,
    Settings,
)


class WebSearchError(Exception):
    """Raised when the configured web search provider fails."""


class WebSearchClient:
    """Search the public web through the Tavily search API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def search(self, query: str) -> list[dict[str, str]]:
        """Return normalized web search results.

        Args:
            query: Search query.

        Returns:
            Normalized results containing title, URL, and excerpt.

        Raises:
            WebSearchError: Search is disabled, unconfigured, or failed.
        """
        if not self.settings.web_search_enabled:
            raise WebSearchError(WEB_SEARCH_DISABLED_ERROR)
        if not self.settings.web_search_api_key:
            raise WebSearchError(WEB_SEARCH_KEY_MISSING_ERROR)
        try:
            response = httpx.post(
                self.settings.web_search_endpoint,
                json={
                    WEB_SEARCH_API_KEY_FIELD: self.settings.web_search_api_key,
                    WEB_SEARCH_QUERY_KEY: query,
                    WEB_SEARCH_MAX_RESULTS_FIELD:
                        self.settings.web_search_max_results,
                    WEB_SEARCH_INCLUDE_ANSWER_FIELD: False,
                },
                timeout=self.settings.tool_timeout_seconds,
            )
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WebSearchError(WEB_SEARCH_FAILED_ERROR) from exc
        results = (
            payload.get(WEB_SEARCH_RESULTS_KEY, [])
            if isinstance(payload, dict)
            else []
        )
        normalized = []
        for result in results:
            if not isinstance(result, dict):
                continue
            url = result.get(WEB_SEARCH_URL_KEY)
            content = result.get(WEB_SEARCH_CONTENT_KEY)
            if isinstance(url, str) and isinstance(content, str):
                normalized.append(
                    {
                        WEB_SEARCH_TITLE_KEY: str(
                            result.get(WEB_SEARCH_TITLE_KEY) or url
                        ),
                        WEB_SEARCH_URL_KEY: url,
                        WEB_SEARCH_CONTENT_KEY: content[
                            : self.settings.tool_result_max_chars
                        ],
                    }
                )
        return normalized
