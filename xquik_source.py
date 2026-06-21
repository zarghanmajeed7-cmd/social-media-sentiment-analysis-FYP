"""Optional Xquik tweet search helpers."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, parse, request

API_URL = "https://xquik.com/api/v1/x/tweets/search"
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_LIMIT = 25


class XquikSourceError(RuntimeError):
    """Raised when Xquik tweet search cannot load usable text."""


def fetch_x_posts(
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    api_key: str | None = None,
    opener: Any | None = None,
) -> list[dict[str, str]]:
    """Fetch X posts for a search query and return text-first records."""
    normalized_query = query.strip()
    if len(normalized_query) < 2:
        raise XquikSourceError("Enter at least 2 characters for the X search query.")

    resolved_key = (api_key or os.getenv("XQUIK_API_KEY") or "").strip()
    if not resolved_key:
        raise XquikSourceError("Set XQUIK_API_KEY before loading X posts.")

    payload = _load_payload(normalized_query, resolved_key, opener)
    posts: list[dict[str, str]] = []
    seen_texts: set[str] = set()

    for candidate in _iter_candidates(payload):
        record = _extract_record(candidate)
        text = record["text"]
        if text and text not in seen_texts:
            posts.append(record)
            seen_texts.add(text)
        if len(posts) >= limit:
            break

    if not posts:
        raise XquikSourceError("No post text was returned for that query.")

    return posts


def _load_payload(query: str, api_key: str, opener: Any | None) -> Any:
    url = f"{API_URL}?{parse.urlencode({'q': query})}"
    api_request = request.Request(
        url,
        headers={
            "accept": "application/json",
            "x-api-key": api_key,
        },
        method="GET",
    )
    open_request = opener or request.urlopen

    try:
        with open_request(api_request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            raw_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raise XquikSourceError(f"Xquik request failed with HTTP {exc.code}.") from exc
    except error.URLError as exc:
        raise XquikSourceError("Xquik request could not connect.") from exc
    except TimeoutError as exc:
        raise XquikSourceError("Xquik request timed out.") from exc

    try:
        return json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise XquikSourceError("Xquik returned invalid JSON.") from exc


def _iter_candidates(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    for key in ("tweets", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    data = payload.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return _iter_candidates(data)

    return [payload]


def _extract_record(candidate: Any) -> dict[str, str]:
    if isinstance(candidate, str):
        return {"text": candidate.strip(), "url": "", "published": ""}

    if not isinstance(candidate, dict):
        return {"text": "", "url": "", "published": ""}

    text = _first_string(
        candidate,
        ("text", "fullText", "full_text", "tweetText", "content", "rawContent"),
    )
    url = _first_string(candidate, ("url", "tweetUrl", "link", "permalink"))
    published = _first_string(candidate, ("createdAt", "created_at", "published", "date"))

    legacy = candidate.get("legacy")
    if not text and isinstance(legacy, dict):
        text = _first_string(legacy, ("full_text", "text"))
        published = published or _first_string(legacy, ("created_at",))

    return {"text": text, "url": url, "published": published}


def _first_string(source: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
