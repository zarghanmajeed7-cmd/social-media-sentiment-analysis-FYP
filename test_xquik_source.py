"""Tests for optional Xquik source parsing."""

from __future__ import annotations

import json
import unittest
from urllib import request

from xquik_source import XquikSourceError, fetch_x_posts


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class XquikSourceTests(unittest.TestCase):
    def test_fetches_posts_from_tweets_key(self) -> None:
        def opener(api_request: request.Request, timeout: int) -> FakeResponse:
            self.assertEqual(timeout, 15)
            self.assertEqual(api_request.get_header("X-api-key"), "test-key")
            self.assertIn("q=cricket+sentiment", api_request.full_url)
            return FakeResponse(
                {
                    "tweets": [
                        {
                            "text": "Great match energy today",
                            "url": "https://x.com/example/status/1",
                            "createdAt": "2026-06-21T00:00:00Z",
                        },
                        {"text": "Tough loss but strong fans"},
                    ]
                }
            )

        posts = fetch_x_posts("cricket sentiment", api_key="test-key", opener=opener)

        self.assertEqual(posts[0]["text"], "Great match energy today")
        self.assertEqual(posts[0]["url"], "https://x.com/example/status/1")
        self.assertEqual(posts[1]["text"], "Tough loss but strong fans")

    def test_fetches_nested_legacy_text(self) -> None:
        def opener(api_request: request.Request, timeout: int) -> FakeResponse:
            self.assertEqual(api_request.get_header("X-api-key"), "test-key")
            return FakeResponse({"data": {"results": [{"legacy": {"full_text": "Policy update reactions"}}]}})

        posts = fetch_x_posts("policy", api_key="test-key", opener=opener)

        self.assertEqual(posts, [{"text": "Policy update reactions", "url": "", "published": ""}])

    def test_deduplicates_and_limits_posts(self) -> None:
        def opener(api_request: request.Request, timeout: int) -> FakeResponse:
            self.assertEqual(api_request.get_method(), "GET")
            return FakeResponse(["same", "same", "different"])

        posts = fetch_x_posts("topic", api_key="test-key", opener=opener, limit=2)

        self.assertEqual([post["text"] for post in posts], ["same", "different"])

    def test_requires_api_key(self) -> None:
        with self.assertRaisesRegex(XquikSourceError, "XQUIK_API_KEY"):
            fetch_x_posts("topic", api_key="")

    def test_reports_empty_results(self) -> None:
        def opener(api_request: request.Request, timeout: int) -> FakeResponse:
            self.assertEqual(api_request.get_method(), "GET")
            return FakeResponse({"tweets": []})

        with self.assertRaisesRegex(XquikSourceError, "No post text"):
            fetch_x_posts("topic", api_key="test-key", opener=opener)


if __name__ == "__main__":
    unittest.main()
