from unittest.mock import patch

import pytest

from article_extractor import ArticleExtractionError, extract_article, validate_article_url


def test_validate_article_url_accepts_http_and_https():
    assert validate_article_url("http://example.com/article") == "http://example.com/article"
    assert validate_article_url("https://example.com/article") == "https://example.com/article"


@pytest.mark.parametrize(
    "url",
    [
        "example.com/article",
        "ftp://example.com/article",
        "file:///C:/Windows/win.ini",
        "javascript:alert(1)",
    ],
)
def test_validate_article_url_rejects_invalid_or_unsupported_urls(url):
    with pytest.raises(ArticleExtractionError):
        validate_article_url(url)


def test_validate_article_url_rejects_empty_input():
    with pytest.raises(ArticleExtractionError, match="Please enter an article URL"):
        validate_article_url("")


def test_extract_article_raises_when_trafilatura_finds_no_article():
    with patch("article_extractor.fetch_webpage", return_value="<html></html>"):
        with patch("trafilatura.extract", return_value=None):
            with pytest.raises(ArticleExtractionError, match="Could not extract"):
                extract_article("https://example.com/article")


def test_extract_article_returns_clean_text_from_extractor():
    article = "This is a realistic article body. " * 12

    with patch("article_extractor.fetch_webpage", return_value="<html></html>"):
        with patch("trafilatura.extract", return_value=article):
            assert extract_article("https://example.com/article") == article.strip()
