from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


FETCH_TIMEOUT_SECONDS = 12
MAX_RESPONSE_BYTES = 2_000_000
MIN_ARTICLE_CHARS = 200


class ArticleExtractionError(Exception):
    """Raised when a URL cannot produce usable article text."""


def validate_article_url(url):
    if not url or not url.strip():
        raise ArticleExtractionError("Please enter an article URL.")

    parsed = urlparse(url.strip())

    if parsed.scheme not in {"http", "https"}:
        raise ArticleExtractionError("Only http:// and https:// URLs are supported.")

    if not parsed.netloc:
        raise ArticleExtractionError("Please enter a valid article URL.")

    return parsed.geturl()


def fetch_webpage(url):
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get("Content-Type", "").lower()

            if not (
                "text/html" in content_type
                or "application/xhtml+xml" in content_type
                or "text/plain" in content_type
            ):
                raise ArticleExtractionError(
                    "This URL does not appear to be an article page."
                )

            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read(MAX_RESPONSE_BYTES + 1)

    except HTTPError as exc:
        raise ArticleExtractionError(
            f"Could not fetch this URL. The server returned HTTP {exc.code}."
        ) from exc
    except URLError as exc:
        raise ArticleExtractionError(
            "Could not fetch this URL. Check the address or try another article."
        ) from exc
    except TimeoutError as exc:
        raise ArticleExtractionError("The URL took too long to respond.") from exc

    if len(body) > MAX_RESPONSE_BYTES:
        raise ArticleExtractionError("This page is too large to process safely.")

    return body.decode(charset, errors="replace")


def extract_article(url):
    url = validate_article_url(url)
    html = fetch_webpage(url)

    try:
        import trafilatura
    except ImportError as exc:
        raise ArticleExtractionError(
            "Article extraction dependency is missing. Install trafilatura."
        ) from exc

    try:
        article = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
    except Exception as exc:
        raise ArticleExtractionError(
            "Could not extract an article from this URL."
        ) from exc

    if not article or len(article.strip()) < MIN_ARTICLE_CHARS:
        raise ArticleExtractionError("Could not extract an article from this URL.")

    return article.strip()
