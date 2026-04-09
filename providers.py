from __future__ import annotations

import html
import re
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import requests

from license_rules import is_allowed_license, openverse_license_filter


USER_AGENT = "ArchStyleDatasetCollector/1.0 (research use; contact: local-user)"


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class SearchResult:
    provider: str
    source: str
    query: str
    title: str
    image_url: str
    landing_url: str
    license_name: str
    license_url: str
    creator: str
    creator_url: str
    attribution: str
    width: int | None = None
    height: int | None = None
    filetype: str | None = None
    external_id: str | None = None
    original_image_url: str | None = None
    download_strategy: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class OpenverseProvider:
    base_url = "https://api.openverse.org/v1/images/"

    def __init__(
        self,
        session: requests.Session,
        *,
        allow_nc: bool = False,
        allow_nd: bool = False,
        timeout: int = 30,
    ) -> None:
        self.session = session
        self.allow_nc = allow_nc
        self.allow_nd = allow_nd
        self.timeout = timeout

    def search(
        self,
        query: str,
        *,
        limit: int,
        per_page: int = 20,
        max_pages: int = 5,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []

        for page in range(1, max_pages + 1):
            if len(results) >= limit:
                break

            params = {
                "q": query,
                "page": page,
                "page_size": min(per_page, limit - len(results)),
                "license": openverse_license_filter(
                    allow_nc=self.allow_nc,
                    allow_nd=self.allow_nd,
                ),
            }
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()

            page_results = payload.get("results", [])
            if not page_results:
                break

            for item in page_results:
                license_name = item.get("license")
                if not is_allowed_license(
                    license_name,
                    allow_nc=self.allow_nc,
                    allow_nd=self.allow_nd,
                ):
                    continue

                image_url = item.get("url")
                landing_url = item.get("foreign_landing_url")
                if not image_url or not landing_url:
                    continue

                results.append(
                    SearchResult(
                        provider="openverse",
                        source=item.get("source") or item.get("provider") or "openverse",
                        query=query,
                        title=(item.get("title") or "").strip() or "untitled",
                        image_url=image_url,
                        landing_url=landing_url,
                        license_name=str(license_name or "").strip(),
                        license_url=str(item.get("license_url") or "").strip(),
                        creator=str(item.get("creator") or "").strip(),
                        creator_url=str(item.get("creator_url") or "").strip(),
                        attribution=str(item.get("attribution") or "").strip(),
                        width=item.get("width"),
                        height=item.get("height"),
                        filetype=item.get("filetype"),
                        external_id=str(item.get("id") or ""),
                    )
                )

                if len(results) >= limit:
                    break

            time.sleep(0.4)

        return results


class WikimediaCommonsProvider:
    api_url = "https://commons.wikimedia.org/w/api.php"

    def __init__(
        self,
        session: requests.Session,
        *,
        allow_nc: bool = False,
        allow_nd: bool = False,
        timeout: int = 30,
        thumb_width: int = 1600,
    ) -> None:
        self.session = session
        self.allow_nc = allow_nc
        self.allow_nd = allow_nd
        self.timeout = timeout
        self.thumb_width = thumb_width

    def search(
        self,
        query: str,
        *,
        limit: int,
        per_page: int = 20,
        max_pages: int = 5,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        offset = 0

        for _ in range(max_pages):
            if len(results) >= limit:
                break

            payload = self._search_titles(query=query, limit=min(per_page, limit - len(results)), offset=offset)
            titles = [item["title"] for item in payload.get("query", {}).get("search", []) if item.get("title")]
            if not titles:
                break

            info_results = self._fetch_file_info(titles=titles, query=query)
            for item in info_results:
                results.append(item)
                if len(results) >= limit:
                    break

            offset = payload.get("continue", {}).get("sroffset")
            if offset is None:
                break
            time.sleep(0.7)

        return results

    def _search_titles(self, *, query: str, limit: int, offset: int) -> dict:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"{query} filetype:bitmap",
            "srnamespace": 6,
            "srlimit": limit,
            "sroffset": offset,
            "format": "json",
        }
        response = self.session.get(self.api_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _fetch_file_info(self, *, titles: Iterable[str], query: str) -> list[SearchResult]:
        params = {
            "action": "query",
            "titles": "|".join(titles),
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": self.thumb_width,
            "format": "json",
        }
        response = self.session.get(self.api_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        results: list[SearchResult] = []
        pages = payload.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                continue

            image_info = (page.get("imageinfo") or [None])[0]
            if not image_info:
                continue

            metadata = image_info.get("extmetadata") or {}
            license_name = strip_html((metadata.get("LicenseShortName") or {}).get("value"))
            license_url = strip_html((metadata.get("LicenseUrl") or {}).get("value"))
            usage_terms = strip_html((metadata.get("UsageTerms") or {}).get("value"))
            creator = strip_html((metadata.get("Artist") or {}).get("value"))
            title = strip_html((metadata.get("ObjectName") or {}).get("value")) or page.get("title", "")
            attribution = usage_terms or f"{title} / {creator}".strip(" /")

            if not is_allowed_license(
                license_name or usage_terms,
                allow_nc=self.allow_nc,
                allow_nd=self.allow_nd,
            ):
                continue

            original_image_url = image_info.get("url")
            image_url = image_info.get("thumburl") or original_image_url
            landing_url = image_info.get("descriptionurl")
            if not image_url or not landing_url:
                continue

            download_width = image_info.get("thumbwidth") or image_info.get("width")
            download_height = image_info.get("thumbheight") or image_info.get("height")
            download_strategy = "wikimedia_thumbnail" if image_info.get("thumburl") else "original"

            results.append(
                SearchResult(
                    provider="wikimedia",
                    source="wikimedia",
                    query=query,
                    title=title or page.get("title", "untitled"),
                    image_url=image_url,
                    landing_url=landing_url,
                    license_name=license_name or usage_terms,
                    license_url=license_url,
                    creator=creator,
                    creator_url="",
                    attribution=attribution,
                    width=download_width,
                    height=download_height,
                    filetype=_infer_filetype_from_url(image_url),
                    external_id=str(page.get("pageid") or ""),
                    original_image_url=original_image_url,
                    download_strategy=download_strategy,
                )
            )

        return results


def _infer_filetype_from_url(url: str) -> str:
    lower = url.lower()
    if lower.endswith(".jpeg"):
        return "jpeg"
    if "." in lower.rsplit("/", 1)[-1]:
        return lower.rsplit(".", 1)[-1].split("?", 1)[0]
    return "jpg"
