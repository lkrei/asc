from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

from license_rules import is_allowed_license
from providers import (
    OpenverseProvider,
    SearchResult,
    WikimediaCommonsProvider,
    create_session,
)


IMAGE_EXTENSIONS = {
    "jpg": ".jpg",
    "jpeg": ".jpg",
    "png": ".png",
    "webp": ".webp",
    "gif": ".gif",
    "tif": ".tif",
    "tiff": ".tif",
}


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[-\s]+", "_", value)
    return value or "style"


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^\w\s.-]", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value.strip())
    return value[:80] or "image"


def build_queries(style: str, extra_queries: Iterable[str]) -> list[str]:
    normalized_style = style.strip()
    lowered = normalized_style.lower()
    templates = [normalized_style]

    if "architecture" not in lowered:
        templates.append(f"{normalized_style} architecture")
    templates.append(f"{normalized_style} facade")
    templates.append(f"{normalized_style} building")

    queries: list[str] = []
    for query in templates:
        query = query.strip()
        if query and query not in queries:
            queries.append(query)
    for item in extra_queries:
        query = item.strip()
        if query and query not in queries:
            queries.append(query)
    return queries


def choose_extension(item: SearchResult, response: requests.Response) -> str:
    if item.filetype:
        normalized = item.filetype.lower().lstrip(".")
        if normalized in IMAGE_EXTENSIONS:
            return IMAGE_EXTENSIONS[normalized]

    content_type = (response.headers.get("Content-Type") or "").lower()
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "gif" in content_type:
        return ".gif"
    if "tiff" in content_type:
        return ".tif"

    path = urlparse(item.image_url).path.lower()
    for key, extension in IMAGE_EXTENSIONS.items():
        if path.endswith(f".{key}"):
            return extension
    return ".jpg"


def validate_download(path: Path, *, min_width: int, min_height: int) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"invalid image: {exc}") from exc

    if width < min_width or height < min_height:
        raise ValueError(f"image too small: {width}x{height}")
    return width, height


def download_file(
    session: requests.Session,
    item: SearchResult,
    destination_dir: Path,
    *,
    min_width: int,
    min_height: int,
    timeout: int,
    max_filesize_mb: int,
    seen_hashes: set[str],
) -> dict | None:
    for attempt in range(1, 4):
        temp_path: Path | None = None
        try:
            response = session.get(item.image_url, stream=True, timeout=timeout)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_seconds = max(int(retry_after), 5) if retry_after else 5 * attempt
                except ValueError:
                    wait_seconds = 5 * attempt
                raise requests.HTTPError(
                    f"429 Too Many Requests, retry after {wait_seconds}s",
                    response=response,
                )
            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > max_filesize_mb:
                    return None

            extension = choose_extension(item, response)
            base_name = sanitize_filename(item.title)
            temp_path = destination_dir / f"{base_name}.part"
            digest = hashlib.sha256()
            downloaded_bytes = 0
            max_bytes = max_filesize_mb * 1024 * 1024

            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if not chunk:
                        continue
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > max_bytes:
                        raise ValueError(
                            f"file exceeds size limit ({max_filesize_mb} MB)"
                        )
                    digest.update(chunk)
                    handle.write(chunk)

            sha256 = digest.hexdigest()
            if sha256 in seen_hashes:
                temp_path.unlink(missing_ok=True)
                return None

            final_name = f"{base_name}_{sha256[:12]}{extension}"
            final_path = destination_dir / final_name
            temp_path.rename(final_path)
            width, height = validate_download(
                final_path,
                min_width=min_width,
                min_height=min_height,
            )
            seen_hashes.add(sha256)

            return {
                **item.to_dict(),
                "saved_path": str(final_path),
                "sha256": sha256,
                "downloaded_width": width,
                "downloaded_height": height,
                "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        except Exception as exc:  # noqa: BLE001
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            if isinstance(exc, requests.HTTPError) and getattr(exc, "response", None) is not None:
                response = exc.response
                if response is not None and response.status_code == 429 and attempt < 3:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        wait_seconds = max(int(retry_after), 5) if retry_after else 5 * attempt
                    except ValueError:
                        wait_seconds = 5 * attempt
                    time.sleep(wait_seconds)
                    continue
            if attempt == 3:
                return None
            time.sleep(1.5 * attempt)

    return None


def collect_candidates(
    *,
    providers: list[str],
    queries: list[str],
    target_count: int,
    per_query_limit: int,
    max_pages: int,
    allow_nc: bool,
    allow_nd: bool,
    timeout: int,
    wikimedia_thumb_width: int,
) -> list[SearchResult]:
    session = create_session()
    provider_map = {
        "openverse": OpenverseProvider(
            session,
            allow_nc=allow_nc,
            allow_nd=allow_nd,
            timeout=timeout,
        ),
        "wikimedia": WikimediaCommonsProvider(
            session,
            allow_nc=allow_nc,
            allow_nd=allow_nd,
            timeout=timeout,
            thumb_width=wikimedia_thumb_width,
        ),
    }

    all_items: list[SearchResult] = []
    seen_keys: set[tuple[str, str]] = set()

    for provider_name in providers:
        provider = provider_map[provider_name]
        for query in queries:
            remaining = target_count - len(all_items)
            if remaining <= 0:
                break

            try:
                found = provider.search(
                    query,
                    limit=min(per_query_limit, remaining),
                    max_pages=max_pages,
                )
            except Exception:  # noqa: BLE001
                continue

            for item in found:
                key = (item.provider, item.image_url)
                if key in seen_keys:
                    continue
                if not is_allowed_license(
                    item.license_name,
                    allow_nc=allow_nc,
                    allow_nd=allow_nd,
                ):
                    continue
                seen_keys.add(key)
                all_items.append(item)
                if len(all_items) >= target_count:
                    break
            time.sleep(0.4)

    return all_items


def write_metadata_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metadata_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("style")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--providers", default="wikimedia")
    parser.add_argument("--output-dir", default="downloads")
    parser.add_argument("--per-query-limit", type=int, default=20)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--min-width", type=int, default=512)
    parser.add_argument("--min-height", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--max-filesize-mb", type=int, default=25)
    parser.add_argument("--download-delay", type=float, default=1.2)
    parser.add_argument("--wikimedia-thumb-width", type=int, default=1600)
    parser.add_argument("--extra-query", action="append", default=[])
    parser.add_argument("--allow-nc", action="store_true")
    parser.add_argument("--allow-nd", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    supported = {"openverse", "wikimedia"}
    unknown = [item for item in providers if item not in supported]
    if unknown:
        print(f"Неизвестные провайдеры: {', '.join(unknown)}", file=sys.stderr)
        return 2

    style_slug = slugify(args.style)
    base_output = Path(args.output_dir).resolve()
    run_dir = base_output / style_slug
    images_dir = run_dir / "images"
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    queries = build_queries(args.style, args.extra_query)

    candidates = collect_candidates(
        providers=providers,
        queries=queries,
        target_count=max(args.limit * 2, args.limit),
        per_query_limit=args.per_query_limit,
        max_pages=args.max_pages,
        allow_nc=args.allow_nc,
        allow_nd=args.allow_nd,
        timeout=args.timeout,
        wikimedia_thumb_width=args.wikimedia_thumb_width,
    )

    summary = {
        "style": args.style,
        "style_slug": style_slug,
        "providers": providers,
        "queries": queries,
        "requested_limit": args.limit,
        "dry_run": args.dry_run,
        "allow_nc": args.allow_nc,
        "allow_nd": args.allow_nd,
        "min_width": args.min_width,
        "min_height": args.min_height,
        "max_filesize_mb": args.max_filesize_mb,
        "download_delay": args.download_delay,
        "wikimedia_thumb_width": args.wikimedia_thumb_width,
        "collected_candidates": len(candidates),
    }

    if args.dry_run:
        preview_path = run_dir / "candidates.json"
        with preview_path.open("w", encoding="utf-8") as handle:
            json.dump([item.to_dict() for item in candidates], handle, ensure_ascii=False, indent=2)
        with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        return 0

    session = create_session()
    seen_hashes: set[str] = set()
    downloaded: list[dict] = []

    for item in candidates:
        if len(downloaded) >= args.limit:
            break

        if item.width and item.width < args.min_width:
            continue
        if item.height and item.height < args.min_height:
            continue

        record = download_file(
            session,
            item,
            images_dir,
            min_width=args.min_width,
            min_height=args.min_height,
            timeout=args.timeout,
            max_filesize_mb=args.max_filesize_mb,
            seen_hashes=seen_hashes,
        )
        if not record:
            continue

        downloaded.append(record)
        time.sleep(args.download_delay)

    metadata_jsonl = run_dir / "metadata.jsonl"
    metadata_csv = run_dir / "metadata.csv"
    write_metadata_jsonl(downloaded, metadata_jsonl)
    write_metadata_csv(downloaded, metadata_csv)

    summary.update(
        {
            "downloaded_count": len(downloaded),
            "metadata_jsonl": str(metadata_jsonl),
            "metadata_csv": str(metadata_csv),
            "images_dir": str(images_dir),
        }
    )
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return 0 if downloaded else 1


if __name__ == "__main__":
    raise SystemExit(main())
