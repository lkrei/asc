from __future__ import annotations

import re
from typing import Iterable


DEFAULT_ALLOWED_LICENSES = {
    "by",
    "by-sa",
    "cc0",
    "pdm",
    "public domain",
    "publicdomain",
    "pd",
    "cc by",
    "cc by-sa",
    "cc-by",
    "cc-by-sa",
    "gfdl",
    "usgov",
}

NC_LICENSE_MARKERS = ("by-nc", "nc", "noncommercial")
ND_LICENSE_MARKERS = ("by-nd", "nd", "no derivatives", "noderivs")


def normalize_license_name(value: str | None) -> str:
    if not value:
        return ""
    text = value.strip().lower()
    text = text.replace("_", "-")
    text = text.replace("creative commons", "cc")
    text = text.replace("attribution", "by")
    text = text.replace("share alike", "by-sa")
    text = text.replace("sharealike", "by-sa")
    text = text.replace("public domain mark", "pdm")
    text = text.replace("public domain dedication", "cc0")
    text = re.sub(r"\s+", " ", text)
    return text


def is_allowed_license(
    license_name: str | None,
    *,
    allow_nc: bool = False,
    allow_nd: bool = False,
    extra_allowed: Iterable[str] | None = None,
) -> bool:
    normalized = normalize_license_name(license_name)
    if not normalized:
        return False

    if not allow_nc and any(marker in normalized for marker in NC_LICENSE_MARKERS):
        return False
    if not allow_nd and any(marker in normalized for marker in ND_LICENSE_MARKERS):
        return False

    allowed = set(DEFAULT_ALLOWED_LICENSES)
    if extra_allowed:
        allowed.update(normalize_license_name(item) for item in extra_allowed)

    if normalized in allowed:
        return True
    return any(token in normalized for token in allowed)


def openverse_license_filter(*, allow_nc: bool = False, allow_nd: bool = False) -> str:
    licenses = ["by", "by-sa", "cc0", "pdm"]
    if allow_nd:
        licenses.append("by-nd")
    if allow_nc:
        licenses.extend(["by-nc", "by-nc-sa"])
        if allow_nd:
            licenses.append("by-nc-nd")
    return ",".join(licenses)
