"""
Tag utility functions.
"""

from collections.abc import Sequence


def _normalize_tags(tags: Sequence[str] | None) -> list[str]:
    if not tags:
        return []
    normalized: list[str] = []
    for tag in tags:
        if tag is None:
            continue
        tag_text = str(tag).strip()
        if tag_text and tag_text not in normalized:
            normalized.append(tag_text)
    return normalized


def _merge_tags(existing: Sequence[str], new_tags: Sequence[str] | None) -> list[str]:
    merged = list(existing)
    for tag in _normalize_tags(new_tags):
        if tag not in merged:
            merged.append(tag)
    return merged
