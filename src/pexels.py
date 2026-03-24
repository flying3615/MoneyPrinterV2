"""
Pexels video search and download utilities.
"""

import os
import random
import requests
from urllib.parse import urlencode
from uuid import uuid4

from config import ROOT_DIR
from status import info, warning


def search_videos(
    search_term: str,
    api_key: str,
    orientation: str = "portrait",
    min_duration: int = 3,
    per_page: int = 15,
) -> list[dict]:
    """
    Search Pexels for videos matching the search term.

    Returns a list of dicts with 'url' and 'duration'.
    """
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0",
    }
    params = {"query": search_term, "per_page": per_page, "orientation": orientation}
    url = f"https://api.pexels.com/videos/search?{urlencode(params)}"

    try:
        r = requests.get(url, headers=headers, timeout=(10, 30), verify=False)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        warning(f"[Pexels] Search failed for '{search_term}': {e}")
        return []

    results = []
    for v in data.get("videos", []):
        if v["duration"] < min_duration:
            continue
        # Pick best matching resolution file (prefer 1080x1920 portrait)
        best = None
        for f in v["video_files"]:
            w, h = f.get("width") or 0, f.get("height") or 0
            if orientation == "portrait" and w == 1080 and h == 1920:
                best = f
                break
            if orientation == "landscape" and w == 1920 and h == 1080:
                best = f
                break
        # Fallback: pick highest-res file
        if not best:
            best = max(
                v["video_files"],
                key=lambda f: (f.get("width") or 0) * (f.get("height") or 0),
                default=None,
            )
        if best and best.get("link"):
            results.append({"url": best["link"], "duration": v["duration"]})

    info(f"[Pexels] '{search_term}' → {len(results)} videos found")
    return results


def download_video(url: str, save_dir: str) -> str:
    """Download a single video URL to save_dir, return local path."""
    os.makedirs(save_dir, exist_ok=True)
    dest = os.path.join(save_dir, str(uuid4()) + ".mp4")
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(15, 120), verify=False) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    f.write(chunk)
        info(f"[Pexels] Downloaded → {dest}")
        return dest
    except Exception as e:
        warning(f"[Pexels] Download failed: {e}")
        return ""


def fetch_clips_for_keywords(
    keywords: list[str],
    api_key: str,
    tts_duration: float,
    clip_duration: int = 5,
    orientation: str = "portrait",
) -> list[str]:
    """
    Search Pexels for each keyword, download enough clips to cover
    tts_duration seconds, return list of local video paths.
    """
    save_dir = os.path.join(ROOT_DIR, ".mp", "pexels_cache")
    all_results = []

    for kw in keywords:
        results = search_videos(kw, api_key, orientation=orientation)
        all_results.extend(results)

    if not all_results:
        warning("[Pexels] No videos found for any keyword")
        return []

    random.shuffle(all_results)

    paths = []
    total = 0.0
    for item in all_results:
        path = download_video(item["url"], save_dir)
        if path:
            paths.append(path)
            total += min(clip_duration, item["duration"])
            if total >= tts_duration:
                break

    info(f"[Pexels] Downloaded {len(paths)} clips covering {total:.1f}s")
    return paths
