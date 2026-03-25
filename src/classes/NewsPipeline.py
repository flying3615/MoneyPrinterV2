"""
NewsPipeline
------------
Extracts news articles from PDFs by preferring embedded text and
falling back to OCR for scanned pages, then matches them against
Google Trends keywords, rewrites the top articles as short-video
scripts, and feeds each script into the existing YouTube pipeline.
"""

import io
import re
import json
import base64
import subprocess
import requests
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageFilter, ImageOps
from openai import OpenAI
from pypdf import PdfReader
from zhipuai import ZhipuAI

from config import (
    ROOT_DIR,
    get_nanobanana2_api_base_url,
    get_nanobanana2_api_key,
    get_openai_api_key,
    get_openai_base_url,
    get_openai_model,
    get_zhipu_api_key,
    get_google_trends_geo,
)
from status import info, warning, success, error


VISION_MODEL = "glm-4.6v"
DEFAULT_TEXT_MODEL = "MiniMax-M2.7"
DEFAULT_CLEANER_MODEL = "gemini-2.5-flash"
MIN_DIRECT_TEXT_LENGTH = 500
NEWS_PIPELINE_DEBUG_DIR = f"{ROOT_DIR}/.mp/news_pipeline"


def _glm_client() -> ZhipuAI:
    api_key = get_zhipu_api_key()
    if not api_key:
        raise RuntimeError("zhipu_api_key is not set in config.json")
    return ZhipuAI(api_key=api_key, base_url="https://api.z.ai/api/coding/paas/v4/")


def _openai_client() -> OpenAI:
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "openai_api_key is not set in config.json and OPENAI_API_KEY is not set"
        )
    base_url = get_openai_base_url() or None
    return OpenAI(api_key=api_key, base_url=base_url)


def _text_model() -> str:
    configured = (get_openai_model() or "").strip()
    return configured or DEFAULT_TEXT_MODEL


def _openai_chat(messages: list[dict], model: str | None = None):
    client = _openai_client()
    return client.chat.completions.create(
        model=model or _text_model(),
        messages=messages,
        extra_body={"reasoning_split": True},
    )


def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _ensure_debug_dir() -> None:
    import os

    os.makedirs(NEWS_PIPELINE_DEBUG_DIR, exist_ok=True)


def _write_debug_text(page_num: int, text: str, source: str) -> None:
    _ensure_debug_dir()
    file_path = f"{NEWS_PIPELINE_DEBUG_DIR}/page-{page_num:03d}-{source}.txt"
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)


def _write_debug_json(filename: str, payload) -> None:
    _ensure_debug_dir()
    file_path = f"{NEWS_PIPELINE_DEBUG_DIR}/{filename}"
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_article_text(text: str) -> str:
    text = text.replace("\r", "\n").replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _infer_source_name(pdf_path: str, sample_text: str = "") -> str:
    candidates = {
        "wall street journal": "The Wall Street Journal",
        "华尔街日报": "The Wall Street Journal",
        "new york times": "The New York Times",
        "guardian": "The Guardian",
        "financial times": "Financial Times",
        "los angeles times": "Los Angeles Times",
    }
    haystack = f"{Path(pdf_path).name.lower()} {sample_text.lower()}"
    for needle, label in candidates.items():
        if needle in haystack:
            return label
    return Path(pdf_path).stem


def _infer_publish_date(sample_text: str) -> str:
    match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        sample_text,
    )
    if not match:
        return ""
    try:
        return datetime.strptime(match.group(0), "%B %d, %Y").date().isoformat()
    except ValueError:
        return ""


def _classify_newspaper_article(article: dict) -> str:
    section = article.get("section", "").strip().lower()
    title = article.get("title", "").strip().lower()

    if "letter" in section:
        return "letter"
    if any(token in section for token in ("bookshelf", "review", "feature")):
        return "review_or_feature"
    if any(token in title for token in ("review", "bookshelf")):
        return "review_or_feature"
    if any(token in section for token in ("opinion", "editorial", "column", "analysis")):
        return "opinion_or_analysis"
    return "news"


def _annotate_newspaper_article(
    article: dict, source_name: str, publish_date: str
) -> dict:
    article["source_type"] = "newspaper"
    article["source_name"] = source_name
    article["publish_date"] = publish_date
    article["article_type"] = _classify_newspaper_article(article)
    return article


def _gemini_generate_json(prompt: str):
    api_key = get_nanobanana2_api_key()
    if not api_key:
        raise RuntimeError("Gemini API key is not configured")

    url = (
        f"{get_nanobanana2_api_base_url().rstrip('/')}"
        f"/models/{DEFAULT_CLEANER_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    body = response.json()
    raw = (
        body.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )
    return json.loads(_strip_code_fence(raw))


def _openai_generate_json(prompt: str, system_prompt: str | None = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = _openai_chat(messages)
    return json.loads(_strip_code_fence(response.choices[0].message.content or ""))


def _looks_like_meaningful_text(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", text)
    if len(cleaned) < MIN_DIRECT_TEXT_LENGTH:
        return False

    alpha_numeric = sum(char.isalnum() for char in cleaned)
    return alpha_numeric / max(len(cleaned), 1) >= 0.6


def _looks_like_english_text(text: str) -> bool:
    if not _looks_like_meaningful_text(text):
        return False
    letters = sum(char.isalpha() for char in text)
    ascii_letters = sum("a" <= char.lower() <= "z" for char in text)
    return ascii_letters / max(letters, 1) >= 0.8


def _render_pdf_page(pdf_path: str, page_num: int, dpi: int = 220) -> Image.Image:
    import os

    _ensure_debug_dir()
    output_prefix = f"{NEWS_PIPELINE_DEBUG_DIR}/page-{page_num:03d}"
    output_path = f"{output_prefix}.png"

    command = [
        "pdftoppm",
        "-f",
        str(page_num),
        "-l",
        str(page_num),
        "-r",
        str(dpi),
        "-png",
        "-singlefile",
        pdf_path,
        output_prefix,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Rendered page image not found: {output_path}")

    return Image.open(output_path).convert("RGB")


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.SHARPEN)

    width, height = gray.size
    target_width = max(width, 1800)
    if width < target_width:
        target_height = int(height * (target_width / width))
        gray = gray.resize((target_width, target_height))

    return gray.convert("RGB")


def _split_image_for_ocr(img: Image.Image) -> list[tuple[str, Image.Image]]:
    width, height = img.size
    if width < 1200:
        return [("full", img)]

    overlap = max(30, width // 40)
    midpoint = width // 2
    left = img.crop((0, 0, min(width, midpoint + overlap), height))
    right = img.crop((max(0, midpoint - overlap), 0, width, height))
    return [("left", left), ("right", right)]


def _ocr_chunk_with_tesseract(img: Image.Image, page_num: int, chunk_id: str) -> str:
    import os
    import tempfile

    _ensure_debug_dir()
    with tempfile.NamedTemporaryFile(
        suffix=f"-page-{page_num:03d}-{chunk_id}.png",
        dir=NEWS_PIPELINE_DEBUG_DIR,
        delete=False,
    ) as tmp:
        tmp_path = tmp.name

    try:
        img.save(tmp_path)
        command = [
            "tesseract",
            tmp_path,
            "stdout",
            "-l",
            "eng",
            "--psm",
            "4",
        ]
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return _normalize_whitespace(result.stdout)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _ocr_chunk_with_vision_model(
    client: ZhipuAI | None, img: Image.Image, page_num: int, chunk_id: str
) -> str:
    client = client or _glm_client()
    prompt = """Please transcribe the newspaper text in this image as faithfully as possible.

Rules:
- Keep the original language.
- Output plain text only.
- Preserve reading order from top to bottom. For newspaper columns, read the current crop naturally.
- Do not summarize or explain.
- Ignore decoration if it is unreadable.
"""

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{_image_to_base64(img)}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    raw = response.choices[0].message.content.strip()
    text = _normalize_whitespace(_strip_code_fence(raw))
    info(
        f"[NewsPipeline] OCR page {page_num} chunk {chunk_id}: {len(text)} chars"
    )
    return text


def _ocr_chunk_to_text(
    client: ZhipuAI | None, img: Image.Image, page_num: int, chunk_id: str
) -> str:
    local_text = ""
    try:
        local_text = _ocr_chunk_with_tesseract(img, page_num, chunk_id)
    except Exception as exc:
        warning(
            f"[NewsPipeline] Local OCR failed on page {page_num} chunk {chunk_id}: {exc}"
        )

    if _looks_like_meaningful_text(local_text):
        info(
            f"[NewsPipeline] OCR page {page_num} chunk {chunk_id}: using local tesseract"
        )
        return local_text

    if local_text:
        warning(
            f"[NewsPipeline] OCR page {page_num} chunk {chunk_id}: local OCR was weak, trying vision fallback"
        )

    return _ocr_chunk_with_vision_model(client, img, page_num, chunk_id)


def _ocr_page_to_text(client: ZhipuAI | None, img: Image.Image, page_num: int) -> str:
    processed = _preprocess_for_ocr(img)
    chunks = _split_image_for_ocr(processed)
    parts = []
    for chunk_id, chunk_img in chunks:
        chunk_text = _ocr_chunk_to_text(client, chunk_img, page_num, chunk_id)
        if chunk_text:
            parts.append(chunk_text)

    text = _normalize_whitespace("\n\n".join(parts))
    _write_debug_text(page_num, text, "ocr")
    return text


def _preclean_page_text(text: str) -> str:
    text = text.replace("\r", "\n").replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
    text = re.sub(r"(?<=\b[A-Za-z])\s(?=[A-Za-z]\b)", "", text)
    text = re.sub(r"([A-Za-z])\s{2,}([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines = []
    drop_patterns = (
        r"^THE W ALL STREET JOURNAL.*$",
        r"^A\d+ \| .*THE W ALL STREET JOURNAL\.$",
        r"^LETTERS TO THE EDITOR$",
        r"^Letters intended for publication.*$",
        r"^include your city, state and telephone number\.$",
        r"^editing, and unpublished letters cannot be acknowledged\.$",
        r"^BOOKSHELF \| .*",
        r"^OPINION$",
    )

    for raw_line in text.splitlines():
        line = _normalize_whitespace(raw_line)
        if not line:
            cleaned_lines.append("")
            continue
        if any(re.match(pattern, line) for pattern in drop_patterns):
            continue
        if line.startswith("/uni"):
            continue
        cleaned_lines.append(line)

    joined = "\n".join(cleaned_lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def _heuristic_candidate_blocks(page_num: int, text: str) -> list[dict]:
    blocks = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    start_indexes = [0]

    def is_titleish(line: str) -> bool:
        normalized = re.sub(r"\s+", " ", line)
        words = normalized.split()

        if len(normalized) < 8 or len(normalized) > 90:
            return False
        if normalized.endswith((".", "?", "!", ":", ";")):
            return False
        if len(words) > 10:
            return False
        if any(char.isdigit() for char in normalized):
            return False
        if "." in normalized:
            return False

        titleish_ratio = sum(
            word[:1].isupper() or word.lower() in {"and", "of", "the", "in", "to"}
            for word in words
        ) / max(len(words), 1)
        if titleish_ratio < 0.7:
            return False

        lowercase_non_stopwords = sum(
            word[:1].islower()
            and word.lower() not in {"and", "of", "the", "in", "to", "for", "a", "an"}
            for word in words
        )
        if lowercase_non_stopwords > 1:
            return False

        return True

    def is_probable_title_start(index: int) -> bool:
        line = lines[index]
        if not is_titleish(line):
            return False

        prev_line = lines[index - 1] if index > 0 else ""
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        next_next_line = lines[index + 2] if index + 2 < len(lines) else ""

        return (
            prev_line.startswith("By ")
            or next_line.startswith("By ")
            or (is_titleish(next_line) and (next_next_line.startswith("By ") or is_titleish(next_next_line)))
        )

    for index, line in enumerate(lines):
        if index == 0:
            continue
        if line.startswith("By "):
            anchor = max(index - 3, 0)
            if anchor not in start_indexes:
                start_indexes.append(anchor)
            continue
        if is_probable_title_start(index):
            start_indexes.append(index)

    start_indexes = sorted(set(start_indexes))
    for block_index, start in enumerate(start_indexes):
        end = start_indexes[block_index + 1] if block_index + 1 < len(start_indexes) else len(lines)
        block_lines = lines[start:end]
        block_text = "\n".join(block_lines).strip()
        if len(block_text) < 250:
            continue
        seed_title = ""
        for candidate in block_lines[:5]:
            if 8 <= len(candidate) <= 90 and not candidate.startswith("By "):
                seed_title = candidate
                break
        blocks.append(
            {
                "page_num": page_num,
                "seed_title": seed_title,
                "raw_text": block_text,
            }
        )

    return blocks


def _gemini_clean_articles(page_num: int, text: str) -> list[dict]:
    prompt = f"""You are cleaning raw text extracted from an English newspaper opinion page.

The input may contain:
- multiple articles mixed together
- title lines out of order because of newspaper columns
- broken spacing like "T rump" or "Ik n o w"
- line-wrap artifacts and page furniture
- letters to the editor or captions that should be skipped

Your task:
1. Identify each substantive article/review/column on the page.
2. Skip letters-to-the-editor snippets, contact instructions, captions, and page furniture.
3. Reconstruct each article in natural reading order as well as possible.
4. Fix spacing and broken words, but do not invent missing facts.
5. Preserve the article's language.

Return JSON only as an array:
[
  {{
    "title": "Article title",
    "author": "Author if known, else empty string",
    "section": "Opinion/Bookshelf/Column/Other",
    "content": "Cleaned full article text",
    "summary": "One or two sentence factual summary",
    "category": "Politics/Culture/International/Economics/Other"
  }}
]

Only include entries whose cleaned content is substantial and article-like.

Page number: {page_num}
Raw article block:
{text}
"""
    articles = _gemini_generate_json(prompt)
    for article in articles:
        article["source_page"] = page_num
        article["content"] = _normalize_article_text(article.get("content", ""))
        article["title"] = _normalize_whitespace(article.get("title", ""))
        article["author"] = _normalize_whitespace(article.get("author", ""))
        article["summary"] = _normalize_article_text(article.get("summary", ""))
        article["section"] = _normalize_whitespace(article.get("section", ""))
        article["category"] = _normalize_whitespace(article.get("category", ""))
    return [article for article in articles if len(article.get("content", "")) >= 250]


def _write_article_previews(page_num: int, articles: list[dict]) -> None:
    preview_lines = []
    for index, article in enumerate(articles, start=1):
        preview_lines.append(f"===== ARTICLE {index} =====")
        preview_lines.append(f"Title: {article.get('title', '')}")
        preview_lines.append(f"Author: {article.get('author', '')}")
        preview_lines.append(f"Source: {article.get('source_name', '')}")
        preview_lines.append(f"Publish Date: {article.get('publish_date', '')}")
        preview_lines.append(f"Section: {article.get('section', '')}")
        preview_lines.append(f"Article Type: {article.get('article_type', '')}")
        preview_lines.append(f"Category: {article.get('category', '')}")
        preview_lines.append(f"Summary: {article.get('summary', '')}")
        preview_lines.append("")
        preview_lines.append(article.get("content", ""))
        preview_lines.append("")

    _write_debug_text(page_num, "\n".join(preview_lines).strip(), "articles")


def _dedupe_articles(articles: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for article in articles:
        section = article.get("section", "").strip().lower()
        if "letter" in section:
            continue
        key = (
            article.get("title", "").strip().lower(),
            article.get("author", "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)
    return unique


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", _normalize_article_text(text))
    return [part.strip() for part in parts if part.strip()]


def _keywords_from_title(title: str) -> str:
    stopwords = {
        "the", "a", "an", "and", "of", "to", "in", "for", "with", "on", "at", "by",
        "how", "why", "what", "is", "are", "was", "were",
    }
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", title.lower())
    keywords = []
    for word in words:
        if word in stopwords:
            continue
        if word not in keywords:
            keywords.append(word)
        if len(keywords) == 3:
            break
    return ", ".join(keywords or ["news", "analysis", "policy"])


def _extract_core_number(text: str) -> str:
    patterns = [
        r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",
        r"\b\d+(?:\.\d+)?%\b",
        r"\b\d+(?:\.\d+)?\s?(?:billion|million|thousand|degrees?|deaths?|strikes?|years?)\b",
        r"\b\d+(?:\.\d+)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def _fallback_video_classifier(article: dict) -> dict:
    title = article.get("title", "")
    summary = article.get("summary", "")
    content = article.get("content", "")
    article_type = article.get("article_type", "news")
    combined = f"{title}. {summary} {content[:2000]}".strip()
    lower = combined.lower()

    conflict_tokens = (
        "war", "drug", "cartel", "killed", "killing", "border", "strike", "crisis",
        "fight", "ban", "illegal", "retaliation", "collapse", "trump", "iran",
        "policy", "failure", "cost", "tax", "threat", "violence",
    )
    climate_tokens = ("heat", "temperature", "climate", "storm", "wildfire", "flood")
    contrast_tokens = ("versus", "vs.", "while", "but", "instead", "meanwhile", "one side")

    conflict_score = min(10, sum(token in lower for token in conflict_tokens) * 2)
    surprise_score = 8 if _extract_core_number(combined) else 4
    if "?" in title or "shocking" in lower or "suddenly" in lower:
        surprise_score = min(10, surprise_score + 2)
    visual_score = 7 if any(token in lower for token in ("killed", "strike", "boats", "airport", "fire", "storm", "border", "cartel")) else 4

    if any(token in lower for token in climate_tokens):
        story_mode = "data_shock"
        topic_cluster = "climate_extreme"
    elif any(token in lower for token in contrast_tokens):
        story_mode = "contrast"
        topic_cluster = "high_conflict_policy"
    elif article_type == "opinion_or_analysis":
        story_mode = "reveal"
        topic_cluster = "argument_or_analysis"
    elif conflict_score >= 6:
        story_mode = "conflict"
        topic_cluster = "political_or_social_conflict"
    else:
        story_mode = "stakes"
        topic_cluster = "high_stakes_news"

    debate_angle = {
        "data_shock": "is this a warning sign people are still underestimating",
        "contrast": "which side of this tradeoff makes more sense",
        "reveal": "is the real problem the headline event or the policy behind it",
        "conflict": "does this prove the current approach is failing",
        "stakes": "is this the start of a bigger shift",
    }.get(story_mode, "what happens next")

    tone = {
        "data_shock": "urgent",
        "contrast": "confrontational",
        "reveal": "provocative",
        "conflict": "tense",
        "stakes": "serious",
    }.get(story_mode, "serious")

    return {
        "is_video_worthy": conflict_score >= 4 or visual_score >= 6 or surprise_score >= 7,
        "controversy_score": conflict_score,
        "surprise_score": surprise_score,
        "visual_score": visual_score,
        "hook_type": story_mode,
        "story_mode": story_mode,
        "core_number": _extract_core_number(combined),
        "location": "",
        "key_people": [],
        "tone": tone,
        "topic_cluster": topic_cluster,
        "debate_angle": debate_angle,
        "why_now": _sentence_split(summary or content)[:1][0] if (summary or content) else title,
    }


def _ensure_video_classifier(article: dict) -> dict:
    classifier = article.get("video_classifier")
    if classifier:
        return classifier
    classifier = _classify_newspaper_for_video(article)
    article["video_classifier"] = classifier
    return classifier


def _video_worthiness_score(article: dict, trending_keywords: list[str]) -> int:
    classifier = _ensure_video_classifier(article)
    score = (
        int(classifier.get("controversy_score", 0)) * 3
        + int(classifier.get("surprise_score", 0)) * 2
        + int(classifier.get("visual_score", 0)) * 2
    )
    if classifier.get("is_video_worthy"):
        score += 10
    if article.get("article_type") == "opinion_or_analysis":
        score += 4
    if classifier.get("core_number"):
        score += 4

    haystack = " ".join(
        [
            article.get("title", ""),
            article.get("summary", ""),
            article.get("category", ""),
            classifier.get("topic_cluster", ""),
            classifier.get("debate_angle", ""),
        ]
    ).lower()
    for keyword in trending_keywords:
        keyword_lower = keyword.strip().lower()
        if keyword_lower and keyword_lower in haystack:
            score += 3
    return score


def _classify_newspaper_for_video(article: dict) -> dict:
    prompt = f"""You are labeling a newspaper article for short-form video scripting.

Return JSON only in this exact shape:
{{
  "is_video_worthy": true,
  "controversy_score": 0,
  "surprise_score": 0,
  "visual_score": 0,
  "hook_type": "number|conflict|reveal|contrast|stakes",
  "story_mode": "data_shock|conflict|reveal|contrast|stakes",
  "core_number": "",
  "location": "",
  "key_people": ["person1", "person2"],
  "tone": "tense|angry|urgent|provocative|serious|hopeful",
  "topic_cluster": "",
  "debate_angle": "",
  "why_now": ""
}}

Rules:
- Score each field from 0 to 10 where applicable.
- Favor labels that create strong short-video hooks only when the article truly supports them.
- "core_number" should be the strongest number, percent, money figure, death toll, temperature, or count if one exists.
- "debate_angle" should be phrased as a clear disagreement, tradeoff, or unresolved question.
- "why_now" should be a single sentence on why viewers should care now.
- Do not mention the publication.

Article metadata:
- Title: {article.get("title", "")}
- Article type: {article.get("article_type", "")}
- Section: {article.get("section", "")}
- Summary: {article.get("summary", "")}
- Content: {article.get("content", "")[:8000]}
"""

    try:
        return _openai_generate_json(prompt)
    except Exception as exc:
        warning(
            f"[NewsPipeline] MiniMax video classification failed for '{article.get('title', '')}', falling back to Gemini: {exc}"
        )
        try:
            return _gemini_generate_json(prompt)
        except Exception as gemini_exc:
            warning(
                f"[NewsPipeline] Gemini video classification failed for '{article.get('title', '')}', using local fallback: {gemini_exc}"
            )
            return _fallback_video_classifier(article)


def _fallback_newspaper_script(article: dict, language: str = "English") -> dict:
    summary_sentences = _sentence_split(article.get("summary", ""))
    content_sentences = _sentence_split(article.get("content", ""))
    classifier = article.get("video_classifier") or _fallback_video_classifier(article)
    lead = summary_sentences[0] if summary_sentences else (
        content_sentences[0] if content_sentences else article.get("title", "")
    )
    support = summary_sentences[1] if len(summary_sentences) > 1 else (
        content_sentences[1] if len(content_sentences) > 1 else ""
    )
    detail = content_sentences[2] if len(content_sentences) > 2 else ""

    story_mode = classifier.get("story_mode", "stakes")
    core_number = classifier.get("core_number", "")
    debate_angle = classifier.get("debate_angle", "what happens next")

    if story_mode == "data_shock" and core_number:
        intro = f"{core_number}. That is the number forcing this story into the spotlight right now. "
        bridge = "The bigger issue is what that number says about where things are heading. "
    elif story_mode == "contrast":
        intro = "One side says this is necessary. The other says it is a massive mistake. "
        bridge = "And that clash is exactly why this story is blowing up. "
    elif story_mode == "reveal":
        intro = "The headline is not even the most interesting part of this story. "
        bridge = "What really matters is the logic underneath it. "
    elif story_mode == "conflict":
        intro = "This story looks less like a solution and more like a fight that is escalating. "
        bridge = "And the deeper you look, the harder it is to argue the current approach is working. "
    else:
        intro = "This story matters more than the headline makes it sound. "
        bridge = "Because once you look at the stakes, it becomes a lot harder to ignore. "

    closing = f"So the real debate is {debate_angle}."
    pieces = [intro + lead]
    if support:
        pieces.append(bridge + support)
    if detail:
        pieces.append(detail)
    pieces.append(closing)

    script = " ".join(piece.strip() for piece in pieces if piece.strip())
    return {
        "script": script,
        "keywords": _keywords_from_title(article.get("title", "")),
        "title": article.get("title", "News Brief"),
    }


def _story_mode_template(story_mode: str) -> str:
    templates = {
        "data_shock": """Template:
- Line 1: lead with the strongest number and why it is shocking.
- Middle: explain why that number changes the way viewers should see the story.
- Ending: turn the number into a debate about what should happen next.""",
        "contrast": """Template:
- Line 1: frame two sides, two choices, or two outcomes in direct opposition.
- Middle: explain why the clash matters and who pays the price.
- Ending: force the audience to weigh the tradeoff.""",
        "reveal": """Template:
- Line 1: say the obvious headline is not the real story.
- Middle: reveal the hidden logic, incentive, or policy failure underneath.
- Ending: ask whether the audience has been looking at the wrong problem.""",
        "conflict": """Template:
- Line 1: open on escalation, backlash, or a breakdown.
- Middle: explain what is colliding and why the current strategy looks unstable.
- Ending: frame the issue as a fight over what happens next.""",
        "stakes": """Template:
- Line 1: open with the consequence that matters most.
- Middle: explain why this is bigger than a routine update.
- Ending: leave viewers with a high-stakes unresolved question.""",
    }
    return templates.get(
        story_mode,
        """Template:
- Line 1: open with the most interesting angle.
- Middle: explain the evidence and consequences.
- Ending: leave viewers with a strong unresolved question.""",
    )


def _rewrite_newspaper_as_script(article: dict, language: str = "English") -> dict:
    classifier = _ensure_video_classifier(article)
    article_type = article.get("article_type", "news")
    if article_type == "opinion_or_analysis":
        framing = (
            "Frame the piece as an explainer of the article's argument. Make clear that this is the publication's or writer's view, not objective fact."
        )
    elif article_type == "review_or_feature":
        framing = (
            "Frame the piece as a concise cultural or feature explainer. Explain what is being reviewed or discussed, what the main judgment is, and why it matters."
        )
    else:
        framing = (
            "Frame the piece as a news briefing with light analysis and context."
        )

    story_mode = classifier.get("story_mode", "stakes")
    story_mode_instructions = {
        "data_shock": "Open with the strongest number immediately. Treat the first line like a jolt, then explain why that number is alarming or surprising.",
        "contrast": "Open with a sharp contrast between two sides, two choices, or two outcomes. Make the conflict feel immediate.",
        "reveal": "Open by challenging the obvious headline and pivoting to the deeper issue people are missing.",
        "conflict": "Open with escalation, failure, or confrontation. Make the audience feel like something is colliding.",
        "stakes": "Open with the highest-stakes consequence and why this is bigger than a routine update.",
    }.get(story_mode, "Open with the most interesting angle immediately.")

    prompt = f"""You are writing an original short-form video script based on a newspaper article.

Role:
- Speak like a sharp short-video narrator with light analytical commentary.
- Sound punchy, specific, and conversational.

Copyright and style rules:
- Do not imitate or closely track the article's sentence order.
- Do not reproduce distinctive phrases except for a very short quote when absolutely necessary.
- Do not read or paraphrase the article line by line.
- Use the article only as factual source material to create a fresh spoken explanation.

Structure:
1. Open with a hook in the first sentence. No throat-clearing.
2. Explain the key facts, reasoning, or evidence.
3. End with a real debate, unresolved tradeoff, or comment-driving question.

Special framing:
- {framing}
- Story mode: {story_mode}
- Hook instruction: {story_mode_instructions}
- Debate angle to land on: {classifier.get("debate_angle", "")}
- If there is a strong number, use it early: {classifier.get("core_number", "")}
- Do not mention the publication, newspaper, or where the article came from.
- Avoid stiff transitions like "here's what happened" or "here's the argument from".
- Make the script more provocative and discussion-worthy, but do not invent facts or overstate what the article supports.
- Follow this story-mode template closely:
{_story_mode_template(story_mode)}

Output requirements:
- Language: {language}
- Length: about 150-220 words
- No markdown
- No bullet points
- No mention of being an AI
- Return JSON only

JSON format:
{{"script": "...", "keywords": "keyword1, keyword2, keyword3", "title": "Short video title"}}

Source:
- Publish date: {article.get("publish_date", "")}
- Section: {article.get("section", "")}
- Article type: {article_type}
- Original title: {article.get("title", "")}
- Author: {article.get("author", "")}
- Video classifier: {json.dumps(classifier, ensure_ascii=False)}
- Summary: {article.get("summary", "")}
- Article content: {article.get("content", "")}
"""

    try:
        result = _gemini_generate_json(prompt)
    except Exception as exc:
        warning(
            f"[NewsPipeline] Gemini rewrite failed for '{article.get('title', '')}', falling back to MiniMax: {exc}"
        )
        try:
            result = _openai_generate_json(prompt)
        except Exception as openai_exc:
            warning(
                f"[NewsPipeline] MiniMax rewrite failed for '{article.get('title', '')}', using local fallback: {openai_exc}"
            )
            result = _fallback_newspaper_script(article, language=language)

    result["script"] = _normalize_article_text(result.get("script", ""))
    result["title"] = _normalize_whitespace(
        result.get("title", article.get("title", ""))
    )
    result["keywords"] = _normalize_whitespace(result.get("keywords", "news"))
    return result


def _extract_articles_from_page_text(page_num: int, text: str) -> list[dict]:
    precleaned = _preclean_page_text(text)
    _write_debug_text(page_num, precleaned, "preclean")
    heuristic_blocks = _heuristic_candidate_blocks(page_num, precleaned)
    _write_debug_json(f"page-{page_num:03d}-blocks.json", heuristic_blocks)

    if _looks_like_english_text(precleaned):
        try:
            articles = []
            blocks_for_ai = heuristic_blocks or [
                {"page_num": page_num, "seed_title": "", "raw_text": precleaned}
            ]
            for block in blocks_for_ai:
                if len(block["raw_text"]) < 250:
                    continue
                block_articles = _gemini_clean_articles(page_num, block["raw_text"][:9000])
                articles.extend(block_articles)
            articles = _dedupe_articles(articles)
            if articles:
                _write_article_previews(page_num, articles)
                info(
                    f"[NewsPipeline] Cleaned page {page_num} into {len(articles)} articles with Gemini"
                )
                return articles
        except Exception as exc:
            warning(f"[NewsPipeline] Gemini cleaning failed on page {page_num}: {exc}")

    prompt = f"""Please extract the news articles from this newspaper page text.

Return a JSON array in this format:
[{{"title": "Article title", "summary": "Short summary within 160 characters", "category": "Finance/Tech/Politics/International/Other", "content": "Cleaned article text"}}]

Requirements:
- Ignore page furniture such as issue headers, stock tables, ads, indexes, and copyright notices.
- Merge wrapped lines back into normal article text mentally before extracting.
- If the page contains multiple articles, return all meaningful articles you can identify.
- If the page contains no article-like content, return [].
- Return JSON only.

Page number: {page_num}
Page text:
{precleaned}
"""
    try:
        articles = _openai_generate_json(prompt)
        info(
            f"[NewsPipeline] Structured page {page_num} into {len(articles)} articles with MiniMax"
        )
        for article in articles:
            article["source_page"] = page_num
            article["content"] = _normalize_article_text(article.get("content", ""))
        _write_article_previews(page_num, articles)
        return articles
    except Exception as exc:
        warning(
            f"[NewsPipeline] Page {page_num}: MiniMax structured extraction failed: {exc}"
        )
        return []


# ── PDF → images ──────────────────────────────────────────────────────────────


def find_opinion_pages(pdf_path: str) -> list[int]:
    reader = PdfReader(pdf_path)
    opinion_pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        lower = text.lower()
        if any(keyword in lower for keyword in ("opinion", "editorial", "analysis")):
            opinion_pages.append(i)
    info(f"[NewsPipeline] Found 'opinion' on pages: {opinion_pages}")
    return opinion_pages


def extract_page_texts(pdf_path: str, target_pages: list[int] | None = None) -> list[dict]:
    reader = PdfReader(pdf_path)
    page_texts = []

    for page_num, page in enumerate(reader.pages, start=1):
        if target_pages is not None and page_num not in target_pages:
            continue

        direct_text = _normalize_whitespace(page.extract_text() or "")
        if _looks_like_meaningful_text(direct_text):
            info(
                f"[NewsPipeline] Page {page_num}: using embedded text ({len(direct_text)} chars)"
            )
            _write_debug_text(page_num, direct_text, "pdftext")
            page_texts.append(
                {"page_num": page_num, "text": direct_text, "source": "embedded_text"}
            )
            continue

        warning(
            f"[NewsPipeline] Page {page_num}: embedded text is missing or weak, falling back to OCR"
        )
        rendered_page = _render_pdf_page(pdf_path, page_num)
        ocr_text = _ocr_page_to_text(None, rendered_page, page_num)
        if not ocr_text:
            warning(f"[NewsPipeline] Page {page_num}: OCR produced no usable text")
            continue

        page_texts.append({"page_num": page_num, "text": ocr_text, "source": "ocr"})

    success(f"[NewsPipeline] Extracted usable text from {len(page_texts)} pages")
    return page_texts


# ── GLM-4V OCR ────────────────────────────────────────────────────────────────


def ocr_page(client: ZhipuAI, img: Image.Image, page_num: int) -> list[dict]:
    text = _ocr_page_to_text(client, img, page_num)
    if not text:
        return []
    return _extract_articles_from_page_text(page_num, text)


def ocr_pdf(pdf_path: str) -> list[dict]:
    opinion_pages = find_opinion_pages(pdf_path)
    if not opinion_pages:
        warning("[NewsPipeline] No 'opinion' pages found, falling back to full scan")
        opinion_pages = None
    page_texts = extract_page_texts(pdf_path, target_pages=opinion_pages)
    sample_text = "\n".join(page["text"][:2000] for page in page_texts[:2])
    source_name = _infer_source_name(pdf_path, sample_text)
    publish_date = _infer_publish_date(sample_text)
    all_articles = []
    for page in page_texts:
        extracted = _extract_articles_from_page_text(page["page_num"], page["text"])
        for article in extracted:
            all_articles.append(
                _annotate_newspaper_article(article, source_name, publish_date)
            )
    success(f"[NewsPipeline] OCR complete — {len(all_articles)} articles total")
    return all_articles


# ── Google Trends ─────────────────────────────────────────────────────────────


def get_trending_keywords(geo: str = "NZ", top_n: int = 20) -> list[str]:
    try:
        from pytrends.request import TrendReq

        geo_map = {
            "NZ": "new_zealand",
            "US": "united_states",
            "AU": "australia",
            "CN": "china",
        }
        pytrends = TrendReq(hl="zh-CN", tz=720)
        df = pytrends.trending_searches(pn=geo_map.get(geo, "new_zealand"))
        keywords = df[0].tolist()[:top_n]
        info(f"[NewsPipeline] Trending keywords: {keywords[:5]}...")
        return keywords
    except Exception as e:
        warning(f"[NewsPipeline] Could not fetch Google Trends: {e}")
        return []


# ── Article selection ─────────────────────────────────────────────────────────


def select_top_articles(
    articles: list[dict], trending_keywords: list[str], top_n: int = 3
) -> list[dict]:
    if not articles:
        return []
    if all(article.get("source_type") == "newspaper" for article in articles):
        ranked_articles = []
        for article in articles:
            classifier = _ensure_video_classifier(article)
            article["video_classifier"] = classifier
            article["video_score"] = _video_worthiness_score(article, trending_keywords)
            ranked_articles.append(article)

        ranked_articles.sort(
            key=lambda item: (
                item.get("video_score", 0),
                int(item.get("video_classifier", {}).get("controversy_score", 0)),
                int(item.get("video_classifier", {}).get("surprise_score", 0)),
            ),
            reverse=True,
        )

        prompt = f"""You are choosing newspaper articles for short-form video coverage.

Pick up to {top_n} articles that are strongest for a concise, engaging explainer video.
Prefer:
- timely relevance
- clear stakes or consequences
- strong argument or narrative hook
- broad audience interest
- high controversy, surprise, or visual potential when the article supports it

Trending keywords:
{", ".join(trending_keywords) if trending_keywords else "None"}

Articles:
{json.dumps(ranked_articles[: min(len(ranked_articles), top_n * 4)], ensure_ascii=False, indent=2)}

Return JSON only as an array of the original selected article objects.
"""
        try:
            selected = _gemini_generate_json(prompt)
            success(f"[NewsPipeline] Selected {len(selected)} articles with Gemini")
            return selected
        except Exception as exc:
            warning(f"[NewsPipeline] Gemini article selection failed: {exc}")
            return ranked_articles[:top_n]

    prompt = f"""以下是从报纸中提取的新闻文章（JSON格式），以及当前热门搜索关键词。

热门关键词：{", ".join(trending_keywords) if trending_keywords else "无"}

新闻文章：
{json.dumps(articles, ensure_ascii=False, indent=2)}

请挑选最多{top_n}篇最值得制作成短视频的文章，选择标准：
1. 与热门关键词相关度高
2. 内容有独到见解或吸引眼球
3. 适合普通观众理解

以JSON数组返回选中文章的原始对象，只返回JSON。"""
    try:
        selected = _openai_generate_json(prompt)
        success(f"[NewsPipeline] Selected {len(selected)} articles with MiniMax")
        return selected
    except Exception as exc:
        warning(f"[NewsPipeline] MiniMax article selection failed: {exc}")
        warning("[NewsPipeline] Selection parse failed, using first article")
        return articles[:1]


# ── Script rewrite ────────────────────────────────────────────────────────────


def rewrite_as_script(article: dict, language: str = "Chinese") -> dict:
    if article.get("source_type") == "newspaper":
        result = _rewrite_newspaper_as_script(article, language=language)
        success(f"[NewsPipeline] Script ready: {result.get('title', article['title'])}")
        return result

    source_text = article.get("content") or article.get("summary", "")

    prompt = f"""请将以下新闻改写为适合YouTube短视频的口语化旁白脚本。

标题：{article["title"]}
内容：{source_text}

要求：
- 时长约60~90秒（约150~200字）
- 口语化、简洁、有吸引力
- 第一句话要抓住观众注意力
- 使用语言：{language}
- 同时输出3个英文关键词供搜索图片素材（逗号分隔）

以JSON格式返回：
{{"script": "旁白脚本", "keywords": "keyword1, keyword2, keyword3", "title": "视频标题（20字以内）"}}
只返回JSON。"""
    result = _openai_generate_json(prompt)
    success(f"[NewsPipeline] Script ready: {result.get('title', article['title'])}")
    return result


# ── Main runner ───────────────────────────────────────────────────────────────


def run_news_pipeline(pdf_path: str, youtube_instance, top_n: int = 2) -> None:
    """
    Full pipeline: PDF → OCR → trends → select → rewrite → generate video → upload.

    Args:
        pdf_path (str): Path to the scanned PDF file.
        youtube_instance: An initialised YouTube instance from classes/YouTube.py.
        top_n (int): Number of articles to produce videos for.
    """
    from classes.Tts import TTS

    geo = get_google_trends_geo()

    # 1. OCR
    articles = ocr_pdf(pdf_path)
    if not articles:
        error("[NewsPipeline] No articles extracted. Aborting.")
        return

    # 2. Trends
    keywords = get_trending_keywords(geo=geo)

    # 3. Select
    selected = select_top_articles(articles, keywords, top_n=top_n)

    # 4. For each article: rewrite → inject into YouTube → generate → upload
    tts = TTS()
    for article in selected:
        script_data = rewrite_as_script(article, language=youtube_instance.language)

        # Inject our content directly into the YouTube instance
        youtube_instance.subject = script_data["title"]
        youtube_instance.script = script_data["script"]
        youtube_instance.video_clips = None  # reset from any previous run
        youtube_instance.pending_keywords = [
            keyword.strip()
            for keyword in script_data.get("keywords", "news").split(",")
            if keyword.strip()
        ]

        def regenerate_script_variant() -> dict:
            refreshed = rewrite_as_script(article, language=youtube_instance.language)
            youtube_instance.subject = refreshed["title"]
            youtube_instance.script = refreshed["script"]
            youtube_instance.pending_keywords = [
                keyword.strip()
                for keyword in refreshed.get("keywords", "news").split(",")
                if keyword.strip()
            ]
            return refreshed

        originality_report = youtube_instance.ensure_script_originality(
            regenerate_callback=regenerate_script_variant,
            max_attempts=3,
        )
        if originality_report.get("too_similar"):
            warning(
                f"[NewsPipeline] Script for '{article.get('title', '')}' stayed too similar "
                f"to '{originality_report.get('matched_title', '')}'. Generating preview only; upload guard will remain active."
            )

        # Generate metadata and TTS (needed by both paths)
        youtube_instance.generate_metadata()
        youtube_instance.generate_script_to_speech(tts)
        visual_source = youtube_instance.prepare_visual_assets(
            keywords=youtube_instance.pending_keywords
        )
        info(f"[NewsPipeline] Visual source resolved to: {visual_source}")

        video_path = youtube_instance.combine()

        info(f"[NewsPipeline] Video ready: {video_path}")

        upload = input("Upload this video to YouTube? (yes/no): ").strip().lower()
        if upload == "yes":
            youtube_instance.upload_video()
