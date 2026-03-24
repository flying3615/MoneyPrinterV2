"""
NewsPipeline
------------
Extracts news articles from a scanned PDF using GLM-4V OCR,
matches them against Google Trends keywords, rewrites the top
articles as short-video scripts, then feeds each script into
the existing YouTube pipeline for video generation and upload.
"""

import io
import json
import base64

from PIL import Image
from pypdf import PdfReader
from zhipuai import ZhipuAI

from config import get_zhipu_api_key, get_zhipu_model, get_google_trends_geo, get_video_source, get_pexels_api_key
from status import info, warning, success, error


def _glm_client() -> ZhipuAI:
    api_key = get_zhipu_api_key()
    if not api_key:
        raise RuntimeError("zhipu_api_key is not set in config.json")
    return ZhipuAI(api_key=api_key)


def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ── PDF → images ──────────────────────────────────────────────────────────────

def extract_page_images(pdf_path: str) -> list[Image.Image]:
    reader = PdfReader(pdf_path)
    images = []
    for page in reader.pages:
        for img_obj in page.images:
            images.append(Image.open(io.BytesIO(img_obj.data)).convert("RGB"))
            break  # one image per page
    info(f"[NewsPipeline] Extracted {len(images)} page images from PDF")
    return images


# ── GLM-4V OCR ────────────────────────────────────────────────────────────────

def ocr_page(client: ZhipuAI, img: Image.Image, page_num: int) -> list[dict]:
    prompt = """这是一份中文报纸的扫描页面。
请提取页面中所有新闻文章，以JSON数组返回，格式如下：
[{"title": "文章标题", "summary": "文章内容摘要（100字以内）", "category": "财经/科技/政治/国际/其他"}]
如果页面没有文章内容（如广告、图片页），返回空数组 []。
只返回JSON，不要其他说明。"""

    response = client.chat.completions.create(
        model="glm-4v-flash",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_image_to_base64(img)}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        articles = json.loads(raw.strip())
        info(f"[NewsPipeline] Page {page_num}: {len(articles)} articles found")
        return articles
    except json.JSONDecodeError:
        warning(f"[NewsPipeline] Page {page_num}: could not parse JSON, skipping")
        return []


def ocr_pdf(pdf_path: str) -> list[dict]:
    client = _glm_client()
    images = extract_page_images(pdf_path)
    all_articles = []
    for i, img in enumerate(images):
        all_articles.extend(ocr_page(client, img, i + 1))
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

def select_top_articles(articles: list[dict], trending_keywords: list[str], top_n: int = 3) -> list[dict]:
    if not articles:
        return []
    client = _glm_client()
    model = get_zhipu_model() or "glm-4-flash"

    prompt = f"""以下是从报纸中提取的新闻文章（JSON格式），以及当前热门搜索关键词。

热门关键词：{", ".join(trending_keywords) if trending_keywords else "无"}

新闻文章：
{json.dumps(articles, ensure_ascii=False, indent=2)}

请挑选最多{top_n}篇最值得制作成短视频的文章，选择标准：
1. 与热门关键词相关度高
2. 内容有独到见解或吸引眼球
3. 适合普通观众理解

以JSON数组返回选中文章的原始对象，只返回JSON。"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        selected = json.loads(raw.strip())
        success(f"[NewsPipeline] Selected {len(selected)} articles")
        return selected
    except json.JSONDecodeError:
        warning("[NewsPipeline] Selection parse failed, using first article")
        return articles[:1]


# ── Script rewrite ────────────────────────────────────────────────────────────

def rewrite_as_script(article: dict, language: str = "Chinese") -> dict:
    client = _glm_client()
    model = get_zhipu_model() or "glm-4-flash"

    prompt = f"""请将以下新闻改写为适合YouTube短视频的口语化旁白脚本。

标题：{article['title']}
内容：{article['summary']}

要求：
- 时长约60~90秒（约150~200字）
- 口语化、简洁、有吸引力
- 第一句话要抓住观众注意力
- 使用语言：{language}
- 同时输出3个英文关键词供搜索图片素材（逗号分隔）

以JSON格式返回：
{{"script": "旁白脚本", "keywords": "keyword1, keyword2, keyword3", "title": "视频标题（20字以内）"}}
只返回JSON。"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw.strip())
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

    video_source = get_video_source()
    pexels_api_key = get_pexels_api_key()

    if video_source == "pexels" and not pexels_api_key:
        warning("[NewsPipeline] video_source is 'pexels' but pexels_api_key is not set — falling back to ai_images")
        video_source = "ai_images"

    info(f"[NewsPipeline] Video source: {video_source}")

    # 4. For each article: rewrite → inject into YouTube → generate → upload
    tts = TTS()
    for article in selected:
        script_data = rewrite_as_script(article, language=youtube_instance.language)

        # Inject our content directly into the YouTube instance
        youtube_instance.subject = script_data["title"]
        youtube_instance.script = script_data["script"]
        youtube_instance.video_clips = None  # reset from any previous run

        # Generate metadata and TTS (needed by both paths)
        youtube_instance.generate_metadata()
        youtube_instance.generate_script_to_speech(tts)

        if video_source == "pexels":
            # Fetch Pexels video clips using the script's keywords
            from pexels import fetch_clips_for_keywords
            from moviepy.editor import AudioFileClip as _AC
            tts_duration = _AC(youtube_instance.tts_path).duration
            keywords = [k.strip() for k in script_data.get("keywords", "news").split(",")]
            youtube_instance.video_clips = fetch_clips_for_keywords(
                keywords=keywords,
                api_key=pexels_api_key,
                tts_duration=tts_duration,
            )
            if not youtube_instance.video_clips:
                warning("[NewsPipeline] No Pexels clips downloaded, falling back to AI images")
                youtube_instance.video_clips = None
                youtube_instance.generate_prompts()
                for prompt in youtube_instance.image_prompts:
                    youtube_instance.generate_image(prompt)
        else:
            # AI image path
            youtube_instance.generate_prompts()
            for prompt in youtube_instance.image_prompts:
                youtube_instance.generate_image(prompt)

        video_path = youtube_instance.combine()

        info(f"[NewsPipeline] Video ready: {video_path}")

        upload = input("Upload this video to YouTube? (yes/no): ").strip().lower()
        if upload == "yes":
            youtube_instance.upload_video()
