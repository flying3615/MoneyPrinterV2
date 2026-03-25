# News OCR Minimal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make newspaper ingestion robust enough to extract readable page text from scanned PDFs before article structuring.

**Architecture:** Prefer embedded PDF text when it is usable. Fall back to page rendering with `pdftoppm`, then run chunked vision OCR and pass the resulting text into a second text-only article extraction step.

**Tech Stack:** Python, `pypdf`, Pillow, `pdftoppm`, ZhipuAI GLM text/vision APIs

---

### Task 1: Stabilize page ingestion

**Files:**
- Modify: `src/classes/NewsPipeline.py`

Implement a text-first extraction path that:
- fixes page number indexing in opinion-page detection
- checks whether `pypdf.extract_text()` is good enough for a page
- renders the page with `pdftoppm` when embedded text is missing or unusable

### Task 2: Add OCR chunking

**Files:**
- Modify: `src/classes/NewsPipeline.py`

Implement a minimal OCR fallback that:
- preprocesses the rendered page image
- splits wide newspaper pages into left/right column chunks
- OCRs each chunk separately with the vision model
- concatenates chunk text in reading order

### Task 3: Keep the pipeline compatible

**Files:**
- Modify: `src/classes/NewsPipeline.py`

Keep `run_news_pipeline()` working by:
- extracting structured article candidates from page text in a second step
- preserving the existing `title` / `summary` / `category` shape expected by the rest of the flow
- writing intermediate page text files under `.mp/news_pipeline/` for inspection

### Task 4: Validate the concept locally

**Files:**
- None

Run the updated pipeline against `华尔街日报-2-26.pdf` and confirm:
- page text extraction prefers embedded text when present
- OCR fallback path is available for scanned pages
- article extraction still returns usable candidates
