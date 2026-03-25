# Opinion Cleaning Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn English opinion pages into article-level cleaned text that can be reviewed before video generation.

**Architecture:** Keep page ingestion text-first. Add code-based block splitting and normalization to isolate likely article chunks, then run an optional AI cleaning pass that repairs reading order, broken spacing, and article boundaries without summarizing.

**Tech Stack:** Python, `pypdf`, Pillow, `tesseract`, optional text LLM cleaning

---

### Task 1: Add article-block preprocessing

**Files:**
- Modify: `src/classes/NewsPipeline.py`

Implement code-based helpers that:
- remove page furniture and common opinion-page boilerplate
- normalize broken spacing and hyphenated line wraps
- split page text into likely article blocks using headings and bylines

### Task 2: Add AI-assisted article cleaning

**Files:**
- Modify: `src/classes/NewsPipeline.py`

Implement a cleaner that:
- takes a candidate article block
- asks a text model to preserve meaning while fixing reading order and spacing
- returns title + cleaned body text + short summary
- remains optional so the pipeline still works if no model is available

### Task 3: Add preview output

**Files:**
- Modify: `src/classes/NewsPipeline.py`

Write cleaned article previews into `.mp/news_pipeline/` so pages and article candidates can be reviewed before video generation.

### Task 4: Validate on the local WSJ PDF

**Files:**
- None

Run the opinion-page extraction against `华尔街日报-2-26.pdf` and confirm:
- mixed articles on A13-A15 are separated more cleanly
- broken spacing is reduced
- cleaned article previews are readable enough to review before script generation
