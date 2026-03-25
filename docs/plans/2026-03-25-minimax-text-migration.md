# MiniMax Text Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all Zhipu/GLM text-generation calls with MiniMax via an OpenAI-compatible client while keeping the existing ZAI vision OCR fallback intact.

**Architecture:** Keep `ollama` as one provider option, add an OpenAI-compatible provider for MiniMax, and route all shared text generation through that layer. In `NewsPipeline`, keep the ZAI vision client only for OCR fallback and move all text structuring, selection, and rewrite steps to MiniMax.

**Tech Stack:** Python 3.12, `openai`, `ollama`, Gemini HTTP API, ZhipuAI vision API

---

### Task 1: Add MiniMax/OpenAI-compatible configuration

**Files:**
- Modify: `src/config.py`
- Modify: `config.example.json`
- Modify: `docs/Configuration.md`

**Step 1:** Add getters for `openai_base_url`, `openai_api_key`, and `openai_model`, with environment-variable fallbacks.

**Step 2:** Keep `zhipu_api_key` only for optional OCR vision fallback and stop exposing `zhipu_model` as the active text-model setting.

### Task 2: Replace shared text-generation provider

**Files:**
- Modify: `src/llm_provider.py`
- Modify: `src/main.py`
- Modify: `src/cron.py`

**Step 1:** Add an OpenAI-compatible client using the `openai` package.

**Step 2:** Route non-Ollama text generation through MiniMax/OpenAI-compatible chat completions.

**Step 3:** Update startup/model-selection code so Ollama still supports interactive local model choice, while MiniMax uses the configured model directly.

### Task 3: Migrate NewsPipeline text calls

**Files:**
- Modify: `src/classes/NewsPipeline.py`

**Step 1:** Keep the Zhipu vision client only for OCR image fallback.

**Step 2:** Replace text JSON generation, article selection, and script rewriting with MiniMax/OpenAI-compatible calls.

**Step 3:** Preserve the Gemini-first strategy and local last-resort fallback for newspaper script generation.

### Task 4: Verification and dependency updates

**Files:**
- Modify: `requirements.txt`

**Step 1:** Add the `openai` dependency.

**Step 2:** Run syntax verification on the touched Python modules.
