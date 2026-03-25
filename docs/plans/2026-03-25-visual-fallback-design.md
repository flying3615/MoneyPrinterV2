# Visual Fallback Design

**Goal:** Unify visual asset generation behind a shared three-layer fallback: Gemini AI images, then Pexels video clips, then local placeholder images.

**Design:**
- Move visual-source selection into `YouTube` so both the news pipeline and the standard video generator share the same behavior.
- Try the configured preferred source first, but never fail the pipeline if that provider is unavailable.
- Keep `combine()` unchanged as the final assembler: it should receive either `video_clips` or `images`.

**Fallback Order:**
1. Gemini AI images
2. Pexels video clips
3. Local placeholder images

**Notes:**
- Pexels requires TTS duration, so TTS must be generated before fetching clips.
- AI image generation should not immediately fall back to placeholders; otherwise Pexels never gets a chance.
- Placeholder images remain the final no-network safety net.
