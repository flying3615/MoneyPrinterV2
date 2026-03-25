# Configuration

All your configurations will be in a file in the root directory, called `config.json`, which is a copy of `config.example.json`. You can change the values in `config.json` to your liking.

## Values

- `verbose`: `boolean` - If `true`, the application will print out more information.
- `firefox_profile`: `string` - The path to your Firefox profile. This is used to use your Social Media Accounts without having to log in every time you run the application.
- `headless`: `boolean` - If `true`, the application will run in headless mode. This means that the browser will not be visible.
- `ollama_base_url`: `string` - Base URL of your local Ollama server (default: `http://127.0.0.1:11434`).
- `ollama_model`: `string` - Ollama model to use for text generation (e.g. `llama3.2:3b`). If empty, the app queries Ollama at startup and lets you pick from the available models interactively.
- `llm_provider`: `string` - Text-generation provider. Options: `ollama`, `openai_compatible`.
- `openai_base_url`: `string` - Base URL for an OpenAI-compatible provider such as MiniMax (default: `https://api.minimaxi.com/v1`). If empty, MPV2 falls back to environment variable `OPENAI_BASE_URL`, then to the MiniMax default URL.
- `openai_api_key`: `string` - API key for the OpenAI-compatible provider. If empty, MPV2 falls back to environment variable `OPENAI_API_KEY`.
- `openai_model`: `string` - Model name for the OpenAI-compatible provider (default: `MiniMax-M2.7`). If empty, MPV2 falls back to environment variable `OPENAI_MODEL`.
- `twitter_language`: `string` - The language that will be used to generate & post tweets.
- `nanobanana2_api_base_url`: `string` - Nano Banana 2 API base URL (default: `https://generativelanguage.googleapis.com/v1beta`).
- `nanobanana2_api_key`: `string` - API key for Nano Banana 2 (Gemini image API). If empty, MPV2 falls back to environment variable `GEMINI_API_KEY`.
- `nanobanana2_model`: `string` - Nano Banana 2 model name (default: `gemini-3.1-flash-image-preview`).
- `nanobanana2_aspect_ratio`: `string` - Aspect ratio for generated images (default: `9:16`).
- `threads`: `number` - The amount of threads that will be used to execute operations, e.g. writing to a file using MoviePy.
- `is_for_kids`: `boolean` - If `true`, the application will upload the video to YouTube Shorts as a video for kids.
- `google_maps_scraper`: `string` - The URL to the Google Maps scraper. This will be used to scrape Google Maps for local businesses. It is recommended to use the default value.
- `zip_url`: `string` - The URL to the ZIP file that contains the to be used Songs for the YouTube Shorts Automater.
- `email`: `object`:
    - `smtp_server`: `string` - Your SMTP server.
    - `smtp_port`: `number` - The port of your SMTP server.
    - `username`: `string` - Your email address.
    - `password`: `string` - Your email password.
- `google_maps_scraper_niche`: `string` - The niche you want to scrape Google Maps for.
- `scraper_timeout`: `number` - The timeout for the Google Maps scraper.
- `outreach_message_subject`: `string` - The subject of your outreach message. `{{COMPANY_NAME}}` will be replaced with the company name.
- `outreach_message_body_file`: `string` - The file that contains the body of your outreach message, should be HTML. `{{COMPANY_NAME}}` will be replaced with the company name.
- `stt_provider`: `string` - Provider for subtitle transcription. Default is `local_whisper`. Options:
    * `local_whisper`
    * `third_party_assemblyai`
- `whisper_model`: `string` - Whisper model for local transcription (for example `base`, `small`, `medium`, `large-v3`).
- `whisper_device`: `string` - Device for local Whisper (`auto`, `cpu`, `cuda`).
- `whisper_compute_type`: `string` - Compute type for local Whisper (`int8`, `float16`, etc.).
- `assembly_ai_api_key`: `string` - Your Assembly AI API key. Get yours from [here](https://www.assemblyai.com/app/).
- `tts_voice`: `string` - Voice for KittenTTS text-to-speech. Default is `Jasper`. Options: `Bella`, `Jasper`, `Luna`, `Bruno`, `Rosie`, `Hugo`, `Kiki`, `Leo`.
- `font`: `string` - The font that will be used to generate images. This should be a `.ttf` file in the `fonts/` directory.
- `imagemagick_path`: `string` - The path to the ImageMagick binary. This is used by MoviePy to render subtitles. If left empty or set to the old placeholder value, MPV2 will try to auto-detect `magick` or `convert` from your system `PATH`.
- `script_sentence_length`: `number` - The number of sentences in the generated video script (default: `4`).
- `zhipu_api_key`: `string` - Optional Zhipu API key used only for the OCR vision fallback in `NewsPipeline` when local OCR is weak.

## Example

```json
{
  "verbose": true,
  "firefox_profile": "",
  "headless": false,
  "ollama_base_url": "http://127.0.0.1:11434",
  "ollama_model": "",
  "twitter_language": "English",
  "nanobanana2_api_base_url": "https://generativelanguage.googleapis.com/v1beta",
  "nanobanana2_api_key": "",
  "nanobanana2_model": "gemini-3.1-flash-image-preview",
  "nanobanana2_aspect_ratio": "9:16",
  "threads": 2,
  "zip_url": "",
  "is_for_kids": false,
  "google_maps_scraper": "https://github.com/gosom/google-maps-scraper/archive/refs/tags/v0.9.7.zip",
  "email": {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "",
    "password": ""
  },
  "google_maps_scraper_niche": "",
  "scraper_timeout": 300,
  "outreach_message_subject": "I have a question...",
  "outreach_message_body_file": "outreach_message.html",
  "stt_provider": "local_whisper",
  "whisper_model": "base",
  "whisper_device": "auto",
  "whisper_compute_type": "int8",
  "assembly_ai_api_key": "",
  "tts_voice": "Jasper",
  "font": "bold_font.ttf",
  "imagemagick_path": "",
  "script_sentence_length": 4,
  "llm_provider": "openai_compatible",
  "openai_base_url": "https://api.minimaxi.com/v1",
  "openai_api_key": "",
  "openai_model": "MiniMax-M2.7",
  "zhipu_api_key": ""
}
```

## Environment Variable Fallbacks

- `GEMINI_API_KEY`: used when `nanobanana2_api_key` is empty.
- `OPENAI_BASE_URL`: used when `openai_base_url` is empty.
- `OPENAI_API_KEY`: used when `openai_api_key` is empty.
- `OPENAI_MODEL`: used when `openai_model` is empty.

Example:

```bash
export GEMINI_API_KEY="your_api_key_here"
export OPENAI_BASE_URL="https://api.minimaxi.com/v1"
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="MiniMax-M2.7"
```
