<div align="center">

# VideoCraft AI 🎬

### Next-Generation AI Automated Video & Reels Production Platform

Provide a video **topic** or **keyword**, and VideoCraft AI will generate scripts, match high-definition footage, create stylish subtitles and background music, and produce viral videos automatically.

[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

English | [ગુજરાતી (Gujarati) Supported]

</div>

---

## Features 🎯

- [x] Provides **Modern Glassmorphism WebUI**, **CLI**, and **API** workflows
- [x] Supports **AI-generated video scripts** and custom prompts
- [x] Supports **HD video formats**:
  - [x] Portrait 9:16 (`1080x1920`) for YouTube Shorts, TikTok, Instagram Reels
  - [x] Landscape 16:9 (`1920x1080`) for YouTube and standard displays
- [x] Supports **batch video generation**
- [x] Supports **multilingual UI & video scripts** (English, Gujarati, Spanish, German, etc.)
- [x] Supports multiple AI Voice Synthesis engines (**Edge TTS**, **Azure Speech**, **Google Gemini TTS**, **ElevenLabs**, **SiliconFlow**)
- [x] Supports **dynamic subtitle generation** with custom fonts, colors, outlines, and frosted background styles
- [x] Supports stock video sources (**Pexels**, **Pixabay**, **Coverr**) and local footage
- [x] Supports leading AI models (**OpenAI**, **Anthropic Claude**, **Google Gemini**, **DeepSeek**, **Ollama**, **LiteLLM**, **Groq**)

---

## Quick Start 🚀

### Windows (PowerShell)

```powershell
# 1. Run the native PowerShell launcher
.\webui.ps1

# Optional: Run on custom port or listen on LAN
.\webui.ps1 -Port 8505 -HostAddress 0.0.0.0
```

### macOS / Linux

```shell
sh webui.sh
```

---

## Manual Setup 📦

### 1. Create Virtual Environment & Install Dependencies

```shell
uv sync --frozen
```

Or using standard `pip`:

```shell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the WebUI

```powershell
.\webui.ps1
```

Open your browser at `http://127.0.0.1:8501`.

---

## Configuration ⚙️

Customize branding and API keys in `config.toml`:

```toml
project_name = "VideoCraft AI"
project_description = "✨ VideoCraft AI - Advanced AI Automated Video Production Studio"
github_repo = "https://github.com/your-username/your-repo-name"

[app]
hide_config = false
script_generation_backend = "local"
```

---

## License 📝

MIT License
