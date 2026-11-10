# Llama Assistant Terminal

A voice + text assistant for macOS that runs mostly on local models (Ollama) and calls Gemini for coding.

## Setup
```bash
brew install portaudio python@3.12 python-tk@3.12
$(brew --prefix)/bin/python3.12 -m venv ai_env && source ai_env/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:3b && ollama pull qwen2.5:7b && ollama pull qwen2.5vl:3b
echo "GEMINI_API_KEY=your_key" > .env        # optional, for Gemini
python assistent.py                           # or: python assistent.py --selftest
```

## How it decides
1. Instant commands (volume, Spotify, open app/folder/file, timers, "run it", "undo") – no AI, several per sentence.
2. Small local model (qwen2.5:3b) with only the matching tools.
3. Big local model (qwen2.5:7b) for hard tasks or after the small one fails.
4. Gemini for coding, screen questions and as a last resort (`/offline on` blocks it).

## Commands
`/mode always|wake|push` · `/lang auto|en|cs` · `/model auto|fast|smart|cloud` · `/stt whisper|google|mlx`
`/offline on|off` · `/mic [n]` · `/undo` · `/pin` · `/compact` · `/autostart on|off` · `/memory` · `/history` · `/help`

Start a sentence with `gemini:` to force Gemini or `think:` to force the big local model.

## Files (in ~/.llama_assistant)
`settings.json` (remembered choices) · `memory.json` · `timers.json` · `history.log`

## Safety
Side-effect tools only run if your own words mention the topic; code is written only inside your home folder
(never hidden files or ~/Library); replacing a file or running a script always asks first; scripts run without
your API keys in their environment; every code change can be undone with `/undo`.