"""
Llama Assistant Terminal (macOS)

Install:
    brew install portaudio
    pip install ollama SpeechRecognition pyaudio faster-whisper numpy ddgs
    ollama pull qwen2.5:3b
    ollama pull llama3.2-vision      # only for "look at my screen" questions
"""
import difflib
import json
import os
import plistlib
import queue
import shutil
import re
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import unicodedata
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from tkinter import scrolledtext

import ollama
import speech_recognition as sr

# ==========================================
# CONFIG
# ==========================================
FAST_MODEL = "qwen2.5:3b"           # handles everyday commands quickly
SMART_MODEL = "qwen2.5:7b"          # takes over harder tasks (ollama pull qwen2.5:7b)
AUTO_ESCALATE = True                # let the assistant decide when to switch to SMART_MODEL
SMART_FOR_SYNTHESIS = True          # use SMART_MODEL to read web results / clipboard / file lists
SMART_KEEP_ALIVE = "5m"             # unload the big model sooner to save RAM
SMART_OPTIONS = {"temperature": 0, "num_ctx": 4096, "num_predict": 400}
CLOUD_ENABLED = True                # Gemini takes over coding & hard tasks (needs GEMINI_API_KEY in a .env file)
CLOUD_AUTO = True                   # route coding requests to Gemini automatically
CLOUD_VISION = True                 # "what's on my screen?" is answered by Gemini (the screenshot goes online); False = local vision model
STREAM_TTS = True                   # start speaking while the answer is still being generated
HOTKEY = "<ctrl>+<alt>+<space>"     # global push-to-talk hotkey (needs: pip install pynput)
MLX_MODEL = "mlx-community/whisper-small-mlx"   # used by /stt mlx (pip install mlx-whisper)
RUN_TIMEOUT = 15                    # seconds a script started with "run it" may run
WHISPER_HINT = "Spotify, Safari, Visual Studio Code, Finder, Desktop, Documents, Downloads, Python."   # helps Whisper with app names
GEMINI_MODELS = ["gemini-flash-latest", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"]  # tried in order
VISION_MODEL = "qwen2.5vl:3b"       # only used for screenshots (alternatives: "gemma3:4b", "llama3.2-vision")
LANGUAGE = "en"                     # "en" = speak & hear English, "cs" = Czech  (switch live: /lang en|cs)
STT_LANGS = {"en": "en-US", "cs": "cs-CZ"}
SPEAK_REPLIES = True
VOICES = {"en": "Samantha", "cs": "Zuzana"}   # macOS voices (English alternatives: Daniel, Karen, Moira)

MAX_TOOL_ROUNDS = 5
MAX_HISTORY = 20
KEEP_ALIVE = "30m"
OLLAMA_OPTIONS = {"temperature": 0, "num_ctx": 4096, "num_predict": 150}
FAST_REPLY = True                   # after successful tools, answer with the tool result (no 2nd AI call)
STREAM_REPLIES = True               # print the AI answer word by word
NEEDS_LLM_AFTER = {"get_clipboard_content", "web_search", "find_files", "read_file", "run_file", "run_and_fix",
                   "get_selected_text", "get_calendar_events"}
SMART_SYNTHESIS_TOOLS = {"get_clipboard_content", "web_search", "get_selected_text"}   # these hand over to SMART_MODEL

STT_ENGINE = "whisper"              # "whisper" = offline (faster-whisper), "google" = online
WHISPER_MODEL = "small"             # "base" = faster, "medium" = more accurate
LISTEN_MODE = "always"              # "always" = hands-free, "wake" = say "Hej Llama" first, "push" = push-to-talk
MODES = ["always", "wake", "push"]  # order used by the mode button
ACTIVE_WINDOW_SECONDS = 20          # after an interaction, no wake word needed for this long
PAUSE_SECONDS = 1.4                 # silence that ends a sentence (higher = fewer cut-off prompts)
PHRASE_LIMIT = 30                   # max length of one spoken prompt in seconds
ENERGY_RATIO = 1.2                  # mic trigger level vs. room noise (default 1.5; lower = more sensitive)
VOICE_DEBUG = True                  # show what the mic captured / heard (set False once it works)
SEARCH_URL = "https://www.google.com/search?q="   # e.g. "https://duckduckgo.com/?q="
ENABLE_WEB_ANSWERS = True           # False = never read search results aloud, only open the browser
DEFAULT_CITY = ""                   # "" = wttr.in guesses by IP; or e.g. "Praha"

APP_DIR = os.path.expanduser("~/.llama_assistant")
MEMORY_FILE = os.path.join(APP_DIR, "memory.json")
HISTORY_FILE = os.path.join(APP_DIR, "history.log")
os.makedirs(APP_DIR, exist_ok=True)

SYSTEM_PROMPTS = {
    "en": (
        "You are a voice assistant for macOS. Answer briefly, in one sentence at most. "
        "Always reply in English. Use the available tools to carry out tasks. "
        "When the user asks you to remember something, use the remember tool. "
        "When the user says search / look up / google / find on the internet, ALWAYS use search_google "
        "(it opens the results in the browser) and never web_search. "
        "Use web_search only for a specific question the user wants answered out loud. "
        "To open a file or folder, call open_file with just its name (or a path returned by find_files); never invent paths. "
        "To open a file in a specific app (e.g. Visual Studio Code) call open_file_with once. "
        "To write or change code use write_code (default target: the last opened file) and put only code in its code argument. "
        "Only call tools the user actually asked for."
    ),
    "cs": (
        "Jsi hlasový asistent pro macOS. Odpovídej stručně, maximálně jednou větou. "
        "Vždy odpovídej česky. Používej dostupné nástroje k plnění úkolů. "
        "Když uživatel chce, aby sis něco zapamatoval, použij nástroj remember. "
        "Když uživatel řekne vyhledej / hledej / najdi na internetu / googli, VŽDY použij search_google "
        "(otevře výsledky v prohlížeči) a nikdy web_search. "
        "web_search použij jen na konkrétní otázku, na kterou chce uživatel slyšet odpověď. "
        "K otevření souboru nebo složky zavolej open_file jen s názvem (nebo cestou z find_files), cesty si nevymýšlej. "
        "K otevření souboru v konkrétní aplikaci zavolej jednou open_file_with. "
        "Kód píšeš nebo měníš přes write_code (výchozí cíl: naposledy otevřený soubor), v argumentu code jen kód. "
        "Volej jen nástroje, o které uživatel opravdu požádal."
    ),
}

TRIGGER_WORDS = ["podívej", "obrazovk", "vidíš", "co je na", "přečti",
                 "look at", "screen", "what do you see", "read this"]
STOP_WORDS = {"stop", "ticho", "prestan", "quiet", "dost"}
YES_WORDS = {"y", "yes", "ano", "jo", "jasne", "ok", "okay", "jop", "urcite", "potvrzuji"}
NO_WORDS = {"n", "no", "ne", "nechci", "zrus", "zrusit", "stop"}
WAKE_RE = re.compile(r"\b(hej|hey|ahoj|ok|okay)[\s,.!-]+(llamas?|lama|lamo|lamy|llamma|lamma|asistent\w*|assistant)\b[\s,.!?:;-]*")

# ==========================================
# SHARED STATE
# ==========================================
ui_q = queue.Queue()     # (tag, text) -> terminal window
cmd_q = queue.Queue()    # (source, text) -> worker thread
stop = threading.Event()
voice_on = threading.Event()
voice_on.set()
busy = threading.Event()             # a command is being processed
speaking = threading.Event()         # text-to-speech is playing
talk_once = threading.Event()        # push-to-talk button pressed
confirm_pending = threading.Event()  # waiting for a yes/no
confirm_event = threading.Event()

STATE = {
    "lang": LANGUAGE,
    "lang_mode": "auto",    # "auto" = follow the language you speak / type, "fixed" = stay on "lang"
    "offline": False,       # True = nothing leaves this Mac (no Gemini, web search, weather, Google speech)
    "mic_index": None,      # chosen microphone (None = system default)
    "mic_changed": False,
    "spoken_stream": False, # sentences were already spoken while streaming
    "last_results": [],     # last find_files results ("open the second one")
    "last_stt_s": 0.0,
    "timing": "",
    "full_tools": False,    # True = give the local model every tool instead of the matching subset
    "pinned": False, "compact": False, "geo": "",
    "turn_fold": "",        # what the user said this turn (folded) – used by the intent guard
    "model_mode": "auto",   # "auto" | "fast" | "smart"
    "smart_ok": True,       # False once SMART_MODEL turned out not to be installed
    "smart_loaded": False,
    "cloud_ok": True,       # False when no key / package / Gemini disabled
    "last_file": "",        # last file that was opened or written (target for "write code inside it")
    "last_model": "",
    "speak_seq": 0,         # increments whenever the assistant starts speaking
    "phase": "",            # "", "loading" (speech model) or "transcribing"
    "mode": LISTEN_MODE,
    "stt": STT_ENGINE,
    "stream": STREAM_REPLIES,
    "active_until": 0.0,
    "talk_deadline": 0.0,
    "confirm_answer": False,
    "speech_proc": None,
}
messages = [{"role": "system", "content": SYSTEM_PROMPTS[LANGUAGE]}]
tts_q = queue.Queue()   # sentences waiting to be spoken while the answer streams
active_timers = {}   # id -> (end_time, message)
_timer_counter = [0]
_mem_lock = threading.Lock()


# ==========================================
# HELPERS
# ==========================================
_L10N = [
    (r"^Aplikace (.+) nenalezena\.$", r"App \1 not found."),
    (r"^Aplikace (.+) byla otevřena\.$", r"Opened \1."),
    (r"^Zrušeno\.$", "Cancelled."),
    (r"^Opravdu zavřít aktivní okno\?$", "Really close the active window?"),
    (r"^Aktivní okno bylo zavřeno\.$", "Closed the active window."),
    (r"^Složka (.+) vytvořena\.$", r"Created folder \1."),
    (r"^Soubor (.+) už existuje\. Přepsat\?$", r"File \1 already exists. Overwrite?"),
    (r"^Soubor (.+) vytvořen\.$", r"Created file \1."),
    (r"^Text byl vložen\.$", "Text pasted."),
    (r"^Schránka je prázdná\.$", "The clipboard is empty."),
    (r"^Schránka: ", "Clipboard: "),
    (r"^Text je ve schránce\.$", "Copied to the clipboard."),
    (r"^Web (.+) otevřen\.$", r"Opened \1."),
    (r"^Vyhledávání '(.+)' otevřeno\.$", r"Searching the web for \1."),
    (r"^Chyba: chybí balíček ddgs", "Error: the ddgs package is missing"),
    (r"^Chyba vyhledávání: ", "Search error: "),
    (r"^Nic jsem nenašel\.$", "Nothing found."),
    (r"^Neplatná akce: ", "Invalid action: "),
    (r"^Neplatný režim: ", "Invalid mode: "),
    (r"^Neplatný směr: ", "Invalid direction: "),
    (r"^Neplatný stav: ", "Invalid state: "),
    (r"^Hlasitost nastavena na (\d+) %\.$", r"Volume set to \1%."),
    (r"^Chyba: neplatný čas\.$", "Error: invalid time."),
    (r"^Časovač nastaven na (.+?): (.+)\.$", r"Timer set for \1: \2."),
    (r"^Časovač skončil$", "Timer finished"),
    (r"^Žádné časovače neběží\.$", "No timers running."),
    (r"^Připomínka '(.+)' přidána\.$", r"Reminder '\1' added."),
    (r"^Chyba: čas musí být ve formátu", "Error: time must be in the format"),
    (r"^Událost '(.+)' je připravená v Kalendáři, potvrď ji tam\.$", r"Event '\1' is ready in Calendar – confirm it there."),
    (r"^Chyba: počasí se nepodařilo zjistit", "Error: couldn't get the weather"),
    (r"^Baterie: (\d+) %, (.+)\.$", r"Battery: \1%, \2."),
    (r"^Informace o baterii nejsou k dispozici\.$", "Battery information isn't available."),
    (r"^Vzhled: (.+)\.$", r"Appearance: \1."),
    (r"^Jas zvýšen\.$", "Brightness increased."),
    (r"^Jas snížen\.$", "Brightness decreased."),
    (r"^Opravdu vypnout Wi-Fi\?$", "Really turn off Wi-Fi?"),
    (r"^Wi-Fi je zapnuté\.$", "Wi-Fi is on."),
    (r"^Wi-Fi je vypnuté\.$", "Wi-Fi is off."),
    (r"^Wi-Fi zapnuto\.$", "Wi-Fi turned on."),
    (r"^Wi-Fi vypnuto\.$", "Wi-Fi turned off."),
    (r"^Chyba: hledání trvalo příliš dlouho\.$", "Error: the search took too long."),
    (r"^Chyba: soubor nenalezen nebo je mimo domovskou složku\.$", "Error: file not found or outside your home folder."),
    (r"^Chyba: tento typ souboru neotevírám\.$", "Error: I won't open this type of file."),
    (r"^Soubor otevřen: (.+)$", r"Opened file: \1"),
    (r"^Otevřeno: (.+) v (.+)$", r"Opened \1 in \2."),
    (r"^Chyba: nepodařilo se otevřít v (.+)\.$", r"Error: couldn't open it in \1."),
    (r"^Otevřeno: (.+)$", r"Opened: \1"),
    (r"^Chyba: nenalezeno: (.+)$", r"Error: couldn't find \1."),
    (r"^Spustit (.+) a až (\d)x automaticky opravit chyby\?$", r"Run \1 and auto-fix errors up to \2 times?"),
    (r"^Chyba: oprava selhala", "Error: the automatic fix failed"),
    (r"^Chyba: offline režim je zapnutý\.$", "Error: offline mode is on."),
    (r"^Chyba: neplatné číslo\.$", "Error: invalid number."),
    (r"^Chyba: není z čeho vybírat\.$", "Error: there are no results to pick from."),
    (r"^Chyba: vyber číslo 1 až (\d+)\.$", r"Error: pick a number from 1 to \1."),
    (r"^Označený text: ", "Selected text: "),
    (r"^Chyba: nic není označeno\.$", "Error: nothing is selected."),
    (r"^Aktivní aplikace: (.+)$", r"Frontmost app: \1"),
    (r"^Chyba: nepodařilo se zjistit aktivní aplikaci\.$", "Error: couldn't tell which app is in front."),
    (r"^Zkratky: ", "Shortcuts: "),
    (r"^Nemáš žádné zkratky\.$", "You have no shortcuts."),
    (r"^Chyba: aplikace Zkratky není dostupná\.$", "Error: the Shortcuts app isn't available."),
    (r"^Spustit zkratku (.+)\?$", r"Run the shortcut \1?"),
    (r"^Zkratka (.+) spuštěna\.$", r"Ran the shortcut \1."),
    (r"^Chyba: zkratka běžela příliš dlouho\.$", "Error: the shortcut ran too long."),
    (r"^Chyba: zkratka selhala", "Error: the shortcut failed"),
    (r"^Události:", "Events:"),
    (r"^Žádné události\.$", "No events."),
    (r"^Chyba: Kalendář neodpověděl včas\.$", "Error: Calendar didn't respond in time."),
    (r"^Spuštěno (.+) \(kód (-?\d+)\):", r"Ran \1 (exit code \2):"),
    (r"^Chyba: program běžel příliš dlouho \(limit (\d+) s\)\.$", r"Error: the program ran too long (\1 s limit)."),
    (r"^Chyba: spouštím jen soubory \.py\.$", "Error: I only run .py files."),
    (r"^Spustit (.+)\?$", r"Run \1?"),
    (r"^Vráceno do původního stavu: (.+)$", r"Restored the previous version of \1."),
    (r"^Nový soubor (.+) odstraněn\.$", r"Removed the new file \1."),
    (r"^Chyba: není co vracet\.$", "Error: nothing to undo."),
    (r"^Kód přidán do (.+)$", r"Added the code to \1."),
    (r"^Kód zapsán do (.+)$", r"Wrote the code to \1."),
    (r"^Chyba: žádný soubor není vybrán\.$", "Error: no file selected – tell me which file."),
    (r"^Chyba: sem nesmím zapisovat\.$", "Error: I'm not allowed to write there."),
    (r"^Chyba: tento soubor nemohu číst\.$", "Error: I can't read that file."),
    (r"^Chyba: to je složka, ne soubor\.$", "Error: that's a folder, not a file."),
    (r"^Chyba: kód je příliš dlouhý\.$", "Error: the code is too long."),
    (r"^Chyba: soubor je příliš velký\.$", "Error: the file is too big."),
    (r"^Zapamatováno\.$", "Got it, I'll remember that."),
    (r"^Smazáno poznámek: (\d+)\.$", r"Deleted notes: \1."),
    (r"^Chyba nástroje: ", "Tool error: "),
    (r"^Nástroj (.+) neexistuje\.$", r"Tool \1 does not exist."),
    (r"^Chyba: ", "Error: "),
]


def localize(text: str) -> str:
    """Tool results are written in Czech internally; translate them when the language is English."""
    if STATE["lang"] != "en" or not text:
        return text
    for pat, repl in _L10N:
        new, n = re.subn(pat, repl, text, count=1)
        if n:
            return new.replace("Časovač skončil", "Timer finished")
    return text.replace("Časovač skončil", "Timer finished")


def save_settings():
    try:
        with open(os.path.join(APP_DIR, "settings.json"), "w", encoding="utf-8") as f:
            json.dump({"lang": STATE["lang"], "mode": STATE["mode"], "model_mode": STATE["model_mode"],
                       "stt": STATE["stt"], "voice_on": voice_on.is_set(), "lang_mode": STATE["lang_mode"],
                       "offline": STATE["offline"], "mic_index": STATE["mic_index"]}, f)
    except OSError:
        pass


def load_settings():
    try:
        with open(os.path.join(APP_DIR, "settings.json"), encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return
    if d.get("lang") in ("en", "cs"):
        STATE["lang"] = d["lang"]
    if d.get("mode") in MODES:
        STATE["mode"] = d["mode"]
    if d.get("model_mode") in ("auto", "fast", "smart", "cloud"):
        STATE["model_mode"] = d["model_mode"]
    if d.get("stt") in ("whisper", "google", "mlx"):
        STATE["stt"] = d["stt"]
    if d.get("voice_on") is False:
        voice_on.clear()
    if d.get("lang_mode") in ("auto", "fixed"):
        STATE["lang_mode"] = d["lang_mode"]
    if isinstance(d.get("offline"), bool):
        STATE["offline"] = d["offline"]
    if isinstance(d.get("mic_index"), int):
        STATE["mic_index"] = d["mic_index"]


_CS_CHARS = set("ěščřžýáíéúůťďň")
_EN_WORDS = {"the", "is", "are", "and", "open", "what", "how", "please", "my", "it", "of", "in", "for", "with",
             "set", "play", "write", "show", "you", "can", "this", "that", "me", "an", "tell", "make", "turn"}
_CS_WORDS = {"otevri", "spust", "je", "jsou", "na", "se", "si", "ten", "ta", "co", "jak", "prosim", "muj", "mi",
             "pro", "nastav", "zapni", "vypni", "pust", "napis", "ukaz", "kolik", "kde", "kdy", "hlasitost", "pauza", "dalsi"}


def guess_lang(text: str):
    """'en' / 'cs' guessed from the words of a typed sentence, or None when unsure."""
    if any(c in _CS_CHARS for c in text.lower()):
        return "cs"
    words = re.findall(r"[a-z]+", fold(text))
    en = sum(w in _EN_WORDS for w in words)
    cs = sum(w in _CS_WORDS for w in words)
    return "en" if en > cs else "cs" if cs > en else None


def _set_lang(code: str):
    if code != STATE["lang"]:
        STATE["lang"] = code
        log("info", f"🌐 Language → {code.upper()}")


def wake_phrase() -> str:
    return "Hey Llama" if STATE["lang"] == "en" else "Hej Llama"


def log(tag: str, text: str):
    ui_q.put((tag, text))


def fold(text: str) -> str:
    """Lowercase and strip diacritics, 1 character in -> 1 character out."""
    return "".join(unicodedata.normalize("NFD", c)[0] for c in text.lower())


def as_str(s: str) -> str:
    """Escape text for use inside an AppleScript string."""
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def notify(title: str, message: str):
    subprocess.run(["osascript", "-e",
                    f'display notification "{as_str(message)}" with title "{as_str(title)}"'])


def speak(text: str):
    if not SPEAK_REPLIES or not text.strip():
        return
    STATE["speak_seq"] += 1
    speaking.set()
    try:
        proc = subprocess.Popen(["say", "-v", VOICES[STATE["lang"]], text])
        STATE["speech_proc"] = proc
        proc.wait()
    except Exception as e:
        log("error", f"Speech failed: {e}")
    finally:
        STATE["speech_proc"] = None
        speaking.clear()


def stop_speaking():
    try:
        while True:
            tts_q.get_nowait()
            tts_q.task_done()
    except queue.Empty:
        pass
    proc = STATE.get("speech_proc")
    if proc and proc.poll() is None:
        proc.terminate()


def ask_confirm(question: str, timeout: int = 30) -> bool:
    """Ask the user yes/no (typed or spoken). Runs inside the worker thread."""
    confirm_event.clear()
    STATE["confirm_answer"] = False
    confirm_pending.set()
    question = localize(question)
    log("confirm", f"❓ {question} ({'yes/no' if STATE['lang'] == 'en' else 'ano/ne'})")
    speak(question)
    got = confirm_event.wait(timeout)
    confirm_pending.clear()
    if not got:
        log("info", "(no answer – cancelled)")
        return False
    return STATE["confirm_answer"]


def answer_confirm(text: str) -> bool:
    """If a confirmation is pending, consume `text` as the answer. Returns True if consumed."""
    if not confirm_pending.is_set():
        return False
    t = re.sub(r"[^\w ]", "", fold(text)).strip()
    first = t.split()[0] if t else ""
    if t in YES_WORDS or first in YES_WORDS:
        STATE["confirm_answer"] = True
    elif t in NO_WORDS or first in NO_WORDS:
        STATE["confirm_answer"] = False
    else:
        log("info", "Please answer yes or no (ano / ne).")
        return True
    confirm_event.set()
    return True


def extend_active():
    STATE["active_until"] = time.time() + ACTIVE_WINDOW_SECONDS


def is_active() -> bool:
    return time.time() < STATE["active_until"]


def fmt_duration(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h} h")
    if m:
        parts.append(f"{m} min")
    if s and not h:
        parts.append(f"{s} s")
    return " ".join(parts) or "0 s"


def safe_desktop_path(name: str) -> str:
    base = os.path.expanduser("~/Desktop")
    return os.path.abspath(os.path.join(base, os.path.basename(name)))


def load_memory() -> list:
    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_memory(notes: list):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


def build_system_prompt(smart: bool = False) -> str:
    en = STATE["lang"] == "en"
    now = datetime.now().strftime("%A %Y-%m-%d %H:%M")
    notes = load_memory()
    head = "Permanent notes about the user:" if en else "Trvalé poznámky o uživateli:"
    mem = (f"\n{head}\n" + "\n".join(f"- {n}" for n in notes)) if notes else ""
    extra = ""
    if smart:
        extra = ("\nThis task is more complex: instead of one sentence you may answer in more detail (at most 5 sentences)."
                 if en else "\nTento úkol je složitější: místo jedné věty smíš odpovědět podrobněji (nejvýše 5 vět).")
    when = "Current date and time" if en else "Aktuální datum a čas"
    lf = ""
    if STATE["last_file"]:
        lf = f"\n{'Last opened file' if en else 'Naposledy otevřený soubor'}: {STATE['last_file']}"
    return f"{SYSTEM_PROMPTS[STATE['lang']]}{extra}\n{when}: {now}.{lf}{mem}"


# ==========================================
# TOOLS
# ==========================================
SPOTIFY_ACTIONS = {"playpause", "next track", "previous track"}
BLOCKED_EXT = {".app", ".command", ".sh", ".pkg", ".dmg", ".terminal", ".workflow", ".scpt", ".jar"}


def open_application(app_name: str) -> str:
    """Otevře aplikaci na macOS podle názvu."""
    r = subprocess.run(["open", "-a", app_name], capture_output=True)
    if r.returncode != 0:
        return f"Aplikace {app_name} nenalezena."
    return f"Aplikace {app_name} byla otevřena."


def close_active_window() -> str:
    """Zavře aktuálně aktivní okno (vyžádá potvrzení)."""
    if not ask_confirm("Opravdu zavřít aktivní okno?"):
        return "Zrušeno."
    subprocess.run(["osascript", "-e",
                    'tell application "System Events" to keystroke "w" using command down'])
    return "Aktivní okno bylo zavřeno."


def create_folder(folder_name: str) -> str:
    """Vytvoří novou složku na ploše."""
    os.makedirs(safe_desktop_path(folder_name), exist_ok=True)
    return f"Složka {folder_name} vytvořena."


def write_to_file(filename: str, content: str) -> str:
    """Vytvoří textový soubor na ploše a zapíše obsah (při přepsání se zeptá)."""
    path = safe_desktop_path(filename)
    if os.path.exists(path) and not ask_confirm(f"Soubor {filename} už existuje. Přepsat?"):
        return "Zrušeno."
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Soubor {filename} vytvořen."


def type_text(text: str) -> str:
    """Vloží zadaný text do aktuálně aktivní aplikace."""
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    p.communicate(text.encode("utf-8"))
    subprocess.run(["osascript", "-e",
                    'tell application "System Events" to keystroke "v" using command down'])
    return "Text byl vložen."


def get_clipboard_content() -> str:
    """Přečte text ze schránky."""
    r = subprocess.run(["pbpaste"], capture_output=True, text=True)
    return f"Schránka: {r.stdout}" if r.stdout.strip() else "Schránka je prázdná."


def set_clipboard(text: str) -> str:
    """Zkopíruje zadaný text do schránky."""
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    p.communicate(text.encode("utf-8"))
    return "Text je ve schránce."


def open_website(url: str) -> str:
    """Otevře URL v Safari."""
    if not url.startswith("http"):
        url = "https://" + url
    subprocess.run(["open", "-a", "Safari", url])
    return f"Web {url} otevřen."


def search_google(query: str) -> str:
    """Vyhledá dotaz na internetu: otevře výsledky vyhledávání v prohlížeči.
    Použij vždy, když uživatel řekne vyhledej, hledej, najdi na internetu nebo googli.

    Args:
        query: Přesně to, co má uživatel vyhledat.
    """
    url = SEARCH_URL + urllib.parse.quote(query)
    subprocess.run(["open", "-a", "Safari", url])
    return f"Vyhledávání '{query}' otevřeno."


def web_search(query: str) -> str:
    """Přečte krátké výsledky z webu, aby šlo odpovědět na konkrétní otázku (např. 'kdo vyhrál včera').
    NEPOUŽÍVEJ, když uživatel řekne vyhledej / hledej / najdi na internetu – tehdy použij search_google.

    Args:
        query: Otázka nebo téma, na které se hledá odpověď.
    """
    if STATE["offline"]:
        return "Chyba: offline režim je zapnutý."
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        return "Chyba: chybí balíček ddgs (pip install ddgs)."
    try:
        results = DDGS().text(query, max_results=4)
    except Exception as e:
        return f"Chyba vyhledávání: {e}"
    if not results:
        return "Nic jsem nenašel."
    return "\n".join(f"- {r.get('title', '')}: {str(r.get('body', ''))[:250]}" for r in results)


def control_spotify(action: str) -> str:
    """Ovládá Spotify.

    Args:
        action: 'playpause', 'next track' nebo 'previous track'.
    """
    if action not in SPOTIFY_ACTIONS:
        return f"Neplatná akce: {action}"
    subprocess.run(["osascript", "-e", f'tell application "Spotify" to {action}'])
    return f"Spotify: {action}."


def set_volume(level: int) -> str:
    """Nastaví systémovou hlasitost Macu (0-100)."""
    level = max(0, min(100, int(level)))
    subprocess.run(["osascript", "-e", f"set volume output volume {level}"])
    return f"Hlasitost nastavena na {level} %."


TIMERS_FILE = os.path.join(APP_DIR, "timers.json")


def _save_timers():
    try:
        with open(TIMERS_FILE, "w", encoding="utf-8") as f:
            json.dump([{"end": end, "message": msg} for end, msg in active_timers.values()], f)
    except OSError:
        pass


def _arm_timer(end: float, message: str):
    _timer_counter[0] += 1
    tid = _timer_counter[0]

    def fire():
        active_timers.pop(tid, None)
        _save_timers()
        msg = localize(message)
        log("alarm", f"⏰ {msg}")
        notify("Llama Assistant", msg)
        speak(msg)

    t = threading.Timer(max(0.1, end - time.time()), fire)
    t.daemon = True
    t.start()
    active_timers[tid] = (end, message)


def restore_timers():
    """Re-arm timers that were still running when the assistant was closed."""
    try:
        with open(TIMERS_FILE, encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        return
    now = time.time()
    for item in saved:
        try:
            end, message = float(item["end"]), str(item["message"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > now:
            _arm_timer(end, message)
        else:
            log("alarm", f"⏰ Missed while closed: {localize(message)}")
    _save_timers()


def set_timer(minutes: float, message: str = "Časovač skončil") -> str:
    """Nastaví časovač/připomínku. Po uplynutí ozve zprávu a pošle notifikaci.

    Args:
        minutes: Za kolik minut (může být desetinné, např. 0.5 = 30 s).
        message: Text, který se na konci ozve.
    """
    try:
        seconds = max(1, int(float(minutes) * 60))
    except (TypeError, ValueError):
        return "Chyba: neplatný čas."
    _arm_timer(time.time() + seconds, message)
    _save_timers()
    return f"Časovač nastaven na {fmt_duration(seconds)}: {message}."


def list_timers() -> str:
    """Vypíše běžící časovače."""
    if not active_timers:
        return "Žádné časovače neběží."
    now = time.time()
    return "; ".join(f"{msg} (za {fmt_duration(max(0, end - now))})" for end, msg in active_timers.values())


def add_reminder(title: str) -> str:
    """Přidá úkol do aplikace Připomínky."""
    r = subprocess.run(["osascript", "-e",
                        f'tell application "Reminders" to make new reminder with properties {{name:"{as_str(title)}"}}'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return f"Chyba: {r.stderr.strip() or 'nepodařilo se přidat připomínku'}"
    return f"Připomínka '{title}' přidána."


def add_calendar_event(title: str, start: str, duration_minutes: int = 60) -> str:
    """Připraví událost v Kalendáři (otevře import ke schválení).

    Args:
        title: Název události.
        start: Začátek ve formátu 'YYYY-MM-DD HH:MM'.
        duration_minutes: Délka v minutách.
    """
    try:
        dt = datetime.strptime(start.strip(), "%Y-%m-%d %H:%M")
        end = dt + timedelta(minutes=int(duration_minutes))
    except (ValueError, TypeError):
        return "Chyba: čas musí být ve formátu YYYY-MM-DD HH:MM."
    safe = re.sub(r"[\r\n]+", " ", title).replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;")
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//LlamaAssistant//EN", "BEGIN:VEVENT",
        f"UID:{uuid.uuid4()}@llama-assistant",
        f"DTSTAMP:{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        f"DTSTART:{dt:%Y%m%dT%H%M%S}", f"DTEND:{end:%Y%m%dT%H%M%S}",
        f"SUMMARY:{safe}", "END:VEVENT", "END:VCALENDAR",
    ]
    path = os.path.join(tempfile.gettempdir(), f"event_{uuid.uuid4().hex[:8]}.ics")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")
    subprocess.run(["open", path])
    return f"Událost '{title}' je připravená v Kalendáři, potvrď ji tam."


def get_weather(city: str = "") -> str:
    """Zjistí aktuální počasí (prázdné město = podle IP adresy)."""
    if STATE["offline"]:
        return "Chyba: offline režim je zapnutý."
    city = city or DEFAULT_CITY
    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=%l:+%C,+%t,+%w&lang={STATE['lang']}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception as e:
        return f"Chyba: počasí se nepodařilo zjistit ({e})."


def get_battery() -> str:
    """Zjistí stav baterie."""
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    m = re.search(r"(\d+)%;\s*([^;]+)", out)
    return f"Baterie: {m.group(1)} %, {m.group(2).strip()}." if m else "Informace o baterii nejsou k dispozici."


def set_dark_mode(mode: str = "toggle") -> str:
    """Přepne vzhled macOS.

    Args:
        mode: 'dark', 'light' nebo 'toggle'.
    """
    exprs = {"dark": "true", "light": "false", "toggle": "not dark mode"}
    if mode not in exprs:
        return f"Neplatný režim: {mode}"
    subprocess.run(["osascript", "-e",
                    f'tell application "System Events" to tell appearance preferences to set dark mode to {exprs[mode]}'])
    return f"Vzhled: {mode}."


def change_brightness(direction: str, steps: int = 4) -> str:
    """Změní jas displeje.

    Args:
        direction: 'up' nebo 'down'.
        steps: Počet kroků (1-16).
    """
    codes = {"up": 144, "down": 145}
    if direction not in codes:
        return f"Neplatný směr: {direction}"
    steps = max(1, min(16, int(steps)))
    script = (f'tell application "System Events"\nrepeat {steps} times\n'
              f'key code {codes[direction]}\nend repeat\nend tell')
    subprocess.run(["osascript", "-e", script])
    return "Jas zvýšen." if direction == "up" else "Jas snížen."


def _wifi_device() -> str:
    out = subprocess.run(["networksetup", "-listallhardwareports"], capture_output=True, text=True).stdout
    m = re.search(r"Hardware Port: (?:Wi-Fi|AirPort)\s*\nDevice: (\w+)", out)
    return m.group(1) if m else "en0"


def wifi(state: str = "status") -> str:
    """Zjistí nebo přepne Wi-Fi (vypnutí vyžádá potvrzení).

    Args:
        state: 'status', 'on' nebo 'off'.
    """
    dev = _wifi_device()
    if state == "status":
        out = subprocess.run(["networksetup", "-getairportpower", dev],
                             capture_output=True, text=True).stdout.strip()
        return f"Wi-Fi je {'zapnuté' if out.endswith('On') else 'vypnuté'}."
    if state not in ("on", "off"):
        return f"Neplatný stav: {state}"
    if state == "off" and not ask_confirm("Opravdu vypnout Wi-Fi?"):
        return "Zrušeno."
    subprocess.run(["networksetup", "-setairportpower", dev, state])
    return f"Wi-Fi {'zapnuto' if state == 'on' else 'vypnuto'}."


JUNK_PARTS = ("/Library/", "/site-packages/", "/node_modules/", "/__pycache__/")


def search_home(query: str, limit: int = 8) -> list:
    """Spotlight name search in the home folder: junk removed, best matches first."""
    home = os.path.expanduser("~")
    r = subprocess.run(["mdfind", "-onlyin", home, "-name", query],
                       capture_output=True, text=True, timeout=10)
    q = query.strip().lower()
    common = tuple(os.path.join(home, d) + os.sep for d in ("Desktop", "Documents", "Downloads"))

    def wanted(p: str) -> bool:
        if not p or any(j in p for j in JUNK_PARTS):
            return False
        return not any(part.startswith(".") for part in p[len(home):].split("/") if part)

    def rank(p: str):
        name = os.path.basename(p).lower()
        return (0 if name == q else 1 if name.startswith(q) else 2,
                0 if p.startswith(common) else 1, len(p))

    return sorted({p for p in r.stdout.splitlines() if wanted(p)}, key=rank)[:limit]


def find_files(query: str) -> str:
    """Najde soubory a složky v domovské složce podle názvu (nejlepší shody první).

    Args:
        query: Část názvu souboru nebo složky.
    """
    try:
        paths = search_home(query)
    except subprocess.TimeoutExpired:
        return "Chyba: hledání trvalo příliš dlouho."
    STATE["last_results"] = list(paths)
    return "\n".join(f"{i}) {p}" for i, p in enumerate(paths, 1)) if paths else "Nic jsem nenašel."


def resolve_path(path: str):
    """Turn a name or a (possibly wrong) path into a real path, or None."""
    path = re.sub(r"^\d+[.)]\s+", "", path.strip())          # "2) /Users/..." from a numbered list
    home = os.path.realpath(os.path.expanduser("~"))
    p = os.path.expanduser(path.strip().strip("'\"`"))
    if p.startswith("/Users/"):                  # models often invent the user name (/Users/username/...)
        parts = p.split("/", 3)
        p = os.path.join(home, parts[3]) if len(parts) > 3 else home
    if os.path.isabs(p):
        candidates = [p]
        if p.startswith(home + os.sep):          # e.g. ~/tes123/test.py really lives in ~/Desktop/tes123/test.py
            rel = p[len(home) + 1:]
            candidates += [os.path.join(home, d, rel) for d in ("Desktop", "Documents", "Downloads")]
    else:
        candidates = [os.path.join(home, d, p) for d in ("Desktop", "Documents", "Downloads", "")]
    for c in candidates:
        if os.path.exists(c):
            return os.path.realpath(c)
    try:
        found = search_home(os.path.basename(p.rstrip("/")))
    except Exception:
        found = []
    return os.path.realpath(found[0]) if found else None


def open_file(path: str) -> str:
    """Otevře soubor nebo složku. Stačí její název nebo cesta; cesty si nikdy nevymýšlej.

    Args:
        path: Název (např. 'tes123'), '~/Desktop/tes123' nebo cesta vrácená z find_files.
    """
    home = os.path.realpath(os.path.expanduser("~"))
    real = resolve_path(path)
    if not real or not (real == home or real.startswith(home + os.sep)):
        return f"Chyba: nenalezeno: {os.path.basename(path.rstrip('/')) or path}"
    if os.path.splitext(real)[1].lower() in BLOCKED_EXT:
        return "Chyba: tento typ souboru neotevírám."
    if os.path.isfile(real):
        STATE["last_file"] = real
    subprocess.run(["open", real])
    return f"Otevřeno: {os.path.basename(real)}"


def allowed_path(real: str) -> bool:
    """Only files inside the home folder; never hidden files/folders or ~/Library."""
    home = os.path.realpath(os.path.expanduser("~"))
    real = os.path.realpath(real)
    if not real.startswith(home + os.sep):
        return False
    parts = real[len(home) + 1:].split(os.sep)
    if parts[0] == "Library" or any(part.startswith(".") for part in parts):
        return False
    return os.path.splitext(real)[1].lower() not in BLOCKED_EXT


def read_file(path: str = "") -> str:
    """Přečte textový soubor, abys viděl jeho obsah (výchozí: naposledy otevřený soubor).

    Args:
        path: Název nebo cesta souboru; prázdné = naposledy otevřený soubor.
    """
    real = resolve_path(path) if path else (STATE["last_file"] or None)
    if not real:
        return "Chyba: žádný soubor není vybrán." if not path else f"Chyba: nenalezeno: {os.path.basename(path.rstrip('/')) or path}"
    if not os.path.isfile(real) or not allowed_path(real):
        return "Chyba: tento soubor nemohu číst."
    if os.path.getsize(real) > 200_000:
        return "Chyba: soubor je příliš velký."
    with open(real, encoding="utf-8", errors="replace") as f:
        data = f.read(15000)
    STATE["last_file"] = real
    return f"Obsah souboru {os.path.basename(real)}:\n{data}"


UNDO_STACK = []     # (path, previous bytes or None if the file did not exist)


def _remember_for_undo(target: str):
    try:
        prev = open(target, "rb").read() if os.path.exists(target) else None
    except OSError:
        prev = None
    if prev is not None and len(prev) > 1_000_000:
        return
    UNDO_STACK.append((target, prev))
    del UNDO_STACK[:-20]


def undo_last_write() -> str:
    """Vrátí zpět poslední zápis kódu do souboru (undo)."""
    if not UNDO_STACK:
        return "Chyba: není co vracet."
    target, prev = UNDO_STACK.pop()
    name = os.path.basename(target)
    if prev is None:
        try:
            os.remove(target)
        except OSError:
            pass
        return f"Nový soubor {name} odstraněn."
    with open(target, "wb") as f:
        f.write(prev)
    return f"Vráceno do původního stavu: {name}"


def _log_diff(name: str, mode: str, old: str, new: str):
    """Show what changed in the window (added lines, or a unified diff when a file was replaced)."""
    if mode == "append" or not old:
        shown = "\n".join("+ " + line for line in new.rstrip("\n").splitlines())
    else:
        shown = "\n".join(list(difflib.unified_diff(old.splitlines(), new.splitlines(), "before", "after",
                                                    lineterm="", n=2))[:80]) or "(no change)"
    log("output", f"┌ {name} ({mode})\n{shown}\n└")


def _run_py(real: str):
    """Run a Python file. Returns (exit code or None on timeout, output text)."""
    env = {k: v for k, v in os.environ.items() if not any(w in k.upper() for w in ("KEY", "TOKEN", "SECRET"))}
    try:
        r = subprocess.run([sys.executable, real], cwd=os.path.dirname(real), capture_output=True, text=True,
                           timeout=RUN_TIMEOUT, stdin=subprocess.DEVNULL, env=env)
    except subprocess.TimeoutExpired:
        return None, ""
    out = (r.stdout or "").rstrip()
    if (r.stderr or "").strip():
        out += ("\n" if out else "") + "[stderr]\n" + r.stderr.rstrip()
    return r.returncode, (out[-3000:] or "(no output)")


def _py_target(path: str):
    """Resolve the .py file for run_file / run_and_fix. Returns (path or None, error message or '')."""
    real = resolve_path(path) if path else (STATE["last_file"] or None)
    if not real:
        return None, ("Chyba: žádný soubor není vybrán." if not path
                      else f"Chyba: nenalezeno: {os.path.basename(path.rstrip('/')) or path}")
    if not os.path.isfile(real) or not allowed_path(real):
        return None, "Chyba: tento soubor nemohu číst."
    if not real.endswith(".py"):
        return None, "Chyba: spouštím jen soubory .py."
    return real, ""


def run_file(path: str = "") -> str:
    """Spustí Python soubor (výchozí: naposledy otevřený) a vrátí jeho výstup. Vždy se předem zeptá.

    Args:
        path: Název nebo cesta .py souboru; prázdné = naposledy otevřený soubor.
    """
    real, err = _py_target(path)
    if not real:
        return err
    name = os.path.basename(real)
    if not ask_confirm(f"Spustit {name}?"):
        return "Zrušeno."
    rc, out = _run_py(real)
    if rc is None:
        return f"Chyba: program běžel příliš dlouho (limit {RUN_TIMEOUT} s)."
    STATE["last_file"] = real
    log("output", f"┌ {name} (exit {rc})\n{out}\n└")
    return f"Spuštěno {name} (kód {rc}):\n{out}"


def _fix_with_llm(code: str, error: str, name: str) -> str:
    """Ask Gemini (or the big local model) for a corrected version of the whole file."""
    prompt = ("Fix this Python file so that it runs without errors and keeps its purpose. "
              "Return ONLY the complete corrected file content: no explanations, no markdown.\n\n"
              f"File: {name}\n--- CODE ---\n{code}\n--- ERROR ---\n{error}\n")
    if cloud_ready():
        from google.genai import types
        cfg = types.GenerateContentConfig(system_instruction="You are a precise Python debugging assistant.")
        r = _gemini_generate([types.Content(role="user", parts=[types.Part.from_text(text=prompt)])], cfg)
        cand = r.candidates[0] if r.candidates else None
        txt = "".join(p.text for p in (cand.content.parts if cand and cand.content else [])
                      if getattr(p, "text", None) and not getattr(p, "thought", False))
    else:
        model = SMART_MODEL if STATE["smart_ok"] else FAST_MODEL
        r = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}],
                        options=SMART_OPTIONS, keep_alive=SMART_KEEP_ALIVE)
        txt = r["message"]["content"]
    return _strip_fences(txt).strip("\n")


def run_and_fix(path: str = "", attempts: int = 3) -> str:
    """Spustí Python soubor a když selže, nechá AI kód opravit a zkusí to znovu (nejvýše 3x). Zeptá se jen jednou.

    Args:
        path: Název nebo cesta .py souboru; prázdné = naposledy otevřený soubor.
        attempts: Nejvýše kolikrát se má kód opravit (1-3).
    """
    real, err = _py_target(path)
    if not real:
        return err
    name = os.path.basename(real)
    attempts = max(1, min(3, int(attempts)))
    if not ask_confirm(f"Spustit {name} a až {attempts}x automaticky opravit chyby?"):
        return "Zrušeno."
    STATE["last_file"] = real
    rc, out = None, ""
    for i in range(attempts + 1):
        rc, out = _run_py(real)
        if rc is None:
            return f"Chyba: program běžel příliš dlouho (limit {RUN_TIMEOUT} s)."
        log("output", f"┌ {name} (run {i + 1}, exit {rc})\n{out}\n└")
        if rc == 0 or i == attempts:
            break
        log("step", f"🔧 Fix attempt {i + 1}/{attempts}")
        try:
            with open(real, encoding="utf-8") as f:
                old = f.read()
            new = _fix_with_llm(old, out, name)
        except Exception as e:
            return f"Chyba: oprava selhala ({str(e)[:80]})"
        if not new.strip() or new.strip() == old.strip():
            break                                      # nothing new to try
        _remember_for_undo(real)
        shutil.copy2(real, real + ".bak")
        with open(real, "w", encoding="utf-8") as f:
            f.write(new.rstrip("\n") + "\n")
        _log_diff(name, "replace", old, new)
    return f"Spuštěno {name} (kód {rc}):\n{out}"


def _strip_fences(code: str) -> str:
    m = re.match(r"^\s*```[\w+-]*\n(.*?)\n?```\s*$", code, re.S)
    return m.group(1) if m else code


def write_code(code: str, path: str = "", mode: str = "append") -> str:
    """Zapíše programový kód do souboru (výchozí: naposledy otevřený soubor).

    Args:
        code: Jen kód, bez vysvětlování a bez markdown bloků.
        path: Název nebo cesta souboru; prázdné = naposledy otevřený soubor.
        mode: 'append' (přidat na konec), 'replace' (přepsat celý soubor, vyžádá potvrzení) nebo 'create' (nový soubor na ploše).
    """
    if mode not in ("append", "replace", "create"):
        return f"Neplatný režim: {mode}"
    code = _strip_fences(code).rstrip("\n") + "\n"
    if len(code) > 100_000:
        return "Chyba: kód je příliš dlouhý."
    home = os.path.realpath(os.path.expanduser("~"))
    if path:
        target = resolve_path(path)
        if not target and mode == "create":
            target = os.path.join(home, "Desktop", os.path.basename(path.rstrip("/")))
        elif not target:
            return f"Chyba: nenalezeno: {os.path.basename(path.rstrip('/')) or path}"
    else:
        target = STATE["last_file"] or None
    if not target:
        return "Chyba: žádný soubor není vybrán."
    if not allowed_path(target):
        return "Chyba: sem nesmím zapisovat."
    if os.path.isdir(target):
        return "Chyba: to je složka, ne soubor."
    name = os.path.basename(target)
    exists = os.path.exists(target)
    if exists and mode in ("replace", "create"):
        if not ask_confirm(f"Soubor {name} už existuje. Přepsat?"):
            return "Zrušeno."
        shutil.copy2(target, target + ".bak")                 # safety copy
    _remember_for_undo(target)
    if mode == "append":
        sep = ""
        if exists and os.path.getsize(target) > 0:
            with open(target, "rb") as f:
                f.seek(-1, os.SEEK_END)
                sep = "\n" if f.read(1) == b"\n" else "\n\n"
        with open(target, "a", encoding="utf-8") as f:
            f.write(sep + code)
    else:
        with open(target, "w", encoding="utf-8") as f:
            f.write(code)
    prev = UNDO_STACK[-1][1] if UNDO_STACK and UNDO_STACK[-1][0] == target else None
    _log_diff(name, mode, prev.decode("utf-8", "replace") if prev else "", code)
    STATE["last_file"] = target
    return f"Kód přidán do {name}" if (mode == "append" and exists) else f"Kód zapsán do {name}"


VSCODE_CLI = "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"


def open_file_with(path: str, app_name: str, new_window: bool = False) -> str:
    """Otevře soubor nebo složku v konkrétní aplikaci (např. test.py ve Visual Studio Code).
    Zavolej to jednou; nevolej zvlášť open_application ani open_file.

    Args:
        path: Název souboru/složky nebo cesta (cesty si nevymýšlej).
        app_name: Název aplikace, např. 'Visual Studio Code', 'TextEdit', 'Safari'.
        new_window: True = otevřít v novém okně / nové instanci aplikace.
    """
    home = os.path.realpath(os.path.expanduser("~"))
    real = resolve_path(path)
    if not real or not (real == home or real.startswith(home + os.sep)):
        return f"Chyba: nenalezeno: {os.path.basename(path.rstrip('/')) or path}"
    if os.path.splitext(real)[1].lower() in BLOCKED_EXT:
        return "Chyba: tento typ souboru neotevírám."
    if os.path.isfile(real):
        STATE["last_file"] = real
    low = fold(app_name).replace(".app", "").strip()
    cli = VSCODE_CLI if os.path.exists(VSCODE_CLI) else shutil.which("code")
    if (low in ("visual studio code", "vs code", "vscode", "code")) and cli:
        cmd = [cli] + (["-n"] if new_window else []) + [real]     # `code -n` = new window
    else:
        cmd = ["open"] + (["-n"] if new_window else []) + ["-a", app_name, real]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return f"Chyba: nepodařilo se otevřít v {app_name}."
    return f"Otevřeno: {os.path.basename(real)} v {app_name}"


def open_result(number: int) -> str:
    """Otevře n-tý výsledek z posledního hledání souborů (např. 'otevři druhý').

    Args:
        number: Pořadí výsledku (1 = první).
    """
    try:
        n = int(number)
    except (TypeError, ValueError):
        return "Chyba: neplatné číslo."
    res = STATE["last_results"]
    if not res:
        return "Chyba: není z čeho vybírat."
    if not 1 <= n <= len(res):
        return f"Chyba: vyber číslo 1 až {len(res)}."
    return open_file(res[n - 1])


def get_selected_text() -> str:
    """Přečte text, který má uživatel právě označený v aktivní aplikaci (pro 'shrň / přelož, co jsem označil')."""
    old = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
    subprocess.run(["pbcopy"], input="", text=True)             # empty it so we can tell if anything was copied
    subprocess.run(["osascript", "-e", 'tell application "System Events" to keystroke "c" using command down'])
    time.sleep(0.25)
    sel = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
    subprocess.run(["pbcopy"], input=old, text=True)            # put the old clipboard text back
    return f"Označený text: {sel}" if sel.strip() else "Chyba: nic není označeno."


def get_frontmost_app() -> str:
    """Zjistí, která aplikace je právě v popředí (aktivní)."""
    r = subprocess.run(["osascript", "-e",
                        'tell application "System Events" to get name of first application process whose frontmost is true'],
                       capture_output=True, text=True)
    name = r.stdout.strip()
    return f"Aktivní aplikace: {name}" if name else "Chyba: nepodařilo se zjistit aktivní aplikaci."


def _shortcut_names() -> list:
    r = subprocess.run(["shortcuts", "list"], capture_output=True, text=True, timeout=15)
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def list_shortcuts() -> str:
    """Vypíše názvy zkratek z aplikace Zkratky (Shortcuts)."""
    try:
        names = _shortcut_names()
    except (OSError, subprocess.TimeoutExpired):
        return "Chyba: aplikace Zkratky není dostupná."
    return ("Zkratky: " + ", ".join(names)) if names else "Nemáš žádné zkratky."


def run_shortcut(name: str) -> str:
    """Spustí zkratku z aplikace Zkratky (Shortcuts) podle názvu. Vždy se předem zeptá.

    Args:
        name: Název zkratky.
    """
    try:
        names = _shortcut_names()
    except (OSError, subprocess.TimeoutExpired):
        return "Chyba: aplikace Zkratky není dostupná."
    q = name.strip().lower()
    match = next((n for n in names if n.lower() == q), None) or next((n for n in names if q and q in n.lower()), None)
    if not match:
        return f"Chyba: nenalezeno: {name}"
    if not ask_confirm(f"Spustit zkratku {match}?"):
        return "Zrušeno."
    try:
        r = subprocess.run(["shortcuts", "run", match], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "Chyba: zkratka běžela příliš dlouho."
    return f"Zkratka {match} spuštěna." if r.returncode == 0 else f"Chyba: zkratka selhala ({r.stderr.strip()[:80]})"


def get_calendar_events(days: int = 1) -> str:
    """Přečte události z Kalendáře na dnešek a následující dny (může chvíli trvat).

    Args:
        days: Na kolik dní dopředu (1 = jen dnes).
    """
    days = max(1, min(14, int(days)))
    script = f"""
set d1 to current date
set time of d1 to 0
set d2 to d1 + ({days} * days)
set out to ""
tell application "Calendar"
    repeat with c in calendars
        try
            set evs to (every event of c whose start date ≥ d1 and start date < d2)
            repeat with e in evs
                set out to out & (summary of e) & " | " & ((start date of e) as string) & linefeed
            end repeat
        end try
    end repeat
end tell
return out
"""
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=40)
    except subprocess.TimeoutExpired:
        return "Chyba: Kalendář neodpověděl včas."
    if r.returncode != 0:
        return f"Chyba: {r.stderr.strip()[:100]}"
    out = r.stdout.strip()
    return ("Události:\n" + out) if out else "Žádné události."


def remember(note: str) -> str:
    """Trvale si zapamatuje poznámku o uživateli (jméno, oblíbené věci, preference)."""
    with _mem_lock:
        notes = load_memory()
        if note not in notes:
            notes.append(note[:200])
        save_memory(notes[-50:])
    return "Zapamatováno."


def forget(keyword: str) -> str:
    """Smaže z paměti poznámky obsahující dané slovo."""
    with _mem_lock:
        notes = load_memory()
        kept = [n for n in notes if keyword.lower() not in n.lower()]
        save_memory(kept)
    return f"Smazáno poznámek: {len(notes) - len(kept)}."


AVAILABLE_TOOLS = {f.__name__: f for f in [
    open_application, close_active_window, create_folder, write_to_file, type_text,
    get_clipboard_content, set_clipboard, open_website, search_google, web_search,
    control_spotify, set_volume, set_timer, list_timers, add_reminder, add_calendar_event,
    get_weather, get_battery, set_dark_mode, change_brightness, wifi,
    find_files, open_file, open_file_with, read_file, write_code, run_file, run_and_fix, undo_last_write, open_result, get_selected_text,
    get_frontmost_app, list_shortcuts, run_shortcut, get_calendar_events, remember, forget,
]}
if not ENABLE_WEB_ANSWERS:
    AVAILABLE_TOOLS.pop("web_search", None)
# Side-effect tools only run when the user's own words mention the topic (stops hallucinated calls,
# e.g. pausing Spotify when you only asked to open a file).
TOOL_GUARDS = {
    "control_spotify": r"\b(spotify|music|song|songs|track|tracks|playlist|pause|play|playing|skip|next|previous|resume)\b|hudb|pisn|skladb|pust|zastav|preskoc|dalsi|predchoz",
    "set_volume": r"\b(volume|louder|quieter|mute|unmute|loud|quiet|turn|lower|raise|increase|decrease|reduce|up|down)\b|hlasit|ztlum|zesil|zeslab",
    "wifi": r"wi-?fi|wireless|network|wlan",
    "set_dark_mode": r"dark|light mode|appearance|theme|tmav|svetl|rezim|vzhled",
    "change_brightness": r"bright|\bdim\b|\bjas\b|jasu|zesvetl|ztmav|\b(up|down|lower|raise|increase|decrease|reduce)\b",
    "close_active_window": r"close|quit|exit|zavri|zavrit",
    "type_text": r"type|typing|write|dictate|paste|napis|\bpis|vloz",
    "write_to_file": r"write|save|create|make|file|note|napis|uloz|vytvor|soubor|poznam",
    "create_folder": r"folder|directory|create|make|slozk|vytvor|adresar",
    "set_timer": r"timer|remind|alarm|countdown|minute|second|hour|casovac|pripomen|minut|sekund|hodin|budik",
    "add_reminder": r"remind|reminder|task|todo|pripom|ukol",
    "add_calendar_event": r"calendar|event|meeting|appointment|schedule|kalendar|udalost|schuzk",
    "remember": r"remember|memori[sz]e|note that|zapamatuj|pamatuj",
    "forget": r"forget|zapomen|smaz",
    "set_clipboard": r"clipboard|copy|schran|zkopiruj|kopiruj",
    "run_shortcut": r"shortcut|zkratk",
    "run_and_fix": r"fix|debug|repair|until|work|auto|oprav|dokud|funguj|spust|run",
    "run_file": r"run|execute|launch|start|test|try|spust|pust|zkus|vyzkous",
    "undo_last_write": r"undo|revert|roll ?back|take (it )?back|vrat|zpet|zrus",
    "write_code": r"write|code|add|insert|put|append|create|implement|function|loop|script|fix|edit|change|replace|refactor|napis|pridej|vloz|uprav|oprav|vytvor|kod|funkc|cyklus|smycku|program",
}

tool_list = list(AVAILABLE_TOOLS.values())

# Which words in a sentence make a tool relevant. The small local model only sees the matching tools
# (fewer choices = faster and fewer wrong calls); after a failure or on the big/cloud model it sees all.
BASE_TOOLS = {"open_application", "open_website", "search_google", "open_file", "find_files"}
TOOL_KEYWORDS = {
    **TOOL_GUARDS,
    "get_clipboard_content": r"clipboard|paste|copied|schran|zkopirov",
    "web_search": r"\b(what|who|when|where|how|why|which|news|latest|price|score|result|many|much)\b|kdo|kdy|kde|jak|proc|zpravy|cena|vysledek|kolik|co je",
    "list_timers": r"timer|casovac|alarm",
    "get_weather": r"weather|temperature|rain|forecast|snow|pocasi|teplot|prsi|predpoved",
    "get_battery": r"batter|charg|bateri|nabit",
    "open_file_with": r"\bwith\b|\busing\b|\bin\b|visual|code|textedit|chrome|safari|pomoci|\bve?\b",
    "read_file": r"read|show|look|see|what|content|explain|summar|fix|check|precti|ukaz|obsah|vysvetli|shrn|oprav",
    "get_selected_text": r"select|highlight|this text|that text|vybran|oznacen|tenhle text",
    "get_frontmost_app": r"front|active app|which app|current app|jaka aplikace|aktivni aplikace",
    "list_shortcuts": r"shortcut|zkratk",
    "get_calendar_events": r"calendar|agenda|schedule|meeting|event|today|tomorrow|kalendar|schuzk|udalost|dnes|zitra|program",
    "open_result": r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|number|prvni|druhy|treti|ctvrty|paty|cislo)\b|\b\d\b",
}


def select_tools(folded_text: str) -> list:
    wanted = set(BASE_TOOLS)
    wanted |= {name for name, pat in TOOL_KEYWORDS.items() if re.search(pat, folded_text)}
    return [f for name, f in AVAILABLE_TOOLS.items() if name in wanted]


def current_tools(model: str) -> list:
    """Small local model: only the tools that match the sentence. Big/cloud model or after a failure: all."""
    if STATE["full_tools"] or (model == SMART_MODEL and model != FAST_MODEL):
        return tool_list
    return select_tools(STATE["turn_fold"])


# ==========================================
# SCREEN
# ==========================================
def capture_screen() -> str:
    path = os.path.join(tempfile.gettempdir(), "screenshot.png")
    subprocess.run(["screencapture", "-x", path])
    subprocess.run(["sips", "-Z", "1024", path], capture_output=True)  # downscale = faster
    return path


def describe_screen(question: str) -> str:
    img = capture_screen()
    r = ollama.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": question, "images": [img]}],
        keep_alive=KEEP_ALIVE,
        options={"num_predict": 200},
    )
    return r["message"]["content"]


# ==========================================
# FAST PATH (no AI needed)
# ==========================================
def extract_search_query(text: str):
    """If the user asks to search the internet, return exactly what to search for, else None."""
    t = text.strip()
    f = fold(t)                       # same length as t, so indexes match
    pre = r"^(?:(?:prosim|please|hej|hey|ok)[ ,]+)?"
    m = re.match(pre + r"(?:vyhledej|vyhledat|hledej|vygoogluj|googli|google|search|look up)"
                 r"\b[ ,]*(?:mi |me )?(?:the (?:web|internet) )?(?:for )?(?:na (?:internetu|webu|googlu) )?(.+)$", f)
    if not m:
        m = re.match(pre + r"najdi (?:mi )?na (?:internetu|webu|googlu) (.+)$", f)
    if not m:
        m = re.match(pre + r"(?:find|look for) (?:me )?(.+) (?:online|on (?:the )?(?:web|internet|google))$", f)
    if not m:
        return None
    q = t[m.start(1):m.end(1)].strip(" .,!?")
    q = re.sub(r"\s+(?:na (?:internetu|webu|googlu)|online|on (?:the )?(?:web|internet|google))\s*$", "",
               q, flags=re.I).strip(" .,!?")
    if not q or re.match(r"(soubor|souboru|soubory|file|files|slozk|dokument)", fold(q)):
        return None                   # "search file ..." = local search, not the web
    return q


_LOC_MAP = {"desktop": "Desktop", "plose": "Desktop", "documents": "Documents", "dokumentech": "Documents",
            "downloads": "Downloads", "stazenych": "Downloads", "stazene": "Downloads"}


def fast_open_item(text: str):
    """'open the folder X on desktop' / 'otevři složku X na ploše' -> open it directly."""
    t = text.strip(" .!?,")
    f = fold(t)
    m = re.match(r"^(?:please |prosim )?(?:open|otevri|spust)\s+(?:the |my )?"
                 r"(?:folder|directory|file|document|slozku|soubor|dokument)\s+(.+?)"
                 r"(?:\s+(?:located |stored |saved )?(?:on|in|from|at|na|v|ve)\s+(?:the |my )?"
                 r"(desktop|documents|downloads|plose|dokumentech|stazenych|stazene)(?: folder)?)?$", f)
    if not m:
        return None
    name = t[m.start(1):m.end(1)].strip(" '\"")
    if re.search(r"\b(inside|within|located|and|then|in|on|with|using)\b", fold(name)):
        return None                    # several clauses in one sentence: not a simple "open X"
    loc = _LOC_MAP.get(m.group(2) or "")
    target = f"~/{loc}/{name}" if loc else name
    return ("open_item", lambda: open_file(target))


_APP_ALIASES = {"visualstudiocode": "Visual Studio Code", "vscode": "Visual Studio Code", "textedit": "TextEdit",
                "safari": "Safari", "googlechrome": "Google Chrome", "chrome": "Google Chrome",
                "sublimetext": "Sublime Text", "pycharm": "PyCharm", "xcode": "Xcode", "preview": "Preview"}
_APP_RE = (r"\b(?:in|with|using|ve|v|pomoci)\s+(visual studio code|vs ?code|text ?edit|safari|google chrome|"
           r"chrome|sublime text|pycharm|xcode|preview)\b")


def fast_open_in_app(text: str):
    """'open test.py inside tes123 folder and open it in Visual Studio Code in a new window' -> done directly."""
    t = text.strip(" .!?,")
    f = fold(t)
    if not re.match(r"^(?:please |prosim )?(?:open|otevri|spust)\b", f):
        return None
    am = re.search(_APP_RE, f)
    head = f[:am.start()] if am else f
    nm = re.search(r"([\w\-]+\.[a-z0-9]{1,6})\b", head)       # a file name with an extension
    if not nm:
        return None
    name = t[nm.start(1):nm.end(1)]
    if re.search(r"\.(com|cz|org|net|io|eu|co|uk)$", name.lower()):
        return None                                          # that is a website, not a file
    after = head[nm.end():]
    fm = re.search(r"\b(?:inside|within|in|from)\s+(?:the |my )?([\w.\-]+)", after)
    if not am and not fm:
        return None
    folder = None
    if fm and fm.group(1) not in _LOC_MAP and fm.group(1) not in ("the", "my", "a"):
        folder = t[nm.end() + fm.start(1): nm.end() + fm.end(1)]
    lm = re.search(r"\b(desktop|documents|downloads|plose|dokumentech|stazenych)\b", after)
    rel = f"{folder}/{name}" if folder else name
    path = f"~/{_LOC_MAP[lm.group(1)]}/{rel}" if lm else rel
    if am:
        app = _APP_ALIASES[am.group(1).replace(" ", "")]
        new_window = bool(re.search(r"new (?:window|instance)|novem okne|nove instanci", f))
        return ("open_in_app", lambda: open_file_with(path, app, new_window))
    return ("open_item", lambda: open_file(path))


def fast_run_fix(text: str):
    """'run it and fix it' – kept whole so it is not split at "and"."""
    tt = fold(text).strip(" .!?,")
    if re.match(r"^(?:please )?(?:run|execute)(?: it| this| that| the (?:file|script|code|program))?(?: and| then|,)? "
                r"(?:fix|debug|repair)(?: it| them| the errors?| errors?| the bugs?| bugs?| any errors?)?$", tt):
        return ("run_fix", lambda: run_and_fix("").splitlines()[0])
    return None


_ORD = {"first": 1, "1st": 1, "prvni": 1, "second": 2, "2nd": 2, "druhy": 2, "third": 3, "3rd": 3, "treti": 3,
        "fourth": 4, "4th": 4, "ctvrty": 4, "fifth": 5, "5th": 5, "paty": 5}


def fast_plan(text: str):
    """Recognise ONE simple command without running it. Returns (label, callable) or None.
    The callable returns the reply text, or None meaning "let the AI handle it"."""
    t = fold(text).strip()
    tt = t.strip(" .!?,")
    m = re.search(r"(hlasitost|volume).*?(\d{1,3})", t)
    if m:
        level = int(m.group(2))
        return "volume", lambda: set_volume(level)
    if re.search(r"(dalsi|next).*(pisn|skladb|song|track)|preskoc|\bskip\b", t):
        return "spotify:next", lambda: control_spotify("next track")
    if re.search(r"(predchozi|previous).*(pisn|skladb|song|track)", t):
        return "spotify:previous", lambda: control_spotify("previous track")
    if re.search(r"\b(pauza|pause|zastav hudbu|pust hudbu|play music)\b", t):
        return "spotify:pause", lambda: control_spotify("playpause")
    if re.search(r"zavri (aktivni )?okno|close (the )?window", t):
        return "close_window", close_active_window
    if re.search(r"\b(baterie|baterii|battery)\b", t):
        return "battery", get_battery
    if (re.match(r"^(?:please )?(?:run|execute)(?: it| this| that| the (?:file|script|code|program))?$", tt)
            or re.match(r"^(?:spust|spustit|pust) (?:to|ho|skript|program|soubor|kod)$", tt)):
        return "run", lambda: run_file("").splitlines()[0]      # the output itself is printed in the window
    if (re.match(r"^(?:please )?(?:undo|revert)(?: it| that| this| the last (?:change|edit))?$", tt)
            or tt in ("vrat to", "vrat to zpet", "zpet")):
        return "undo", undo_last_write
    m = re.search(r"(casovac|timer)\D{0,15}?(\d+)\s*(sek\w*|sec\w*|s\b|min\w*|hod\w*|hour\w*)", t)
    if m:
        unit = m.group(3)
        factor = 1 / 60 if unit.startswith("s") else 60 if unit.startswith("h") else 1
        minutes = int(m.group(2)) * factor
        msg = "Timer finished" if STATE["lang"] == "en" else "Časovač skončil"
        return "timer", lambda: set_timer(minutes, msg)
    if STATE["last_results"]:                                  # "open the second one" after a file search
        m = re.match(r"^(?:please )?(?:open|otevri)(?: the)?(?: result)? (first|second|third|fourth|fifth|1st|2nd|3rd|"
                     r"4th|5th|prvni|druhy|treti|ctvrty|paty|number \d+|cislo \d+|\d+)(?: one| result)?$", tt)
        if m:
            word = m.group(1)
            n = _ORD.get(word) or int(re.search(r"\d+", word).group())
            return "open_result", lambda: open_result(n)
    for opener in (fast_open_in_app, fast_open_item):
        plan = opener(text)
        if plan:
            return plan
    lt = text.lower().strip()
    m = re.match(r"(otevři|otevri|spusť|spust|open)\s+(aplikaci\s+|app\s+)?([\w .+-]{2,40})$", lt)
    if m and not re.search(r"web|stránk|strank|http|\.(com|cz|org)", lt):
        name = m.group(3).strip()
        if len(name.split()) <= 4 and not re.match(r"(it|this|that|the|my|a|an)\b", fold(name)):
            def _open(app=name.title()):
                res = open_application(app)
                return None if "nenalezena" in res else res
            return "open_app", _open
    return None


CHAIN_SPLIT_RE = re.compile(r"\s*(?:,\s*)?\b(?:and then|and also|after that|then|and|a pak|a potom|potom|a take|a taky)\b\s*", re.I)


def plan_fast(text: str):
    """Decide what the instant paths would do WITHOUT running anything.
    Returns ([(label, callable, part), ...], leftover text for the AI or None)."""
    for opener in (fast_run_fix, fast_open_in_app, fast_open_item):    # structured sentences are never split
        plan = opener(text)
        if plan:
            return [(plan[0], plan[1], text)], None
    parts = [x.strip(" ,.;!?") for x in CHAIN_SPLIT_RE.split(text)]
    parts = [x for x in parts if x]
    if len(parts) >= 2:
        plans, i = [], 0
        while i < len(parts):
            plan = fast_plan(parts[i])
            if not plan:
                break
            plans.append((plan[0], plan[1], parts[i]))
            i += 1
        # first part not an instant command -> the whole sentence goes to the AI
        return (plans, " and ".join(parts[i:]) if i < len(parts) else None) if plans else ([], None)
    plan = fast_plan(text)
    return ([(plan[0], plan[1], text)], None) if plan else ([], None)


def run_fast(text: str):
    """Run the instant (no-AI) paths. Returns (reply or None, leftover text for the AI or None)."""
    plans, rest = plan_fast(text)
    if not plans:
        return None, None
    results = []
    for label, fn, part in plans:
        res = fn()
        if res is None:
            if len(plans) == 1 and rest is None:
                return None, None                  # e.g. app not found: let the AI try
            res = f"Chyba: nenalezeno: {part}"
        log("step", f"⚡ {label}" + (f": {part}" if len(plans) > 1 else " (no AI needed)"))
        results.append(localize(res))
    return " ".join(results), rest


# ==========================================
# SPEECH-TO-TEXT
# ==========================================
_google = sr.Recognizer()
_whisper = None
_whisper_lock = threading.Lock()


def load_whisper():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            log("info", f"⏳ Loading Whisper '{WHISPER_MODEL}' (first run downloads the model)...")
            STATE["phase"] = "loading"
            try:
                _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            finally:
                STATE["phase"] = ""
            log("info", "✓ Whisper ready (offline speech recognition)")
    return _whisper


NOISE_EXACT = {"thank you", "thanks", "you", "bye", "thanks for watching", "dekuji"}
NOISE_PHRASES = ("spotify, safari, visual studio code", "titulky", "dekuji za pozornost", "thanks for watching", "thank you for watching",
                 "podekovani", "amara.org")


def _to_samples(audio):
    import numpy as np
    raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(audio) -> str:
    if STATE["offline"] and STATE["stt"] == "google":
        STATE["stt"] = "whisper"                    # offline mode: audio must not leave this Mac
    if VOICE_DEBUG:
        dur = len(audio.frame_data) / float(audio.sample_rate * audio.sample_width)
        log("info", f"🎙 Captured {dur:.1f} s of audio – transcribing ({STATE['stt']})...")
    STATE["phase"] = "transcribing"
    t0 = time.time()
    auto = STATE["lang_mode"] == "auto"
    try:
        if STATE["stt"] == "mlx":
            try:
                import mlx_whisper
                samples = _to_samples(audio)

                def _run_mlx(language):
                    return mlx_whisper.transcribe(samples, path_or_hf_repo=MLX_MODEL, language=language,
                                                  initial_prompt=WHISPER_HINT)
                res = _run_mlx(None if auto else STATE["lang"])
                if auto:
                    det = res.get("language")
                    if det in ("en", "cs"):
                        _set_lang(det)
                    else:
                        res = _run_mlx(STATE["lang"])
                return (res.get("text") or "").strip()
            except ImportError:
                log("error", "mlx-whisper missing (pip install mlx-whisper) – using faster-whisper")
                STATE["stt"] = "whisper"
        if STATE["stt"] == "whisper":
            try:
                model = load_whisper()
                STATE["phase"] = "transcribing"
                samples = _to_samples(audio)

                def _run(language):
                    segs, info = model.transcribe(
                        samples, language=language, beam_size=1, initial_prompt=WHISPER_HINT,
                        vad_filter=True,
                        vad_parameters=dict(threshold=0.3, min_silence_duration_ms=1200, speech_pad_ms=500),
                        log_prob_threshold=-2.0, condition_on_previous_text=False)
                    return " ".join(seg.text.strip() for seg in segs).strip(), info

                text, info = _run(None if auto else STATE["lang"])
                if auto:
                    det = getattr(info, "language", None)
                    if det in ("en", "cs") and getattr(info, "language_probability", 0) >= 0.5:
                        _set_lang(det)
                    else:                           # unsure (short phrase / other language): redo in the current one
                        text, _ = _run(STATE["lang"])
                return text
            except ImportError:
                log("error", "faster-whisper / numpy missing (pip install faster-whisper numpy) – using Google STT")
                STATE["stt"] = "google"
        if STATE["offline"]:
            raise RuntimeError("offline mode: install faster-whisper for speech recognition")
        return _google.recognize_google(audio, language=STT_LANGS[STATE["lang"]])
    finally:
        STATE["phase"] = ""
        STATE["last_stt_s"] = time.time() - t0


# ==========================================
# COMMAND PROCESSING (worker thread)
# ==========================================
def trim_history():
    if len(messages) > MAX_HISTORY + 1:
        messages[:] = [messages[0]] + messages[-MAX_HISTORY:]
    while len(messages) > 1 and messages[1].get("role") == "tool":
        messages.pop(1)


def fmt_args(args) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in dict(args).items())


FORCE_SMART_RE = re.compile(r"^(?:premyslej|think|podrobne|detailne)\b[ :,]*")
COMPLEX_RE = re.compile(r"\b(?:napi\w*|naps\w*|sepi\w*|shrn\w*|vysvetli\w*|preloz\w*|analyzuj\w*|porovnej\w*|"
                        r"navrhni\w*|vymysli\w*|naplanuj\w*|write|summari[sz]e|explain|translate|compare|draft)\b")
MULTI_RE = re.compile(r"\b(?:a pak|a potom|potom|pote|nasledne|a take|a taky|a nakonec|and then|then|after that)\b")


def needs_smart_model(text: str) -> bool:
    """Cheap heuristic: long request, several steps, or writing / explaining / translating."""
    f = fold(text)
    return (len(text) > 120 or len(text.split()) > 18
            or bool(MULTI_RE.search(f)) or bool(COMPLEX_RE.search(f)))


CLOUD = "gemini-cloud"      # routing marker (not a real model name)
CODE_RE = re.compile(r"\b(?:code|coding|script|function|loop|variable|algorithm|programming|python|javascript|"
                     r"typescript|html|css|sql|bash|regex|bug|debug|refactor|implement|snippet|syntax|json)\b"
                     r"|\bkod|skript|funkc|cyklus|smycku")
FORCE_CLOUD_RE = re.compile(r"^(?:ask\s+)?(?:gemini|google ai|the cloud|cloud)\b[ :,]*")
_gem = {"client": None, "idx": 0, "history": [], "err": "", "model": ""}


def is_coding(text: str) -> bool:
    return bool(CODE_RE.search(fold(text)))


def _gemini_client():
    if _gem["client"] is not None:
        return _gem["client"]
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        _gem["err"] = "no GEMINI_API_KEY found (put GEMINI_API_KEY=... in a .env file next to this script)"
        return None
    try:
        from google import genai
    except ImportError:
        _gem["err"] = "package missing – run: pip install google-genai python-dotenv"
        return None
    _gem["client"] = genai.Client(api_key=key)
    return _gem["client"]


def cloud_ready() -> bool:
    return CLOUD_ENABLED and STATE["cloud_ok"] and not STATE["offline"] and _gemini_client() is not None


def select_model(text: str, force: bool, force_cloud: bool = False):
    """Returns (model, reason). `model` is CLOUD when Gemini should handle the turn."""
    mode = STATE["model_mode"]
    explicit = force_cloud or mode == "cloud"
    if explicit or (mode == "auto" and CLOUD_AUTO and is_coding(text)):
        if cloud_ready():
            return CLOUD, ("requested" if explicit else "coding task")
        log("info", f"☁ Gemini unavailable ({_gem['err'] or 'disabled'}) – using the local model")
        force = True                      # hard task: at least use the bigger local model
    if not STATE["smart_ok"] or SMART_MODEL == FAST_MODEL:
        return FAST_MODEL, ""
    if mode == "smart" or force:
        return SMART_MODEL, "requested"
    if mode == "auto" and AUTO_ESCALATE and needs_smart_model(text):
        return SMART_MODEL, "complex task"
    return FAST_MODEL, ""


def _feed_tts(buf: str, piece: str) -> str:
    """Collect streamed text and hand every finished sentence to the speaker right away."""
    buf += piece
    if not (STREAM_TTS and SPEAK_REPLIES) or buf.count("```") % 2 == 1:
        return buf
    while True:
        m = re.search(r"[.!?…]\s", buf)
        if not m:
            return buf
        sentence, buf = buf[:m.end()], buf[m.end():]
        clean = speakable(sentence)
        if clean:
            tts_q.put(clean)
            STATE["spoken_stream"] = True


def start_hotkey():
    """Global push-to-talk hotkey that works from any app (optional: pip install pynput)."""
    try:
        from pynput import keyboard
    except ImportError:
        log("info", f"⌨ Global hotkey off – run `pip install pynput` to enable {HOTKEY}")
        return

    def trigger():
        STATE["talk_deadline"] = time.time() + 12
        talk_once.set()

    try:
        keyboard.GlobalHotKeys({HOTKEY: trigger}).start()
        log("info", f"⌨ Global push-to-talk hotkey: {HOTKEY} (macOS may ask for Input Monitoring permission)")
    except Exception as e:
        log("info", f"⌨ Hotkey unavailable: {e}")


def set_autostart(on: bool) -> str:
    plist = os.path.expanduser("~/Library/LaunchAgents/com.llama.assistant.plist")
    if on:
        script = os.path.abspath(sys.argv[0])
        data = {"Label": "com.llama.assistant", "ProgramArguments": [sys.executable, script], "RunAtLoad": True,
                "WorkingDirectory": os.path.dirname(script),
                "StandardErrorPath": os.path.join(APP_DIR, "autostart.log"),
                "StandardOutPath": os.path.join(APP_DIR, "autostart.log")}
        os.makedirs(os.path.dirname(plist), exist_ok=True)
        with open(plist, "wb") as f:
            plistlib.dump(data, f)
        return "Autostart on – the assistant will open the next time you log in."
    if os.path.exists(plist):
        os.remove(plist)
        return "Autostart off (effective from the next login)."
    return "Autostart was not enabled."


def tts_worker():
    while not stop.is_set():
        try:
            sentence = tts_q.get(timeout=0.3)
        except queue.Empty:
            continue
        try:
            if sentence.strip():
                speak(sentence)
        finally:
            tts_q.task_done()


def chat_round(model: str):
    """One AI call. Streams the answer into the window. Returns (content, tool_calls, streamed)."""
    is_smart = model == SMART_MODEL and model != FAST_MODEL
    kwargs = dict(model=model, messages=messages, tools=current_tools(model),
                  keep_alive=SMART_KEEP_ALIVE if is_smart else KEEP_ALIVE,
                  options=SMART_OPTIONS if is_smart else OLLAMA_OPTIONS)
    if is_smart and not STATE["smart_loaded"]:
        log("info", f"⏳ First use of {model}: loading it into memory may take a few seconds...")
    if STATE["stream"]:
        started, content, calls, tts_buf = False, "", [], ""
        try:
            for chunk in ollama.chat(stream=True, **kwargs):
                m = chunk["message"]
                piece = m.get("content") or ""
                if piece:
                    if not started:
                        ui_q.put(("stream_start", "🤖 "))
                        started = True
                    content += piece
                    ui_q.put(("stream", piece))
                    tts_buf = _feed_tts(tts_buf, piece)
                if m.get("tool_calls"):
                    calls.extend(m["tool_calls"])
        except Exception as e:
            if started:
                ui_q.put(("stream_end", ""))
                raise
            if "not found" in str(e).lower():
                raise                      # missing model: let the caller handle it
            STATE["stream"] = False
            log("info", f"(streaming unavailable: {e}; using normal mode)")
        else:
            if started:
                ui_q.put(("stream_end", ""))
            if STREAM_TTS and SPEAK_REPLIES and tts_buf.strip():
                tts_q.put(speakable(tts_buf))
                STATE["spoken_stream"] = True
            if is_smart:
                STATE["smart_loaded"] = True
            return content, calls, started
    r = ollama.chat(**kwargs)
    if is_smart:
        STATE["smart_loaded"] = True
    m = r["message"]
    return m.get("content") or "", list(m.get("tool_calls") or []), False


def run_llm_loop(model: str):
    """Run the tool-calling loop with `model`.
    Returns (reply, streamed, steps, failed, last_model). `failed` is only set when nothing
    was actually executed, so retrying with the bigger model cannot repeat any action."""
    steps, reply, streamed, failed = [], "", False, False
    cur = model

    for _ in range(MAX_TOOL_ROUNDS):
        content, calls, streamed_now = chat_round(cur)
        streamed = streamed or streamed_now
        msg = {"role": "assistant", "content": content}
        if calls:
            msg["tool_calls"] = calls
        messages.append(msg)

        if not calls:
            reply = content
            if not content.strip() and not steps:
                failed = True              # model produced nothing
            break

        round_steps = []
        round_raw = []
        for call in calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            guard = TOOL_GUARDS.get(name)
            if guard and not re.search(guard, STATE["turn_fold"]):
                log("info", f"🛡 Skipped {name} – you didn't ask for that")
                messages.append({"role": "tool", "tool_name": name,
                                 "content": "Skipped: the user did not ask for this action."})
                continue
            log("step", f"⚙ {name}({fmt_args(args)})")
            func = AVAILABLE_TOOLS.get(name)
            try:
                result = func(**args) if func else f"Nástroj {name} neexistuje."
            except Exception as e:
                result = f"Chyba nástroje: {e}"
            raw = str(result)
            round_raw.append(raw)
            steps.append((name, args, localize(raw)))
            round_steps.append(steps[-1])
            messages.append({"role": "tool", "tool_name": name, "content": localize(raw)})

        if not round_steps:
            continue                       # every call was skipped by the intent guard
        if len(steps) == len(round_steps) and all(
                r.startswith(("Chyba nástroje", "Nástroj")) for r in round_raw):
            failed = True                  # unknown tool / bad arguments: nothing ran
            break

        ok = all(not r.startswith(("Chyba", "Neplatn", "Nástroj")) for r in round_raw)
        needs_reading = any(n in NEEDS_LLM_AFTER for n, _, _ in round_steps)
        if FAST_REPLY and ok and not needs_reading:
            reply = " ".join(r for _, _, r in round_steps)
            messages.append({"role": "assistant", "content": reply})
            streamed = False
            break
        if (any(n in SMART_SYNTHESIS_TOOLS for n, _, _ in round_steps)
                and SMART_FOR_SYNTHESIS and cur != SMART_MODEL and SMART_MODEL != FAST_MODEL
                and STATE["smart_ok"] and STATE["model_mode"] != "fast"):
            cur = SMART_MODEL
            log("info", f"🧠 Switching to {SMART_MODEL} to work with the results")

    return reply, streamed, steps, failed, cur


def run_model(model: str, base_len: int):
    """One attempt with `model`. If the big model is not installed, falls back to the fast one."""
    messages[0]["content"] = build_system_prompt(smart=(model == SMART_MODEL and model != FAST_MODEL))
    try:
        return run_llm_loop(model)
    except Exception as e:
        if model != FAST_MODEL and "not found" in str(e).lower():
            STATE["smart_ok"] = False
            log("error", f"Model {model} is not installed – run:  ollama pull {model}   (using {FAST_MODEL} for now)")
            del messages[base_len:]
            messages[0]["content"] = build_system_prompt(smart=False)
            return run_llm_loop(FAST_MODEL)
        raise


def exec_tool(name: str, args) -> tuple:
    """Run one tool with the intent guard. Returns (raw, localized), or (None, None) if skipped."""
    guard = TOOL_GUARDS.get(name)
    if guard and not re.search(guard, STATE["turn_fold"]):
        log("info", f"🛡 Skipped {name} – you didn't ask for that")
        return None, None
    log("step", f"⚙ {name}({fmt_args(args)})")
    func = AVAILABLE_TOOLS.get(name)
    try:
        result = func(**args) if func else f"Nástroj {name} neexistuje."
    except Exception as e:
        result = f"Chyba nástroje: {e}"
    raw = str(result)
    return raw, localize(raw)


def _gemini_generate(contents, cfg):
    """generate_content with automatic fallback to the next Gemini model on quota / availability errors."""
    client = _gemini_client()
    last = None
    for _ in range(len(GEMINI_MODELS)):
        model = GEMINI_MODELS[_gem["idx"]]
        _gem["model"] = model
        try:
            return client.models.generate_content(model=model, contents=contents, config=cfg)
        except Exception as e:
            last = e
            if any(k in str(e) for k in ("503", "404", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "NOT_FOUND", "Quota")):
                log("info", f"⚠ {model} unavailable – trying the next Gemini model")
                _gem["idx"] = (_gem["idx"] + 1) % len(GEMINI_MODELS)
                time.sleep(0.5)
                continue
            raise
    raise last


def run_cloud_loop(user_content: str, image_path: str = None):
    """Gemini handles the turn with the same tools. Returns (reply, streamed, steps, failed, used)."""
    from google.genai import types
    en = STATE["lang"] == "en"
    addendum = (
        "\nYou are the cloud brain of this assistant and can write code. When asked to write or change code, put ONLY "
        "the code into a file with write_code (default: the last opened file; call read_file first if you need to see "
        "the existing code). Never read code aloud. After acting, answer in one short sentence."
        if en else
        "\nJsi cloudový mozek asistenta a umíš psát kód. Když tě uživatel požádá o napsání nebo úpravu kódu, vlož "
        "POUZE kód do souboru přes write_code (výchozí: naposledy otevřený soubor; potřebuješ-li vidět stávající kód, "
        "nejdřív zavolej read_file). Kód nikdy nečti nahlas. Po akci odpověz jednou krátkou větou."
    )
    cfg = types.GenerateContentConfig(
        tools=tool_list,
        system_instruction=build_system_prompt(smart=False) + addendum,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    first = [types.Part.from_text(text=user_content)]
    if image_path:
        try:
            with open(image_path, "rb") as f:
                first.append(types.Part.from_bytes(data=f.read(), mime_type="image/png"))
        except OSError:
            pass
    contents = list(_gem["history"]) + [types.Content(role="user", parts=first)]
    steps, reply, failed = [], "", False

    for _ in range(MAX_TOOL_ROUNDS):
        response = _gemini_generate(contents, cfg)
        cand = response.candidates[0] if response.candidates else None
        if cand is None or cand.content is None or not cand.content.parts:
            failed = True
            break
        contents.append(cand.content)
        parts = cand.content.parts
        calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        if not calls:
            reply = text.strip()
            break
        answers = []
        for fc in calls:
            args = dict(fc.args or {})
            raw, shown = exec_tool(fc.name, args)
            if raw is None:
                shown = "Skipped: the user did not ask for this action."
            else:
                steps.append((fc.name, args, shown))
            answers.append(types.Part.from_function_response(name=fc.name, response={"result": shown}))
        contents.append(types.Content(role="user", parts=answers))

    hist = contents[-30:]
    while hist and not (hist[0].role == "user" and any(getattr(p, "text", None) for p in (hist[0].parts or []))):
        hist.pop(0)
    for c in hist:                                   # don't carry screenshots along in the history
        if c.role == "user" and c.parts:
            c.parts = [p for p in c.parts if not getattr(p, "inline_data", None)]
    _gem["history"] = hist

    if not reply and steps:
        reply = steps[-1][2].splitlines()[0]
    return reply, False, steps, failed, f"☁ {_gem['model']}"


def speakable(text: str) -> str:
    """Never read code aloud."""
    note = "(the code is shown in the window)" if STATE["lang"] == "en" else "(kód je v okně)"
    text = re.sub(r"```.*?```", f" {note} ", text, flags=re.S)
    return re.sub(r"[`*#]+", "", text).strip()


def process(source: str, text: str):
    icon = "🎤" if source == "voice" else "›"
    log("user", f"{icon} {text}")
    t_start = time.time()

    # "přemýšlej: …" / "think: …" forces the big model
    force_smart = force_cloud = False
    cm = FORCE_CLOUD_RE.match(fold(text))            # "gemini: ..." forces the cloud model
    if cm and cm.end() < len(text.strip()):
        text = text[cm.end():].strip()
        force_cloud = True
    pm = FORCE_SMART_RE.match(fold(text))            # "think: ..." forces the big local model
    if pm and pm.end() < len(text.strip()):
        text = text[pm.end():].strip()
        force_smart = True

    STATE["full_tools"] = False
    STATE["spoken_stream"] = False
    if STATE["lang_mode"] == "auto" and (source == "text" or STATE["stt"] == "google"):
        g = guess_lang(text)
        if g:
            _set_lang(g)
    STATE["turn_fold"] = fold(text)
    sq = extract_search_query(text)
    if sq:
        log("step", f"🔎 Searching the internet for: {sq}")
        res = localize(search_google(sq))
        log("ai", f"🤖 {res}")
        speak(res)
        return

    needs_screen = any(w in text.lower() for w in TRIGGER_WORDS)
    if not needs_screen:
        fast_reply, leftover = run_fast(text)
        if fast_reply:
            log("ai", f"🤖 {fast_reply}")
            speak(speakable(fast_reply.splitlines()[0])[:300])
            if not leftover:
                return
            log("info", f"↪ Continuing with: {leftover}")
            text = leftover                      # the rest of the sentence goes to the AI
            STATE["turn_fold"] = fold(text)

    user_content = text
    shot = None
    if needs_screen and CLOUD_VISION and STATE["model_mode"] != "fast" and cloud_ready():
        log("step", "📸 Taking a screenshot – Gemini will look at it (the image goes online)")
        shot = capture_screen()
        force_cloud = True
    elif needs_screen:
        log("step", "📸 Taking a screenshot and analysing it...")
        try:
            info = describe_screen(text)
            user_content = f"{text}\n\n[Popis obrazovky: {info}]"
            log("step", "✓ Screen analysed")
        except Exception as e:
            log("error", f"Screen analysis failed: {e}")
            log("info", f"Fix: update Ollama (`ollama --version`) and run `ollama pull {VISION_MODEL}`. "
                        "Also allow Screen Recording for your terminal / VS Code in System Settings → Privacy.")
            msg = "I couldn't look at the screen." if STATE["lang"] == "en" else "Nepodařilo se mi podívat na obrazovku."
            log("ai", f"🤖 {msg}")
            speak(msg)
            return

    messages.append({"role": "user", "content": user_content})
    base_len = len(messages)

    model, why = select_model(text, force_smart, force_cloud)
    result, cloud_tried = None, False
    if model == CLOUD:
        cloud_tried = True
        log("info", f"☁ Using Gemini ({GEMINI_MODELS[_gem['idx']]}) – {why}. This request goes online.")
        try:
            result = run_cloud_loop(user_content, shot)
            messages.append({"role": "assistant", "content": result[0] or "(done)"})
        except Exception as e:
            log("error", f"Gemini failed: {e} – using the local model instead")
            del messages[base_len:]
            model = SMART_MODEL if (STATE["smart_ok"] and SMART_MODEL != FAST_MODEL) else FAST_MODEL
    if result is None:
        if model == SMART_MODEL and model != FAST_MODEL:
            log("info", f"🧠 Using {SMART_MODEL} ({why})")
        else:
            log("info", "… thinking")
        result = run_model(model, base_len)
    reply, streamed, steps, failed, used = result

    if (failed and used == FAST_MODEL and AUTO_ESCALATE and STATE["smart_ok"]
            and SMART_MODEL != FAST_MODEL and STATE["model_mode"] != "fast"):
        log("info", f"↗ {FAST_MODEL} struggled – retrying with {SMART_MODEL}")
        STATE["full_tools"] = True               # give it every tool this time
        del messages[base_len:]
        reply, streamed, steps, failed, used = run_model(SMART_MODEL, base_len)

    if failed and not cloud_tried and CLOUD_AUTO and STATE["model_mode"] != "fast" and cloud_ready():
        log("info", "↗ local models struggled – asking Gemini")
        del messages[base_len:]
        try:
            reply, streamed, steps, failed, used = run_cloud_loop(user_content, shot)
            messages.append({"role": "assistant", "content": reply or "(done)"})
        except Exception as e:
            log("error", f"Gemini failed: {e}")

    STATE["last_model"] = used

    if steps:
        log("summary", "── Summary of steps ──")
        for i, (name, args, result) in enumerate(steps, 1):
            first_line = result.splitlines()[0] if result else ""
            log("summary", f" {i}. {name} → {first_line}")

    t_ready = time.time()
    if reply:
        if not streamed:
            log("ai", f"🤖 {reply}")
        tts_q.join()                                  # sentences spoken while streaming must finish first
        if not (streamed and STATE["spoken_stream"]):
            full = streamed or str(used).startswith("☁")
            spoken = reply if full or "\n" not in reply else reply.splitlines()[0]
            speak(speakable(spoken)[:500])
    elif not steps:
        log("info", "(no response)")
    parts = [f"stt {STATE['last_stt_s']:.1f}s"] if source == "voice" else []
    parts += [f"think {t_ready - t_start:.1f}s", f"speak {time.time() - t_ready:.1f}s"]
    STATE["timing"] = " · ".join(parts)

    trim_history()


def worker():
    while not stop.is_set():
        try:
            source, text = cmd_q.get(timeout=0.3)
        except queue.Empty:
            continue
        busy.set()
        t0 = time.time()
        try:
            STATE["last_model"] = ""
            STATE["timing"] = ""
            process(source, text)
            extra = f" · {STATE['last_model']}" if STATE["last_model"] else ""
            if STATE["timing"]:
                extra += f"   ({STATE['timing']})"
            log("info", f"⏱ {time.time() - t0:.1f} s{extra}")
        except Exception as e:
            log("error", f"Error: {e}")
            if any(k in str(e).lower() for k in ("connection refused", "failed to connect", "connecterror", "connection error")):
                log("info", "Is Ollama running? Open the Ollama app (llama icon in the menu bar) and try again.")
        finally:
            busy.clear()
            extend_active()


def warmup():
    try:
        ollama.chat(model=FAST_MODEL, messages=[{"role": "user", "content": "hi"}],
                    keep_alive=KEEP_ALIVE, options={"num_predict": 1})
        log("info", f"✓ Model {FAST_MODEL} loaded and ready")
    except Exception as e:
        log("error", f"Model warm-up failed: {e} (is Ollama running? `ollama pull {FAST_MODEL}`)")
    if STATE["stt"] == "whisper":
        try:
            load_whisper()
        except ImportError:
            log("error", "faster-whisper not installed (pip install faster-whisper numpy) – using Google STT")
            STATE["stt"] = "google"
        except Exception as e:
            log("error", f"Whisper failed to load: {e} – using Google STT")
            STATE["stt"] = "google"
    if CLOUD_ENABLED:
        if _gemini_client():
            log("info", f"☁ Gemini ready – used for coding and hard tasks ({GEMINI_MODELS[0]})")
        else:
            log("info", f"☁ Gemini off: {_gem['err']}")


# ==========================================
# VOICE INPUT (background thread)
# ==========================================
MODE_LABELS = {"always": "Always listening", "wake": "Wake word", "push": "Push-to-talk"}


def mode_hint() -> str:
    return {
        "always": "always listening (just talk). Switch with /mode wake|push.",
        "wake": f"say “{wake_phrase()}, …” first. Switch with /mode always|push.",
        "push": "press 🎤 Talk (or Ctrl+Space) and speak. Switch with /mode always|wake.",
    }[STATE["mode"]]


def strip_wake(text: str):
    """Returns (wake word found, remaining command in original spelling)."""
    m = WAKE_RE.search(fold(text))
    if not m:
        return False, ""
    return True, text[m.end():].strip()


audio_q = queue.Queue()  # (AudioData, forced) recorded phrases waiting to be transcribed


def listening_paused() -> bool:
    """Should the microphone be idle right now?"""
    if talk_once.is_set() and time.time() > STATE["talk_deadline"]:
        talk_once.clear()
    forced = talk_once.is_set()
    if speaking.is_set():
        return True                          # never record the assistant's own voice
    if confirm_pending.is_set():
        return False                         # always listen for a yes / no
    if not voice_on.is_set() and not forced:
        return True
    if STATE["mode"] == "push" and not forced:
        return True
    if busy.is_set() or not cmd_q.empty():
        return True
    return False


def voice_loop():
    """Capture thread: records phrases and hands them over for transcription immediately,
    so nothing you say is lost while an earlier phrase is still being transcribed."""
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_ratio = ENERGY_RATIO
    recognizer.pause_threshold = PAUSE_SECONDS
    recognizer.non_speaking_duration = min(0.8, PAUSE_SECONDS)   # audio kept before/after speech
    try:
        mic = sr.Microphone(device_index=STATE["mic_index"])
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
    except Exception as e:
        log("error", f"Microphone unavailable: {e} (text input still works)")
        return

    log("info", f"🎤 Voice input ready – {mode_hint()}")
    if VOICE_DEBUG:
        log("info", f"   (mic trigger level: {int(recognizer.energy_threshold)}, "
                    f"sentence ends after {PAUSE_SECONDS} s of silence)")
    threading.Thread(target=transcribe_loop, daemon=True).start()

    while not stop.is_set():
        if STATE["mic_changed"]:                     # /mic <number>
            STATE["mic_changed"] = False
            try:
                mic = sr.Microphone(device_index=STATE["mic_index"])
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                log("info", "🎤 Microphone switched")
            except Exception as e:
                log("error", f"Couldn't use that microphone: {e}")
        if listening_paused():
            time.sleep(0.2)
            continue
        with mic as source:                  # stream stays open while we are actively listening
            while not stop.is_set() and not listening_paused() and not STATE["mic_changed"]:
                forced = talk_once.is_set()
                seq = STATE["speak_seq"]
                try:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=PHRASE_LIMIT)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    log("error", f"Microphone error: {e}")
                    time.sleep(1)
                    break
                if speaking.is_set() or STATE["speak_seq"] != seq:
                    continue                 # that was the assistant talking, not you
                if forced:
                    talk_once.clear()        # push-to-talk = one phrase
                audio_q.put((audio, forced))


def safe_transcribe(audio) -> str:
    try:
        return (transcribe(audio) or "").strip()
    except sr.UnknownValueError:
        if VOICE_DEBUG:
            log("info", "👂 (speech not understood)")
        return ""
    except Exception as e:
        log("error", f"Voice error: {e}")
        return ""


def spotify_playing() -> bool:
    """True while Spotify is playing (its lyrics would otherwise be heard as commands)."""
    try:
        r = subprocess.run(["osascript", "-e",
                            'if application "Spotify" is running then\n'
                            'tell application "Spotify" to return player state as string\n'
                            'else\nreturn "off"\nend if'],
                           capture_output=True, text=True, timeout=3)
        return r.stdout.strip().lower() == "playing"
    except Exception:
        return False


def transcribe_loop():
    """Transcription thread: turns recorded phrases into commands."""
    while not stop.is_set():
        try:
            audio, forced = audio_q.get(timeout=0.3)
        except queue.Empty:
            continue
        text = safe_transcribe(audio)
        # glue together phrases that were recorded while we were busy transcribing
        while True:
            try:
                more, _ = audio_q.get_nowait()
            except queue.Empty:
                break
            extra = safe_transcribe(more)
            if extra:
                text = f"{text} {extra}".strip()

        if len(text) < 2:
            if VOICE_DEBUG:
                log("info", "👂 (no words recognised)")
            continue
        if any(n in fold(text) for n in NOISE_PHRASES) or fold(text).strip(" .!?,") in NOISE_EXACT:
            if VOICE_DEBUG:
                log("info", f"👂 (ignored as noise: {text})")
            continue
        if VOICE_DEBUG:
            log("info", f"👂 Heard: {text}")
        if answer_confirm(text):
            continue

        hit, rest = strip_wake(text)
        music = STATE["mode"] == "always" and not forced and spotify_playing()
        needs_wake = (STATE["mode"] == "wake" and not is_active() and not forced) or music
        if needs_wake and not hit:
            if VOICE_DEBUG:
                log("info", f"   (ignored – {'music is playing, ' if music else ''}"
                            f"say “{wake_phrase()}” first or press Talk)")
            continue
        if hit:
            if not rest:
                log("info", "👂 Yes?")
                extend_active()
                speak("Yes?" if STATE["lang"] == "en" else "Ano?")
                continue
            text = rest
        cmd_q.put(("voice", text))


# ==========================================
# TERMINAL WINDOW (main thread)
# ==========================================
BG, BAR, FG = "#0d1117", "#161b22", "#c9d1d9"
COLORS = {
    "user": "#58a6ff", "step": "#d29922", "summary": "#a371f7", "ai": "#3fb950",
    "error": "#f85149", "info": "#8b949e", "confirm": "#ffa657", "alarm": "#ff7b72", "output": "#79c0ff",
}

HELP = (
    "Type a command like you would say it (e.g. 'open Safari', 'timer 5 minutes', 'weather').\n"
    "Voice modes (button at the bottom or /mode):\n"
    "  always = always listening (default)   wake = say “Hey/Hej Llama” first   push = 🎤 Talk / Ctrl+Space\n"
    "Press ⏹ Stop / Esc (or type 'stop') to silence speech.\n"
    "  /mode always|wake|push   change listening mode\n"
    "  /lang auto|en|cs   language you speak and the assistant answers in (auto = follows you)\n"
    "  /model auto|fast|smart|cloud   auto = small local model, big one for hard tasks, Gemini for coding\n"
    "  (start with “gemini:” to force Gemini; coding requests go to Gemini automatically)\n"
    "  (start a sentence with “přemýšlej:” or “think:” to force the big model)\n"
    "  /voice on|off      microphone\n"
    "  /stt whisper|google   speech recognition engine\n"
    "  /memory [clear]    show / clear long-term memory\n"
    "  /undo              undo the last code the assistant wrote into a file\n"
    "  /offline on|off    block everything that goes online (Gemini, web, weather, Google speech)\n"
    "  /mic [number]      list or choose the microphone\n"
    "  /pin  /compact     keep the window on top / shrink it\n"
    "  /autostart on|off  open the assistant at login\n"
    "  /history           open the conversation log\n"
    "  say or type “run it” to run the last Python file; Cmd +/- changes the font size\n"
    "  /clear  /reset     clear screen / forget chat context\n"
    "  /help   /quit\n"
)

def run_selftest() -> int:
    """python assistent.py --selftest  – checks the instant commands, routing and safeguards. Touches nothing."""
    fails, total = [], [0]

    def check(name, got, want):
        total[0] += 1
        ok = got == want
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"\n        got:  {got!r}\n        want: {want!r}"))
        if not ok:
            fails.append(name)

    def labels(text):
        plans, rest = plan_fast(text)
        return [lab for lab, _, _ in plans], rest

    STATE["lang"], STATE["last_results"] = "en", []
    for sentence, want_labels, want_rest in [
        ("set the volume to 30", ["volume"], None),
        ("volume 50", ["volume"], None),
        ("next song", ["spotify:next"], None),
        ("pause the music", ["spotify:pause"], None),
        ("open Safari", ["open_app"], None),
        ("open Spotify and set the volume to 20", ["open_app", "volume"], None),
        ("set the volume to 10 and then write a poem about cats", ["volume"], "write a poem about cats"),
        ("write a poem and set the volume to 10", [], None),
        ("open file test.py inside tes123 folder, and open it in visual studio code in a new window", ["open_in_app"], None),
        ("open the folder tes123 located on desktop", ["open_item"], None),
        ("run it", ["run"], None),
        ("run it and fix it", ["run_fix"], None),
        ("undo", ["undo"], None),
        ("timer for 5 minutes", ["timer"], None),
        ("what's the battery", ["battery"], None),
        ("close the window", ["close_window"], None),
        ("tell me a joke", [], None),
        ("write a small for loop inside it", [], None),
    ]:
        check(f"instant: {sentence[:60]}", labels(sentence), (want_labels, want_rest))
    STATE["last_results"] = ["/a", "/b"]
    check("instant: open the second one", labels("open the second one"), (["open_result"], None))
    STATE["last_results"] = []

    check("search intent", extract_search_query("Search for best pizza near me"), "best pizza near me")
    check("search intent: local file is not a web search", extract_search_query("search file invoice"), None)
    check("coding request detected", is_coding("write a small for loop inside it"), True)
    check("plain command is not coding", is_coding("open Safari"), False)
    check("hard task -> big model", needs_smart_model("write an email to my boss about the holiday"), True)
    check("easy task -> small model", needs_smart_model("open Safari"), False)
    check("wake word", strip_wake("Hey, Llama, open Safari"), (True, "open Safari"))
    check("language: English", guess_lang("open the folder and show me the files"), "en")
    check("language: Czech", guess_lang("otevři složku na ploše"), "cs")
    names = {f.__name__ for f in select_tools(fold("set the volume to 20"))}
    check("tool filter keeps set_volume", "set_volume" in names, True)
    check("tool filter hides get_weather", "get_weather" in names, False)
    check("tool filter finds get_weather when asked",
          "get_weather" in {f.__name__ for f in select_tools(fold("what is the weather in Prague"))}, True)
    check("guard: opening a file cannot pause Spotify",
          bool(re.search(TOOL_GUARDS["control_spotify"], fold("open test.py in visual studio code"))), False)
    check("guard: 'pause the music' may use Spotify",
          bool(re.search(TOOL_GUARDS["control_spotify"], fold("pause the music"))), True)
    check("translation of tool results", localize("Hlasitost nastavena na 30 %."), "Volume set to 30%.")
    check("code is never read aloud", "```" in speakable("ok ```python\nprint(1)\n```"), False)
    print(f"\n{total[0] - len(fails)}/{total[0]} checks passed")
    return 1 if fails else 0


if "--selftest" in sys.argv:
    sys.exit(run_selftest())

load_settings()          # restore language / mode / model choice from the last session
root = tk.Tk()
print(f"Python {sys.version.split()[0]} | Tk {tk.TkVersion} | {sys.executable}")
if tk.TkVersion < 8.6:
    print(
        "\n⚠️  Your Python uses Tk %s, which draws a blank window on modern macOS.\n"
        "Fix:  brew install python-tk@3.12\n"
        "Then run this script with /opt/homebrew/opt/python@3.12/bin/python3.12 (new virtualenv).\n" % tk.TkVersion
    )
    root.destroy()
    sys.exit(1)

root.title("Llama Assistant Terminal")
root.geometry("940x640")
root.configure(bg=BG)

status_label = tk.Label(root, text="", bg=BAR, fg=FG, anchor="w", font=("Menlo", 11), padx=10, pady=5)
status_label.pack(side="top", fill="x")

input_frame = tk.Frame(root, bg=BG)
input_frame.pack(side="bottom", fill="x")

output = scrolledtext.ScrolledText(
    root, bg=BG, fg=FG, insertbackground=FG, font=("Menlo", 12),
    wrap="word", state="disabled", borderwidth=0, padx=10, pady=10,
)
output.pack(fill="both", expand=True)
for tag, color in COLORS.items():
    output.tag_configure(tag, foreground=color)
output.tag_configure("summary", foreground=COLORS["summary"], font=("Menlo", 12, "italic"))

tk.Label(input_frame, text="›", bg=BG, fg=COLORS["user"], font=("Menlo", 14, "bold")).pack(side="left", padx=(10, 4))
entry = tk.Entry(input_frame, bg=BG, fg=FG, insertbackground=FG, font=("Menlo", 12),
                 borderwidth=0, highlightthickness=0)
entry.pack(side="left", fill="x", expand=True, pady=10, padx=(0, 10))
entry.focus_set()

try:
    _hist_file = open(HISTORY_FILE, "a", encoding="utf-8", buffering=1)
except Exception:
    _hist_file = None


def write_history(s: str):
    if _hist_file:
        try:
            _hist_file.write(s)
        except Exception:
            pass


def raw_insert(tag: str, text: str):
    output.configure(state="normal")
    output.insert("end", text, tag)
    output.configure(state="disabled")
    output.see("end")


def append(tag: str, text: str):
    raw_insert(tag, text + "\n")
    write_history(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}\n")


def current_status() -> str:
    if STATE["phase"] == "loading":
        s = "⏳ Loading speech model (first run downloads it)..."
    elif STATE["phase"] == "transcribing":
        s = "✍ Transcribing what you said..."
    elif speaking.is_set():
        s = "🔊 Speaking   (Esc = stop)"
    elif confirm_pending.is_set():
        s = "❓ Waiting for your yes / no"
    elif busy.is_set():
        s = "🧠 Thinking"
    elif talk_once.is_set():
        s = "🎤 Listening (push-to-talk)"
    elif not voice_on.is_set():
        s = "🔇 Microphone off   (/voice on)"
    elif STATE["mode"] == "push":
        s = "⏸ Push-to-talk: press 🎤 Talk or Ctrl+Space"
    elif STATE["mode"] == "wake" and not is_active():
        s = f"💤 Waiting for “{wake_phrase()}”"
    else:
        s = "🎤 Listening"
    if active_timers:
        s += f"      ⏰ {len(active_timers)} timer(s)"
    lang = STATE["lang"].upper() + (" (auto)" if STATE["lang_mode"] == "auto" else "")
    off = "OFFLINE      " if STATE["offline"] else ""
    return s + f"      {off}STT: {STATE['stt']}      Model: {STATE['model_mode']}      Lang: {lang}"


_streaming = {"on": False, "held": []}


def poll_ui():
    try:
        while True:
            tag, text = ui_q.get_nowait()
            if tag == "stream_start":
                _streaming["on"] = True
                raw_insert("ai", text)
                write_history(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}")
            elif tag == "stream":
                raw_insert("ai", text)
                write_history(text)
            elif tag == "stream_end":
                raw_insert("ai", "\n")
                write_history("\n")
                _streaming["on"] = False
                for held_tag, held_text in _streaming["held"]:
                    append(held_tag, held_text)
                _streaming["held"].clear()
            elif _streaming["on"]:
                _streaming["held"].append((tag, text))   # show after the answer finished
            else:
                append(tag, text)
    except queue.Empty:
        pass
    status_label.configure(text=current_status())
    if not stop.is_set():
        root.after(100, poll_ui)


def quit_app(event=None):
    stop.set()
    stop_speaking()
    root.destroy()


def set_mode(mode: str):
    STATE["mode"] = mode
    talk_once.clear()
    if mode == "wake":
        STATE["active_until"] = 0.0
    mode_btn.configure(text=f"Mode: {MODE_LABELS[mode]}")
    append("info", f"Listening mode → {MODE_LABELS[mode]}: {mode_hint()}")
    save_settings()


def cycle_mode():
    set_mode(MODES[(MODES.index(STATE["mode"]) + 1) % len(MODES)])


def handle_slash(text: str):
    parts = text.lower().split()
    cmd = parts[0]
    arg = parts[1] if len(parts) > 1 else ""
    if cmd == "/quit":
        quit_app()
    elif cmd == "/clear":
        output.configure(state="normal")
        output.delete("1.0", "end")
        output.configure(state="disabled")
    elif cmd == "/reset":
        messages[:] = [messages[0]]
        append("info", "Conversation context cleared.")
    elif cmd == "/voice":
        if arg == "off":
            voice_on.clear()
            append("info", "🎤 Voice off.")
        else:
            voice_on.set()
            append("info", "🎤 Voice on.")
    elif cmd == "/mode":
        if arg in MODES:
            set_mode(arg)
        else:
            append("error", "Usage: /mode always|wake|push")
    elif cmd == "/wake":          # old alias
        set_mode("wake" if arg != "off" else "always")
    elif cmd == "/lang":
        if arg == "auto":
            STATE["lang_mode"] = "auto"
            append("info", "Language: automatic – the assistant follows the language you speak or type (EN / CS).")
        elif arg in ("en", "cs"):
            STATE["lang"] = arg
            STATE["lang_mode"] = "fixed"
            append("info", f"Language pinned to {'English' if arg == 'en' else 'Czech'} "
                           f"(speech recognition {STT_LANGS[arg]}, voice {VOICES[arg]}, replies in {arg.upper()})")
        else:
            append("error", "Usage: /lang auto|en|cs")
    elif cmd == "/model":
        if arg in ("auto", "fast", "smart", "cloud"):
            STATE["model_mode"] = arg
            STATE["smart_ok"] = True      # re-check (e.g. after `ollama pull`)
            _gem["client"] = None         # re-read the API key
            desc = {"auto": f"{FAST_MODEL} for everyday tasks, {SMART_MODEL} for hard ones, Gemini for coding",
                    "fast": f"always {FAST_MODEL}", "smart": f"always {SMART_MODEL}",
                    "cloud": "always Gemini (online)"}[arg]
            append("info", f"Model mode → {arg}: {desc}")
        else:
            append("error", "Usage: /model auto|fast|smart|cloud")
    elif cmd == "/stt":
        if arg in ("whisper", "google", "mlx"):
            STATE["stt"] = arg
            append("info", f"Speech recognition: {arg}")
            if arg == "whisper":
                threading.Thread(target=warmup, daemon=True).start()
        else:
            append("error", "Usage: /stt whisper|google|mlx")
    elif cmd == "/memory":
        if arg == "clear":
            with _mem_lock:
                save_memory([])
            append("info", "Memory cleared.")
        else:
            notes = load_memory()
            append("info", "Memory:\n" + "\n".join(f"  - {n}" for n in notes) if notes else "Memory is empty.")
    elif cmd == "/offline":
        STATE["offline"] = arg != "off"
        if STATE["offline"]:
            if STATE["stt"] == "google":
                STATE["stt"] = "whisper"
            append("info", "Offline mode ON – Gemini, web search, weather and Google speech recognition are blocked.")
        else:
            append("info", "Offline mode off.")
    elif cmd == "/mic":
        try:
            names = sr.Microphone.list_microphone_names()
        except Exception as e:
            append("error", f"Can't list microphones: {e}")
            names = []
        if arg.isdigit() and int(arg) < len(names):
            STATE["mic_index"] = int(arg)
            STATE["mic_changed"] = True
            append("info", f"Microphone → {names[int(arg)]}")
        elif arg in ("default", "auto"):
            STATE["mic_index"] = None
            STATE["mic_changed"] = True
            append("info", "Microphone → system default")
        elif names:
            current = STATE["mic_index"]
            append("info", "Microphones:\n" + "\n".join(
                f"  {i}: {n}{'   ← in use' if i == current else ''}" for i, n in enumerate(names))
                   + "\nUse /mic <number>, or /mic default")
    elif cmd == "/pin":
        STATE["pinned"] = not STATE["pinned"]
        root.attributes("-topmost", STATE["pinned"])
        append("info", "Window stays on top." if STATE["pinned"] else "Window no longer stays on top.")
    elif cmd == "/compact":
        if not STATE["compact"]:
            STATE["geo"] = root.geometry()
            root.geometry("560x240")
        elif STATE["geo"]:
            root.geometry(STATE["geo"])
        STATE["compact"] = not STATE["compact"]
    elif cmd == "/autostart":
        append("info", set_autostart(arg != "off"))
    elif cmd == "/undo":
        append("info", localize(undo_last_write()))
    elif cmd == "/history":
        subprocess.run(["open", HISTORY_FILE])
    elif cmd == "/stop":
        stop_speaking()
    elif cmd == "/help":
        append("info", HELP)
    else:
        append("error", f"Unknown command: {text} (try /help)")


history, hist_idx = [], 0


def on_enter(event=None):
    global hist_idx
    text = entry.get().strip()
    entry.delete(0, "end")
    if not text:
        return
    history.append(text)
    hist_idx = len(history)

    if text.startswith("/"):
        handle_slash(text)
        save_settings()
        return
    if fold(text).strip() in STOP_WORDS:
        append("user", f"› {text}")
        stop_speaking()
        answer_confirm("ne")
        return
    if confirm_pending.is_set():
        append("user", f"› {text}")
        answer_confirm(text)
        return
    cmd_q.put(("text", text))


def history_up(event):
    global hist_idx
    if history and hist_idx > 0:
        hist_idx -= 1
        entry.delete(0, "end")
        entry.insert(0, history[hist_idx])


def history_down(event):
    global hist_idx
    entry.delete(0, "end")
    if hist_idx < len(history) - 1:
        hist_idx += 1
        entry.insert(0, history[hist_idx])
    else:
        hist_idx = len(history)


def talk_click():
    STATE["talk_deadline"] = time.time() + 12
    talk_once.set()
    entry.focus_set()


mode_btn = tk.Button(input_frame, text=f"Mode: {MODE_LABELS[STATE['mode']]}", command=cycle_mode,
                     highlightbackground=BG)
mode_btn.pack(side="right", padx=(0, 6))
tk.Button(input_frame, text="🎤 Talk", command=talk_click, highlightbackground=BG).pack(side="right", padx=(0, 6))
tk.Button(input_frame, text="⏹ Stop", command=stop_speaking, highlightbackground=BG).pack(side="right", padx=(0, 6))

entry.bind("<Return>", on_enter)
entry.bind("<Up>", history_up)
entry.bind("<Down>", history_down)
root.bind("<Escape>", lambda e: stop_speaking())
root.bind("<Control-space>", lambda e: talk_click())

_font = {"size": 12}


def zoom(delta: int):
    _font["size"] = 12 if delta == 0 else max(8, min(28, _font["size"] + delta))
    output.configure(font=("Menlo", _font["size"]))
    output.tag_configure("summary", font=("Menlo", _font["size"], "italic"))


for _key in ("<Command-equal>", "<Command-plus>"):
    root.bind(_key, lambda e: zoom(1))
root.bind("<Command-minus>", lambda e: zoom(-1))
root.bind("<Command-0>", lambda e: zoom(0))
root.protocol("WM_DELETE_WINDOW", quit_app)

append("info", "Llama Assistant ready. Just talk or type – /help for commands, /mode to change how it listens.\n")
restore_timers()

threading.Thread(target=warmup, daemon=True).start()
threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=tts_worker, daemon=True).start()
threading.Thread(target=start_hotkey, daemon=True).start()
threading.Thread(target=voice_loop, daemon=True).start()
poll_ui()
root.mainloop()