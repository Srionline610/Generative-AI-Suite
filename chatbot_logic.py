import os
import time
import uuid
import fitz
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from gtts import gTTS
from typing import Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-2.5-flash")

if not GEMINI_KEY:
    raise ValueError("Set GEMINI_API_KEY in environment variables!")

chatbot = ChatGoogleGenerativeAI(
    model=GENAI_MODEL,
    google_api_key=GEMINI_KEY,
    temperature=0.3
)

_pdf_cache = {}

def load_pdf_text(pdf_path: str, max_chars: int = 4000) -> str:
    if not pdf_path:
        return ""
    if pdf_path in _pdf_cache:
        return _pdf_cache[pdf_path][:max_chars]
    try:
        pages = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pages.append(page.get_text("text"))
        full_text = "\n".join(pages).strip()
        _pdf_cache[pdf_path] = full_text
        return full_text[:max_chars]
    except Exception as e:
        return f"[Error reading PDF: {e}]"

def _normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if url.startswith("www."):
        url = "https://" + url
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return url

def scrape_website(url: str, max_chars: int = 3000) -> str:
    if not url:
        return ""
    url = _normalize_url(url)
    try:
        resp = requests.get(url, timeout=5, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9"
        })
        if resp.status_code != 200:
            return f"[Error scraping website: HTTP {resp.status_code}]"
        soup = BeautifulSoup(resp.text, "html.parser")
        content = soup.get_text(separator=" ", strip=True)
        return content[:max_chars]
    except Exception as e:
        return f"[Error scraping website: {e}]"

def build_context(pdf_text: str, web_text: str, state: dict, max_turns: int = 6) -> Tuple[str, bool]:
    parts = []
    has_external = False

    if pdf_text and not pdf_text.startswith("[Error"):
        parts.append("PDF_CONTENT:\n" + pdf_text[:4000])
        has_external = True

    if web_text and not web_text.startswith("[Error"):
        parts.append("WEB_CONTENT:\n" + web_text[:3000])
        has_external = True

    history = state.get("history", [])
    if history:
        flat = []
        last = history[-(max_turns*2):]
        for item in last:
            role = item.get("role", "user").capitalize()
            msg = item.get("content", "")
            if msg.strip():
                flat.append(f"{role}: {msg.strip()}")
        if flat:
            parts.append("RECENT_CHAT_HISTORY:\n" + "\n".join(flat))

    return "\n\n".join(parts), has_external

def ask_gemini(user_question: str, context: str, strict_use_context: bool) -> str:
    if not user_question:
        return "[No question provided.]"
    system_prompt = (
        "You are a helpful assistant. "
        + ("Use ONLY the provided CONTEXT to answer.\n" if strict_use_context else "")
        + f"CONTEXT:\n{context}" if strict_use_context else ""
    )
    try:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_question)]
        resp = chatbot.invoke(messages)
        return str(getattr(resp, "content", resp)).strip()
    except Exception as e:
        return f"[Gemini API error: {e}]"

def make_tts(answer_text: str, lang: str = "en") -> str:
    if not answer_text:
        return None
    tmp_dir = os.path.join(os.getcwd(), "tmp_tts")
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"response_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp3")
    gTTS(text=answer_text, lang=lang).save(path)
    return path

def answer_from_sources(user_input: str = "", pdf_path: str = None, url: str = None, state: dict = None, use_tts: bool = True):
    if state is None:
        state = {"history": []}

    pdf_text = load_pdf_text(pdf_path) if pdf_path else ""
    web_text = scrape_website(url) if url else ""

    context, has_external = build_context(pdf_text, web_text, state)
    strict = has_external

    answer = ask_gemini(user_input or "Hello", context, strict_use_context=strict)

    audio_path = make_tts(answer) if use_tts and answer and not answer.startswith("[Error") else None
    return answer, audio_path