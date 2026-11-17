
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import (
    FileResponse,
    StreamingResponse,
    JSONResponse
)
from fastapi import HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import requests
import time
import langid
import re
import io
import uuid

from typing import Dict, List

from groq import Groq

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from gtts import gTTS


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_MODEL = "openai/gpt-oss-20b"

groq_client = None

if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_KEY":
    print("WARNING: GROQ_API_KEY is missing.")
else:
    try:
        groq_client = Groq(
            api_key=GROQ_API_KEY,
            timeout=45.0,
            max_retries=0
        )
        print("Groq client initialized successfully.")

    except Exception as e:
        print(f"Groq initialization error: {e}")


# ============================================================
# HUGGING FACE CONFIGURATION
# ============================================================

HF_IMAGE_URL = (
    "https://api-inference.huggingface.co/models/"
    "black-forest-labs/FLUX.1-schnell"
)


# ============================================================
# USAGE LIMITS
# ============================================================

image_limit_per_day = 5
question_limit_per_day = 200

ip_usage_tracker: Dict[str, Dict] = {}
chat_history: Dict[str, List[Dict[str, str]]] = {}
last_answer: Dict[str, str] = {}


# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI(
    title="Desh AI",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def serve_homepage():
    return FileResponse("static/index.html")


# ============================================================
# REQUEST MODELS
# ============================================================

class Question(BaseModel):
    question: str


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def convert_to_latex_math(text: str) -> str:

    replacements = {
        r"\b1\s+upon\s+2\b": r"\\frac{1}{2}",
        r"\b1\s+upon\s+3\b": r"\\frac{1}{3}",
        r"\bsquare\s+root\s+of\s+(\w+)": r"\\sqrt{\1}",
        r"\bcube\s+root\s+of\s+(\w+)": r"\\sqrt[3]{\1}",
        r"\bx\s+square\b": r"x^2",
        r"\bx\s+cube\b": r"x^3",
        r"\bx\s+power\s+(\d+)": r"x^{\1}",
        r"\bupon\b": "/",
        r"\btimes\b": r"\\times ",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return text


def detect_language(text: str) -> str:

    try:
        lang, _ = langid.classify(text)
        return lang

    except Exception:
        return "en"


def get_client_ip(request: Request) -> str:

    if request.client:
        return request.client.host

    return "unknown"


def reset_if_new_day(ip: str):

    today = time.strftime("%Y-%m-%d")

    if (
        ip not in ip_usage_tracker
        or ip_usage_tracker[ip].get("date") != today
    ):

        ip_usage_tracker[ip] = {
            "count": 0,
            "img_count": 0,
            "date": today
        }

        chat_history[ip] = []
        last_answer.pop(ip, None)


# ============================================================
# YOUTUBE SEARCH
# ============================================================

def fetch_youtube_videos(
    query: str,
    max_results: int = 1
):

    if not YOUTUBE_API_KEY:
        return []

    url = "https://www.googleapis.com/youtube/v3/search"

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=8
        )

        if response.status_code != 200:
            print(
                f"YouTube error: {response.status_code}"
            )
            return []

        items = response.json().get("items", [])

        videos = []

        for item in items:

            video_id = item.get(
                "id", {}
            ).get("videoId")

            snippet = item.get("snippet", {})

            thumbnails = snippet.get(
                "thumbnails", {}
            )

            thumbnail = thumbnails.get(
                "medium", {}
            ).get("url")

            if not video_id:
                continue

            videos.append({
                "title": snippet.get(
                    "title",
                    "YouTube Video"
                ),
                "thumbnail": thumbnail,
                "videoId": video_id
            })

        return videos

    except requests.RequestException as e:

        print(f"YouTube connection error: {e}")
        return []

    except Exception as e:

        print(f"YouTube API error: {e}")
        return []


# ============================================================
# RESPONSE LENGTH SAFETY
# ============================================================

def summarize_text(
    text: str,
    max_tokens: int = 1000
) -> str:

    words = text.split()

    estimated_limit = int(max_tokens * 0.75)

    shortened = words[:estimated_limit]

    if not shortened:
        return ""

    return (
        " ".join(shortened)
        + "... (summary)"
    )


def ensure_token_safe_response(
    full_text: str,
    max_tokens: int = 1500
) -> str:

    if not full_text:
        return ""

    estimated_tokens = len(full_text) / 4

    if estimated_tokens > max_tokens:

        return (
            "🔍 Summary due to length:\n"
            + summarize_text(
                full_text,
                max_tokens
            )
        )

    return full_text


# ============================================================
# GROQ API HELPER
# ============================================================

def ask_groq_api(
    messages: List[Dict[str, str]],
    max_tokens: int = 800
):

    if not groq_client:

        raise HTTPException(
            status_code=503,
            detail=(
                "Groq API is not configured. "
                "Please check GROQ_API_KEY."
            )
        )

    try:

        response = groq_client.chat.completions.create(

            model=GROQ_MODEL,

            messages=messages,

            max_completion_tokens=max_tokens

        )

        if not response.choices:

            raise HTTPException(
                status_code=502,
                detail="Groq returned an empty response."
            )

        content = response.choices[0].message.content

        if not content:

            raise HTTPException(
                status_code=502,
                detail="Groq returned an empty answer."
            )

        return str(content)

    except HTTPException:
        raise

    except Exception as e:

        print(
            f"Groq API error: {type(e).__name__}: {e}"
        )

        status_code = getattr(
            e,
            "status_code",
            None
        )

        # Free-tier rate limit reached.
        if status_code == 429:

            raise HTTPException(
                status_code=429,
                detail=(
                    "Desh AI has temporarily reached "
                    "its Groq API usage limit. "
                    "Please try again later."
                )
            )

        # Invalid or unauthorized API key.
        if status_code == 401:

            raise HTTPException(
                status_code=503,
                detail=(
                    "Groq API authentication failed. "
                    "Please check your API key."
                )
            )

        # Model unavailable.
        if status_code == 404:

            raise HTTPException(
                status_code=503,
                detail=(
                    "The selected Groq model is unavailable. "
                    "Check the current supported model list."
                )
            )

        # Temporary provider/server errors.
        if status_code in [500, 502, 503, 504]:

            raise HTTPException(
                status_code=503,
                detail=(
                    "Groq is temporarily unavailable. "
                    "Please try again shortly."
                )
            )

        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to generate an answer right now. "
                "Please try again."
            )
        )


# ============================================================
# HUGGING FACE IMAGE GENERATION
# ============================================================

def generate_image_hf(
    prompt: str,
    ip: str
):

    if not HF_API_KEY:

        return {
            "error": (
                "❌ HF_API_KEY is not configured "
                "on the server."
            )
        }

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "width": 1024,
            "height": 768,
            "num_inference_steps": 30
        }
    }

    try:

        res = requests.post(
            HF_IMAGE_URL,
            headers=headers,
            json=payload,
            timeout=90
        )

        if res.status_code == 200:

            image_bytes = res.content

            if not image_bytes.startswith(
                b"\x89PNG\r\n\x1a\n"
            ):

                return {
                    "error": (
                        "❌ Image provider returned "
                        "an invalid image response."
                    )
                }

            os.makedirs(
                "static",
                exist_ok=True
            )

            file_name = (
                f"generated_{uuid.uuid4().hex}.png"
            )

            file_path = os.path.join(
                "static",
                file_name
            )

            with open(file_path, "wb") as f:
                f.write(image_bytes)

            ip_usage_tracker[ip]["img_count"] += 1

            notice = (
                "🖼 Your image is ready:\n"
                f"[📥 Click to View Image]"
                f"(/static/{file_name})\n\n"
                f"⚠ Limit: "
                f"{image_limit_per_day} per day"
            )

            last_answer[ip] = notice

            return {
                "answer": notice,
                "youtube_videos":
                    fetch_youtube_videos(prompt)
            }

        try:
            error_details = res.json()
        except Exception:
            error_details = res.text[:500]

        print(
            f"Hugging Face status {res.status_code}: "
            f"{error_details}"
        )

        return {
            "error": (
                "❌ Image generation is temporarily "
                "unavailable. Please try again later."
            )
        }

    except requests.Timeout:

        return {
            "error": (
                "❌ Image generation timed out. "
                "Please try again."
            )
        }

    except requests.RequestException as e:

        print(f"Hugging Face connection error: {e}")

        return {
            "error": (
                "❌ Could not connect to the image provider."
            )
        }

    except Exception as e:

        print(f"Hugging Face image error: {e}")

        return {
            "error": (
                "❌ Image generation failed. "
                "Please try again."
            )
        }


# ============================================================
# MAIN CHAT ROUTE
# ============================================================

@app.post("/ask")
def ask_question(
    data: Question,
    request: Request
):

    prompt = data.question.strip()

    if not prompt:

        return JSONResponse(
            {
                "answer": "❌ Please enter a question."
            },
            status_code=400
        )

    if len(prompt) > 12000:

        return JSONResponse(
            {
                "answer": (
                    "❌ Your question is too long. "
                    "Please shorten it."
                )
            },
            status_code=413
        )

    ip = get_client_ip(request)

    reset_if_new_day(ip)

    # --------------------------------------------------------
    # DAILY QUESTION LIMIT
    # --------------------------------------------------------

    if (
        ip_usage_tracker[ip]["count"]
        >= question_limit_per_day
    ):

        return JSONResponse(
            {
                "answer": (
                    f"❌ Daily limit reached "
                    f"({question_limit_per_day}/day)."
                )
            },
            status_code=429
        )

    # --------------------------------------------------------
    # LANGUAGE DETECTION
    # --------------------------------------------------------

    detected_lang = detect_language(prompt)

    prompt_lower = prompt.lower()

    # --------------------------------------------------------
    # FOUNDER / ABOUT DESH AI
    # --------------------------------------------------------

    founder_keywords = [
        "founder of",
        "who is your founder",
        "who made desh ai",
        "who created you",
        "creates you",
        "created you",
        "founded you",
        "your founder",
        "makes you",
        "ceo of desh ai",
        "owner of desh ai"
    ]

    if any(
        kw in prompt_lower
        for kw in founder_keywords
    ):

        reply = (
            "The Vision Behind 𝕯𝖊𝖘𝖍 𝐀𝖎: "
            "This platform is an AI-driven system "
            "established in 2025 to democratize "
            "advanced technology. Led by "
            "𝗗𝗲𝘃𝗲𝘀𝗵 𝗦𝗶𝗻𝗴𝗵 "
            "(Founder & Managing Director), "
            "the company focuses on digital innovation.\n\n"

            "Core Capabilities and Innovations:\n"
            "Desh AI features conversational AI, "
            "voice synthesis, PDF tools, and creative "
            "utilities designed for everyday use.\n\n"

            "Interactive Entertainment and API:\n"
            "The platform also includes interactive "
            "games, language tools and AI-powered "
            "experiences.\n\n"

            "The 2025 Milestone:\n"
            "Founded in 2025, Desh AI aims to make "
            "useful technology accessible through "
            "a simple web-based platform."
        )

        last_answer[ip] = reply

        return {
            "answer": reply,
            "youtube_videos":
                fetch_youtube_videos(prompt)
        }

    # --------------------------------------------------------
    # IMAGE GENERATION TRIGGER
    # --------------------------------------------------------

    image_keywords = [
        "generate image",
        "create image",
        "make an image",
        "draw an image",
        "generate a picture",
        "create a picture",
        "make a picture",
        "image generation"
    ]

    if any(
        word in prompt_lower
        for word in image_keywords
    ):

        if (
            ip_usage_tracker[ip]["img_count"]
            >= image_limit_per_day
        ):

            return JSONResponse(
                {
                    "answer": (
                        f"❌ Image limit reached "
                        f"({image_limit_per_day}/day)."
                    )
                },
                status_code=429
            )

        return generate_image_hf(
            prompt,
            ip
        )

    # --------------------------------------------------------
    # RESPONSE LENGTH
    # --------------------------------------------------------

    explain_keywords = [
        "explain",
        "describe",
        "elaborate",
        "in detail",
        "long answer"
    ]

    if any(
        kw in prompt_lower
        for kw in explain_keywords
    ):

        max_tokens = 1500

    else:

        max_tokens = 800

    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = {
        "role": "system",
        "content": (
            "You are Desh AI, a helpful, accurate and "
            "friendly AI assistant.\n"
            f"Reply in {detected_lang} language whenever "
            "possible.\n"
            "Understand the user's actual question before "
            "answering.\n"
            "Give accurate, clear and useful answers.\n"
            "Do not invent facts, sources, or statistics.\n"
            "If you are uncertain, say so honestly.\n"
            "Use emojis naturally, without overusing them.\n"
            "Format mathematical expressions clearly.\n"
            "Keep simple answers concise and explain "
            "complex questions properly."
        )
    }

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    previous_messages = chat_history.get(
        ip,
        []
    )[-8:]

    messages = (
        [system_prompt]
        + previous_messages
        + [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # --------------------------------------------------------
    # GROQ API REQUEST
    # --------------------------------------------------------

    try:

        reply = ask_groq_api(
            messages,
            max_tokens=max_tokens
        )

        reply = convert_to_latex_math(reply)

        reply = ensure_token_safe_response(
            reply,
            max_tokens=max_tokens
        )

        # Save history only after successful response.
        chat_history[ip] = (
            previous_messages
            + [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": reply
                }
            ]
        )

        last_answer[ip] = reply

        # Count only successful AI responses.
        ip_usage_tracker[ip]["count"] += 1

        return {
            "answer": reply,
            "youtube_videos":
                fetch_youtube_videos(prompt)
        }

    except HTTPException as e:

        return JSONResponse(
            {
                "error": e.detail,
                "answer": e.detail
            },
            status_code=e.status_code
        )

    except Exception as e:

        print(f"Error in /ask route: {e}")

        return JSONResponse(
            {
                "error": (
                    "An unexpected server error occurred. "
                    "Please try again."
                )
            },
            status_code=500
        )


# ============================================================
# TEXT TO SPEECH ROUTE
# ============================================================

@app.post("/tts")
def text_to_speech(data: dict):

    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():

        return JSONResponse(
            {
                "error": "No text provided."
            },
            status_code=400
        )

    if len(text) > 10000:

        return JSONResponse(
            {
                "error": "Text is too long for TTS."
            },
            status_code=413
        )

    lang = detect_language(text)

    if lang not in ["en", "hi"]:
        lang = "en"

    try:

        mp3_fp = io.BytesIO()

        tts = gTTS(
            text=text,
            lang=lang,
            slow=False
        )

        tts.write_to_fp(mp3_fp)

        mp3_fp.seek(0)

        return StreamingResponse(
            mp3_fp,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition":
                    "inline; filename=tts_output.mp3"
            }
        )

    except Exception as e:

        print(f"TTS error: {e}")

        return JSONResponse(
            {
                "error": (
                    "TTS generation failed. "
                    "Please try again."
                )
            },
            status_code=500
        )


# ============================================================
# PDF DOWNLOAD ROUTE
# ============================================================

@app.get("/download-pdf")
def download_pdf(request: Request):

    ip = get_client_ip(request)

    if (
        ip not in last_answer
        or not last_answer[ip].strip()
    ):

        return JSONResponse(
            {
                "error": (
                    "❌ No answer available for PDF."
                )
            },
            status_code=400
        )

    try:

        buffer = io.BytesIO()

        doc = SimpleDocTemplate(buffer)

        styles = getSampleStyleSheet()

        answer_text = last_answer[ip]

        answer_text = (
            answer_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

        story = [
            Paragraph(
                answer_text,
                styles["Normal"]
            )
        ]

        doc.build(story)

        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                    "attachment; filename=chat_answer.pdf"
            }
        )

    except Exception as e:

        print(f"PDF generation error: {e}")

        return JSONResponse(
            {
                "error": (
                    "PDF generation failed. "
                    "Please try again."
                )
            },
            status_code=500
        )


# ============================================================
# STATIC PAGE ROUTES
# ============================================================

@app.get("/numpuzz")
def serve_numpuzz():
    return FileResponse("static/numpuzz.html")


@app.get("/snake")
def serve_snake():
    return FileResponse("static/snake.html")


@app.get("/calculator")
def serve_calculator():
    return FileResponse("static/calculator.html")


@app.get("/BMI")
def serve_bmi():
    return FileResponse("static/BMI.html")


@app.get("/Dictionary")
def serve_dictionary():
    return FileResponse("static/Dictionary.html")


@app.get("/desh.html")
def serve_desh():
    return FileResponse("static/desh.html")


@app.get("/Tic")
def serve_tic():
    return FileResponse("static/Tic.html")


@app.get("/Tac")
def serve_tac():
    return FileResponse("static/Tac.html")


@app.get("/50")
def serve_50():
    return FileResponse("static/50.html")


@app.get("/neon")
def serve_neon():
    return FileResponse("static/neon.html")


@app.get("/waves")
def serve_waves():
    return FileResponse("static/waves.html")


@app.get("/dot")
def serve_dot():
    return FileResponse("static/grid.html")


@app.get("/pdf")
def serve_pdf():
    return FileResponse("static/pdf.html")


@app.get("/smart")
def serve_smart():
    return FileResponse("static/smart.html")


@app.get("/study")
def serve_study():
    return FileResponse("static/study.html")


@app.get("/vsics")
def serve_vsics():
    return FileResponse("static/vsics.html")


@app.get("/cam")
def serve_cam():
    return FileResponse("static/cam.html")


@app.get("/mag")
def serve_mag():
    return FileResponse("static/mag.html")
