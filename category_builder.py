"""Pipeline to build a new quiz category from a user-provided topic.

Steps:
1. Ask an LLM for a list of famous people in the category, with genders,
   fame levels and short descriptions.
2. Fetch each person's portrait from Wikipedia (MediaWiki API).
3. Detect the face with OpenCV, crop a square around it and resize it to the
   standard picture format used by the quiz.
"""

import io
import json

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from openai import OpenAI

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
HTTP_HEADERS = {"User-Agent": "FaceQuiz/1.0 (https://facequiz.streamlit.app)"}
REQUEST_TIMEOUT = 20  # seconds

OUTPUT_IMAGE_SIZE = 400  # px, square
FACE_MARGIN = 0.8  # extra space around the detected face box, as a fraction of its size
MIN_IMAGE_SIDE = 150  # px, reject Wikipedia images smaller than this

VALID_GENDERS = {"male", "female"}
VALID_FAME_LEVELS = {1, 2, 3}

LLM_SYSTEM_PROMPT = "You are a careful assistant that produces structured data for a face-recognition quiz."

LLM_USER_PROMPT_TEMPLATE = """List {n} real famous people for the quiz category "{category}".

Rules:
- Only real people who have an English Wikipedia page with a portrait photo of their face.
- Use the exact name of their English Wikipedia page (without any disambiguation suffix in parentheses).
- fame_level: 1 = world famous, 2 = well known, 3 = known mostly to fans of the field. Aim for a roughly even split across the three levels.
- Include a mix of genders when the category allows it.
- short_description: one sentence of at most 20 words describing the person. It is shown as a hint during the quiz, so it MUST NOT contain the person's name (first or last) or initials.
- Do not include any of these people, they are already in the quiz: {excluded}

Respond with JSON only, in this exact shape:
{{"people": [{{"name": "...", "gender": "male", "fame_level": 1, "short_description": "..."}}]}}
"""


def blob_name_for(person_name):
    """GCS path for a person's picture, matching the convention used by the quiz."""
    return "DATA/Pictures/" + person_name.replace(" ", "_") + ".png"


def generate_people_with_llm(category, existing_names, n=18):
    """Ask the LLM for a list of people in the category. Returns a list of dicts
    with keys: name, gender, fame_level, short_description."""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

    existing_lower = {name.strip().lower() for name in existing_names}
    prompt = LLM_USER_PROMPT_TEMPLATE.format(
        n=n,
        category=category,
        excluded=", ".join(sorted(existing_names)[:200]) or "none",
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    data = json.loads(response.choices[0].message.content)

    people = []
    seen = set()
    for entry in data.get("people", []):
        person = _validate_person(entry)
        if person is None:
            continue
        key = person["name"].lower()
        if key in existing_lower or key in seen:
            continue
        seen.add(key)
        people.append(person)
    return people


def _validate_person(entry):
    """Sanitize a single LLM-generated entry, returning None if it is unusable."""
    if not isinstance(entry, dict):
        return None
    name = str(entry.get("name", "")).strip()
    gender = str(entry.get("gender", "")).strip().lower()
    description = str(entry.get("short_description", "")).strip()
    try:
        fame_level = int(entry.get("fame_level"))
    except (TypeError, ValueError):
        return None
    if not name or not description or gender not in VALID_GENDERS or fame_level not in VALID_FAME_LEVELS:
        return None
    return {
        "name": name,
        "gender": gender,
        "fame_level": fame_level,
        "short_description": description,
    }


def fetch_wikipedia_image(name):
    """Return the raw bytes of the person's Wikipedia portrait, or None."""
    image_url = _query_page_image(name)
    if image_url is None:
        # The exact title didn't resolve; fall back to a Wikipedia search.
        title = _search_wikipedia_title(name)
        if title:
            image_url = _query_page_image(title)
    if image_url is None or image_url.lower().endswith(".svg"):
        return None

    response = requests.get(image_url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content


def _query_page_image(title):
    """Get the URL of the main image of a Wikipedia page, following redirects."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageimages",
        "piprop": "original",
        "redirects": 1,
        "format": "json",
    }
    response = requests.get(WIKI_API_URL, params=params, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    for page in pages.values():
        original = page.get("original")
        if original and original.get("source"):
            return original["source"]
    return None


def _search_wikipedia_title(name):
    """Return the title of the best Wikipedia search match for a name, or None."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": name,
        "srlimit": 1,
        "format": "json",
    }
    response = requests.get(WIKI_API_URL, params=params, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    results = response.json().get("query", {}).get("search", [])
    return results[0]["title"] if results else None


@st.cache_resource
def _get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_and_crop_face(image_bytes):
    """Crop a square PNG centered on the largest detected face.

    Returns the PNG bytes, or None if the image is unusable (too small,
    unreadable, or no face found)."""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None

    width, height = pil_image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        return None

    # Downscale very large images before detection to keep it fast.
    detection_scale = min(1.0, 1200 / max(width, height))
    if detection_scale < 1.0:
        detection_image = pil_image.resize(
            (int(width * detection_scale), int(height * detection_scale)), Image.LANCZOS
        )
    else:
        detection_image = pil_image

    bgr = cv2.cvtColor(np.array(detection_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        return None

    # Largest face, mapped back to full-resolution coordinates
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    x, y, w, h = (int(v / detection_scale) for v in (x, y, w, h))

    # The cascade box excludes hair and chin, so grow it and shift it up a little.
    center_x = x + w / 2
    center_y = y + h / 2 - 0.1 * h
    half_side = max(w, h) * (1 + FACE_MARGIN) / 2
    half_side = min(half_side, width / 2, height / 2)

    left = int(np.clip(center_x - half_side, 0, width - 2 * half_side))
    top = int(np.clip(center_y - half_side, 0, height - 2 * half_side))
    side = int(2 * half_side)

    cropped = pil_image.crop((left, top, left + side, top + side))
    cropped = cropped.resize((OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), Image.LANCZOS)

    output = io.BytesIO()
    cropped.save(output, format="PNG")
    return output.getvalue()


def build_person_picture(name):
    """Fetch the Wikipedia portrait for a person and return face-cropped PNG bytes.

    Returns (png_bytes, None) on success or (None, reason) on failure."""
    try:
        image_bytes = fetch_wikipedia_image(name)
    except requests.RequestException as e:
        return None, f"Wikipedia request failed ({e})"
    if image_bytes is None:
        return None, "no usable Wikipedia picture found"

    png_bytes = detect_and_crop_face(image_bytes)
    if png_bytes is None:
        return None, "no face detected in the Wikipedia picture"
    return png_bytes, None
