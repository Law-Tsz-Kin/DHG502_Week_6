#!/usr/bin/env python3
"""Run OCR on hongloumeng.jpeg using the OpenRouter chat completions API.

Uses only the Python standard library. The API key is read from the
OPENROUTER_API_KEY environment variable.
"""

import base64
import json
import os
import sys
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.5-9b"
IMAGE_PATH = "hongloumeng.jpeg"
OUTPUT_PATH = "hongloumeng.txt"
TIMEOUT_SECONDS = 120

PROMPT = (
    "Perform OCR on this image. It is a page from a traditional Chinese "
    "woodblock-printed book, typeset in vertical columns.\n"
    "\n"
    "Reading order: each column is read from top to bottom, and the columns "
    "are ordered from the RIGHT side of the page to the LEFT. Transcribe the "
    "text in that correct reading order: start with the rightmost column, "
    "read it top to bottom, then move to the column immediately to its left, "
    "and continue until the leftmost column. Emit one output line per "
    "vertical column, preserving the line breaks.\n"
    "\n"
    "Character form: keep every character in its ORIGINAL traditional form "
    "(繁體字). Do NOT convert, simplify, or normalise characters to "
    "simplified Chinese; do not substitute variant forms. Transcribe "
    "exactly the glyphs that appear.\n"
    "\n"
    "Transcribe every piece of text visible on the page, including any "
    "library stamps, accession numbers, or seal impressions, keeping them "
    "on their own lines. Output only the transcribed text, with no "
    "explanations, translation, or extra commentary."
)


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 1

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: image file not found: {IMAGE_PATH}", file=sys.stderr)
        return 1

    with open(IMAGE_PATH, "rb") as fh:
        image_b64 = base64.b64encode(fh.read()).decode("ascii")

    payload = {
        "model": MODEL,
        "reasoning": {"enabled": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error {exc.code} from OpenRouter:\n{detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc.reason}", file=sys.stderr)
        return 1

    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print("Unexpected API response:", json.dumps(body, indent=2), file=sys.stderr)
        return 1

    text = (text or "").strip()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")

    print(text)
    print(f"\n[saved to {OUTPUT_PATH}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
