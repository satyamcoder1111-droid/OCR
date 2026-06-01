import base64
import json
import os
import re
import threading
from io import BytesIO
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename


load_dotenv()

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_IMAGE_DIMENSION = 1600
JPEG_QUALITY = 82

SYSTEM_PROMPT = """You are an accurate OCR and document extraction system for cheque and online-transfer images.

Return ONLY valid JSON. No markdown and no explanation.
The response must always use exactly this top-level structure:
{"online_transfer":{},"cheque":{}}

If the image is an online transfer receipt, fill online_transfer and keep cheque as {}.
If the image is a cheque, fill cheque and keep online_transfer as {}.
If the image is neither an online transfer receipt nor a cheque, return {"online_transfer":{},"cheque":{}}.

Use snake_case keys inside the matching object only. Never guess or invent fields.
Normalize dates to DD-MM-YYYY for cheques and DD/MM/YYYY for online transfers when confident; otherwise keep original date text.
Preserve useful visible identifiers and details such as cheque_no, receipt_no, transaction_no, payee, payer, beneficiary_name, beneficiary_iban, amount, amount_in_words, bank_name, account_no, iban, branch, from_account, and account_type.
Always include from_ocr:true and extraction_method:"llm" inside the filled object.
If text is unclear, return the best readable value with confidence. Omit invisible values and avoid nulls unless important and partially visible."""


llm_clients: dict[str, ChatGroq] = {}
key_rotation_lock = threading.Lock()
next_key_index = 0


def get_groq_api_keys() -> list[str]:
    key_names = ["GROQ_API_KEY"] + [f"GROQ_API_KEY_{index}" for index in range(1, 6)]
    keys: list[str] = []

    for key_name in key_names:
        key = os.getenv(key_name, "").strip()
        if key and key not in keys:
            keys.append(key)

    return keys


def get_llm(api_key: str) -> ChatGroq:
    if api_key not in llm_clients:
        llm_clients[api_key] = ChatGroq(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=2000,
            api_key=api_key,
        )

    return llm_clients[api_key]


def get_rotated_keys(keys: list[str]) -> list[str]:
    global next_key_index

    with key_rotation_lock:
        start_index = next_key_index % len(keys)
        next_key_index = (next_key_index + 1) % len(keys)

    return keys[start_index:] + keys[:start_index]


def is_rate_limit_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)

    return (
        status_code == 429
        or response_status == 429
        or "rate limit" in error_text
        or "rate_limit" in error_text
        or "too many requests" in error_text
        or "429" in error_text
    )


def invoke_groq_with_fallbacks(messages: list[SystemMessage | HumanMessage]):
    keys = get_groq_api_keys()
    if not keys:
        raise RuntimeError("No Groq API keys configured. Set GROQ_API_KEY and optional GROQ_API_KEY_1 through GROQ_API_KEY_5.")

    rotated_keys = get_rotated_keys(keys)

    for api_key in rotated_keys:
        try:
            return get_llm(api_key).invoke(messages)
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise

    raise RuntimeError(f"All configured Groq API keys are rate limited. Tried {len(keys)} key(s).")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_mime_type(filename: str) -> str:
    extension = filename.rsplit(".", 1)[1].lower()
    mime_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }
    return mime_types[extension]


def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def optimize_image(image_bytes: bytes) -> tuple[bytes, str]:
    with Image.open(BytesIO(image_bytes)) as image:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))

        if image.mode not in ("RGB", "L"):
            background = Image.new("RGB", image.size, "white")
            if image.mode == "RGBA":
                background.paste(image, mask=image.getchannel("A"))
            else:
                background.paste(image)
            image = background
        elif image.mode == "L":
            image = image.convert("RGB")

        output = BytesIO()
        image.save(output, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return output.getvalue(), "image/jpeg"


def clean_json_response(response_text: str) -> str:
    cleaned = response_text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    if not cleaned.startswith(("{", "[")):
        json_match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if json_match:
            cleaned = json_match.group(1).strip()

    return cleaned


def parse_json_response(response_text: str) -> dict[str, Any] | list[Any]:
    cleaned = clean_json_response(response_text)
    return json.loads(cleaned)


def looks_like_online_transfer(data: dict[str, Any]) -> bool:
    transfer_keys = {
        "receipt_no",
        "transaction_no",
        "transaction_id",
        "reference_no",
        "beneficiary_name",
        "beneficiary_iban",
        "from_account",
        "account_type",
    }
    document_type = str(data.get("document_type", "")).lower()
    return "transfer" in document_type or "receipt" in document_type or any(key in data for key in transfer_keys)


def looks_like_cheque(data: dict[str, Any]) -> bool:
    cheque_keys = {
        "cheque_no",
        "payee",
        "payer",
        "amount_in_words",
        "branch",
    }
    document_type = str(data.get("document_type", "")).lower()
    return "cheque" in document_type or "check" in document_type or any(key in data for key in cheque_keys)


def add_extraction_metadata(data: dict[str, Any]) -> dict[str, Any]:
    if data:
        data.setdefault("from_ocr", True)
        data.setdefault("extraction_method", "llm")
    return data


def normalize_extraction_response(data: dict[str, Any] | list[Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {"online_transfer": {}, "cheque": {}}

    if not isinstance(data, dict):
        return normalized

    online_transfer = data.get("online_transfer")
    cheque = data.get("cheque")

    if isinstance(online_transfer, dict) or isinstance(cheque, dict):
        normalized["online_transfer"] = add_extraction_metadata(online_transfer or {}) if isinstance(online_transfer, dict) else {}
        normalized["cheque"] = add_extraction_metadata(cheque or {}) if isinstance(cheque, dict) else {}
        return normalized

    if looks_like_online_transfer(data):
        normalized["online_transfer"] = add_extraction_metadata(data)
    elif looks_like_cheque(data):
        normalized["cheque"] = add_extraction_metadata(data)

    return normalized


def success_response(data: dict[str, Any] | list[Any]):
    return jsonify(data), 200


def error_response(message: str, status_code: int, **extra: Any):
    payload = {
        "success": False,
        "status": "failed",
        "code": status_code,
        "error": message,
    }
    payload.update(extra)
    return jsonify(payload), status_code


@app.route("/extract", methods=["POST"])
def extract_document():
    if "file" not in request.files:
        return error_response("No file field found. Use multipart/form-data with field name 'file'.", 400)

    uploaded_file = request.files["file"]

    if not uploaded_file or uploaded_file.filename == "":
        return error_response("No file selected.", 400)

    filename = secure_filename(uploaded_file.filename)
    if not filename or not allowed_file(filename):
        return error_response("Invalid file type. Allowed types: jpg, jpeg, png, webp.", 400)

    if not get_groq_api_keys():
        return error_response(
            "No Groq API keys configured. Set GROQ_API_KEY and optional GROQ_API_KEY_1 through GROQ_API_KEY_5.",
            500,
        )

    try:
        image_bytes = uploaded_file.read()
        if not image_bytes:
            return error_response("Uploaded file is empty.", 400)

        try:
            optimized_image_bytes, mime_type = optimize_image(image_bytes)
        except UnidentifiedImageError:
            return error_response("Uploaded file is not a valid image.", 400)

        base64_image = encode_image_to_base64(optimized_image_bytes)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Extract the cheque or online-transfer data from this image and return only the required JSON structure.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                        },
                    },
                ]
            ),
        ]

        response = invoke_groq_with_fallbacks(messages)
        raw_response = response.content if isinstance(response.content, str) else json.dumps(response.content)

        try:
            extracted_data = parse_json_response(raw_response)
        except json.JSONDecodeError:
            return error_response(
                "Model response was not valid JSON.",
                502,
                raw_model_response=raw_response,
            )

        return success_response(normalize_extraction_response(extracted_data))

    except Exception as exc:
        return error_response(str(exc), 500)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
