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
app.json.sort_keys = False

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_IMAGE_DIMENSION = 2400
JPEG_QUALITY = 92

SYSTEM_PROMPT = """You are an accurate OCR and document extraction system for cheque and online-transfer images.

Return ONLY valid JSON. No markdown and no explanation.
The response must always use exactly this top-level structure:
{"status":false,"data":{"online_transfer":{},"cheque":{}}}

Your first priority is to read visible text from the image and extract fields. Do not return empty objects when any readable cheque or transfer details are visible.

If the image contains cheque/check words, cheque number, payee, payer, amount in words, account number, IBAN, branch, bank name, MICR-like numbers, or a pay-to line, treat it as a cheque. Fill cheque and keep online_transfer as {}.
If the image contains receipt number, transaction number, reference number, beneficiary, sender/from account, transfer status, IBAN, bank code, or online/mobile banking transfer details, treat it as an online transfer. Fill online_transfer and keep cheque as {}.
If both are possible, choose the one with stronger evidence.
Only return {"status":false,"data":{"online_transfer":{},"cheque":{}}} when the image has no readable financial text at all.

Use snake_case keys inside the matching object only. Never guess or invent fields.
Normalize dates to DD-MM-YYYY for cheques and DD/MM/YYYY for online transfers when confident; otherwise keep original date text.
Preserve useful visible identifiers and details such as cheque_no, receipt_no, transaction_no, payee, payer, beneficiary_name, beneficiary_iban, amount, amount_in_words, bank_name, account_no, iban, branch, from_account, and account_type.
Always include from_ocr:true and extraction_method:"llm" inside the filled object.
If text is unclear, return the best readable value with confidence. Omit invisible values and avoid nulls unless important and partially visible."""

USER_EXTRACTION_PROMPT = """Look carefully at this image and extract the cheque or online-transfer details.

Return only this top-level JSON shape:
{"status":false,"data":{"online_transfer":{},"cheque":{}}}

Fill exactly one object when any readable details are visible and set status to true. Do not answer with both objects empty unless the image has no readable financial text."""

EMPTY_RETRY_PROMPT = """The previous extraction returned empty objects.

Re-read the image carefully. It is expected to be either a cheque or an online transfer receipt.
Extract any visible financial fields you can read, even if some text is unclear.

Return only this top-level JSON shape:
{"status":false,"data":{"online_transfer":{},"cheque":{}}}

Fill cheque if you see cheque/check, payee, payer, amount in words, account number, IBAN, bank, branch, or cheque number.
Fill online_transfer if you see receipt, transaction, reference, beneficiary, sender/from account, IBAN, amount, or transfer status.
Do not return both objects empty unless absolutely no text is readable. Set status to true when any fields are extracted."""


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


def build_vision_messages(prompt: str, base64_image: str, mime_type: str) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt,
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
        "receipt_number",
        "transaction_no",
        "transaction_number",
        "transaction_id",
        "transaction_reference",
        "reference_no",
        "reference_number",
        "beneficiary_name",
        "beneficiary_iban",
        "beneficiary_account",
        "beneficiary_account_no",
        "beneficiary_account_number",
        "from_account",
        "from_account_no",
        "from_account_number",
        "account_type",
    }
    document_type = str(data.get("document_type", "")).lower()
    return (
        "transfer" in document_type
        or "receipt" in document_type
        or "transaction" in document_type
        or any(key in data for key in transfer_keys)
    )


def looks_like_cheque(data: dict[str, Any]) -> bool:
    cheque_keys = {
        "cheque_no",
        "cheque_number",
        "check_no",
        "check_number",
        "payee",
        "pay_to",
        "payer",
        "drawer",
        "amount_in_words",
        "branch",
        "account_no",
        "account_number",
        "iban",
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

    extraction_data = data.get("data")
    if isinstance(extraction_data, dict):
        data = extraction_data

    online_transfer = data.get("online_transfer")
    cheque = data.get("cheque")
    remaining_data = {
        key: value
        for key, value in data.items()
        if key not in {"status", "online_transfer", "cheque"} and value not in ("", None, {}, [])
    }

    if isinstance(online_transfer, dict) or isinstance(cheque, dict):
        normalized["online_transfer"] = add_extraction_metadata(online_transfer or {}) if isinstance(online_transfer, dict) else {}
        normalized["cheque"] = add_extraction_metadata(cheque or {}) if isinstance(cheque, dict) else {}
        if remaining_data and not normalized["online_transfer"] and not normalized["cheque"]:
            if looks_like_online_transfer(remaining_data):
                normalized["online_transfer"] = add_extraction_metadata(remaining_data)
            elif looks_like_cheque(remaining_data):
                normalized["cheque"] = add_extraction_metadata(remaining_data)
        return normalized

    if looks_like_online_transfer(remaining_data):
        normalized["online_transfer"] = add_extraction_metadata(remaining_data)
    elif looks_like_cheque(remaining_data):
        normalized["cheque"] = add_extraction_metadata(remaining_data)

    return normalized


def is_empty_extraction(data: dict[str, Any]) -> bool:
    return not data.get("online_transfer") and not data.get("cheque")


def success_response(data: dict[str, Any] | list[Any]):
    if isinstance(data, dict) and "online_transfer" in data and "cheque" in data:
        return jsonify({"status": not is_empty_extraction(data), "data": data}), 200

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


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "success": True,
            "status": "running",
            "message": "OCR API is running. Upload images with POST /extract using multipart/form-data field 'file'.",
            "endpoints": {
                "extract": {
                    "method": "POST",
                    "path": "/extract",
                    "form_field": "file",
                    "allowed_types": sorted(ALLOWED_EXTENSIONS),
                },
                "health": {
                    "method": "GET",
                    "path": "/health",
                },
            },
        }
    ), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"success": True, "status": "healthy"}), 200


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

        messages = build_vision_messages(USER_EXTRACTION_PROMPT, base64_image, mime_type)
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

        normalized_data = normalize_extraction_response(extracted_data)
        if is_empty_extraction(normalized_data):
            app.logger.warning("First model response was empty after normalization. Raw model response: %s", raw_response)
            retry_messages = build_vision_messages(EMPTY_RETRY_PROMPT, base64_image, mime_type)
            retry_response = invoke_groq_with_fallbacks(retry_messages)
            retry_raw_response = retry_response.content if isinstance(retry_response.content, str) else json.dumps(retry_response.content)
            app.logger.warning("Retry raw model response: %s", retry_raw_response)

            try:
                retry_extracted_data = parse_json_response(retry_raw_response)
                retry_normalized_data = normalize_extraction_response(retry_extracted_data)
                if not is_empty_extraction(retry_normalized_data):
                    normalized_data = retry_normalized_data
                else:
                    app.logger.warning("Retry extraction was still empty after normalization.")
            except json.JSONDecodeError:
                app.logger.warning("Retry model response was not valid JSON.")

        return success_response(normalized_data)

    except Exception as exc:
        return error_response(str(exc), 500)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
