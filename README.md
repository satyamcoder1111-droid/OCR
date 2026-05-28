# Dynamic Document OCR API

Python Flask API that uses LangChain and Groq Vision to extract dynamic structured JSON from uploaded financial or business document images.

The API does not use a fixed schema. It asks the vision model to create fields based only on visible document content.

## Requirements

- Python 3.10+
- Groq API key

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your environment file:

```bash
copy .env.example .env
```

4. Open `.env` and set your Groq API keys:

```env
GROQ_API_KEY=your_primary_groq_api_key
GROQ_API_KEY_1=your_fallback_groq_api_key_1
GROQ_API_KEY_2=your_fallback_groq_api_key_2
GROQ_API_KEY_3=your_fallback_groq_api_key_3
GROQ_API_KEY_4=your_fallback_groq_api_key_4
GROQ_API_KEY_5=your_fallback_groq_api_key_5
```

Only `GROQ_API_KEY` is required. The fallback keys are optional. The API rotates through configured keys on each request. If the selected key is rate limited, it automatically retries with the next configured key.

Keep real API keys only in `.env` or your deployment platform environment variables. `.env.example` is only a template and should not contain real secrets.

## Run The Server

Development:

```bash
python app.py
```

The API will run at:

```text
http://localhost:5000
```

Production on Linux deploy platforms:

```bash
gunicorn app:app
```

If your platform provides a `PORT` environment variable, `app.py` will use it automatically when run with `python app.py`.

## Endpoint

### `POST /extract`

Uploads an image and returns extracted JSON.

Request type:

```text
multipart/form-data
```

Form field:

```text
file
```

Allowed file types:

```text
jpg, jpeg, png, webp
```

Successful response:

```json
{
  "success": true,
  "data": {
    "document_type": "invoice",
    "raw_text": "Readable text from the image",
    "extracted_fields": {}
  }
}
```

## Curl Example

```bash
curl -X POST http://localhost:5000/extract ^
  -F "file=@C:\path\to\document.jpg"
```

On macOS or Linux:

```bash
curl -X POST http://localhost:5000/extract \
  -F "file=@/path/to/document.jpg"
```

## Postman Instructions

1. Set method to `POST`.
2. Set URL to `http://localhost:5000/extract`.
3. Open the `Body` tab.
4. Select `form-data`.
5. Add a key named `file`.
6. Change the key type from `Text` to `File`.
7. Choose a `.jpg`, `.jpeg`, `.png`, or `.webp` document image.
8. Click `Send`.

## Notes

- The returned JSON structure is dynamic and depends on the uploaded document.
- If the model returns invalid JSON, the API responds with an error and includes the raw model response for debugging.
- The API supports one primary Groq key plus five optional fallback keys.
- Uploaded images are resized and compressed before being sent to Groq to reduce latency.
- The app uses the Groq model `meta-llama/llama-4-scout-17b-16e-instruct`.
