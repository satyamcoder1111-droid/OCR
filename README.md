# Cheque And Online Transfer OCR API

Python Flask API that uses LangChain and Groq Vision to extract structured JSON from uploaded cheque, POS, or online-transfer images.

Successful OCR responses always use the same top-level structure:

```json
{
  "status": false,
  "data": {
    "online_transfer": {},
    "cheque": {},
    "pos": {}
  }
}
```

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

Optional form field:

```text
collection_by
```

Values:

```text
1 = cheque
2 = pos
3 = online_transfer
```

Allowed file types:

```text
jpg, jpeg, png, webp
```

Successful cheque response:

```json
{
  "status": true,
  "data": {
    "online_transfer": {},
    "cheque": {
      "date": "13-11-2025",
      "cheque_no": "010116",
      "payee": "RJS FOOD SERVICE SUPPLIES LLC",
      "payer": "ZABEEL FOODSTUFF TRADING LLC",
      "amount": 1706.25,
      "amount_in_words": "One Thousand Seven Hundred And Six And Fils Twenty Five Only",
      "bank_name": "RAKBANK",
      "account_no": "0542270043001",
      "iban": "AE360400000542270043001",
      "branch": "KHALIDIYA, ABU DHABI",
      "from_ocr": true,
      "extraction_method": "llm"
    },
    "pos": {}
  }
}
```

Successful online-transfer response:

```json
{
  "status": true,
  "data": {
    "online_transfer": {
      "date": "20/10/2025",
      "receipt_no": "LN57238259639058",
      "from_account": "",
      "amount": 1706.35,
      "beneficiary_name": "RUS FOODSERVICE SUPPLIES LLC",
      "beneficiary_iban": "AE320330000019100215065",
      "bank_name": "WIOBAEADXXX",
      "account_type": "",
      "from_ocr": true,
      "extraction_method": "llm"
    },
    "cheque": {},
    "pos": {}
  }
}
```

Successful POS response:

```json
{
  "status": true,
  "data": {
    "online_transfer": {},
    "cheque": {},
    "pos": {
      "merchant_name": "GOODSERVICE SUPPLIE",
      "merchant_id": "200601919038",
      "terminal_id": "13268178",
      "batch_no": "5446",
      "receipt_no": "000115",
      "date": "26-05-2026",
      "time": "14:51",
      "amount": "524.25",
      "currency": "AED",
      "card_last4": "1010",
      "card_scheme": "MASTERCARD",
      "approval_code": "688345",
      "transaction_status": "Approved",
      "payment_method": "MASTER",
      "from_ocr": true,
      "extraction_method": "llm"
    }
  }
}
```

## Curl Example

```bash
curl -X POST http://localhost:5000/extract ^
  -F "file=@C:\path\to\document.jpg" ^
  -F "collection_by=2"
```

On macOS or Linux:

```bash
curl -X POST http://localhost:5000/extract \
  -F "file=@/path/to/document.jpg" \
  -F "collection_by=2"
```

## Postman Instructions

1. Set method to `POST`.
2. Set URL to `http://localhost:5000/extract`.
3. Open the `Body` tab.
4. Select `form-data`.
5. Add a key named `file`.
6. Change the key type from `Text` to `File`.
7. Choose a `.jpg`, `.jpeg`, `.png`, or `.webp` document image.
8. Add a key named `collection_by` with value `1`, `2`, or `3`.
9. Click `Send`.

## Notes

- Successful OCR responses always contain top-level `status` and `data` keys.
- `data` always contains `online_transfer`, `cheque`, and `pos` keys.
- `status` is `true` when data is extracted and `false` when all extraction objects are empty.
- The matching object is filled based on the uploaded image, and the unused object is `{}`.
- If the model returns invalid JSON, the API responds with an error and includes the raw model response for debugging.
- The API supports one primary Groq key plus five optional fallback keys.
- Uploaded images are resized and compressed before being sent to Groq to reduce latency.
- The app uses the Groq model `meta-llama/llama-4-scout-17b-16e-instruct`.
