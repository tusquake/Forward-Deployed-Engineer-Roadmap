import os
import shutil
import uuid
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .schemas import Receipt, ExpenseRecord, Category
from .extractor import extract_receipt
from .storage import (
    init_db,
    save_expense,
    get_all_expenses,
    get_expense_by_id,
    delete_expense,
    get_stats,
    generate_csv_data,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
init_db()

app = FastAPI(
    title="Gemini & Groq Receipt & Expense Tracker",
    description="Multimodal GenAI Expense Tracking with Pydantic Validation & Fallback Guardrails",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory runtime override for API Keys
runtime_config = {
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "groq_key": os.getenv("GROQ_API_KEY", ""),
    "model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
}


@app.get("/api/config")
async def get_configuration():
    gemini_key = runtime_config["api_key"] or os.getenv("GEMINI_API_KEY", "")
    groq_key = runtime_config["groq_key"] or os.getenv("GROQ_API_KEY", "")
    has_gemini = bool(gemini_key and gemini_key.strip() not in ("your_gemini_api_key_here", ""))
    has_groq = bool(groq_key and groq_key.strip() not in ("your_groq_api_key_here", ""))
    
    return {
        "is_configured": has_gemini or has_groq,
        "has_gemini": has_gemini,
        "has_groq_fallback": has_groq,
        "model": runtime_config["model"],
        "available_models": [
            {"id": "gemini-3.8-flash", "name": "Gemini 3.8 Flash (Default / High Speed)", "provider": "Google"},
            {"id": "gemini-flash-latest", "name": "Gemini Flash Latest", "provider": "Google"},
            {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro (Deep Reasoning)", "provider": "Google"},
            {"id": "qwen/qwen3.8-27b", "name": "Groq Vision Fallback (Qwen 3.8 27B)", "provider": "Groq"}
        ],
        "categories": [c.value for c in Category]
    }


@app.post("/api/config")
async def update_configuration(
    api_key: str = Form(""),
    groq_key: str = Form(""),
    model: str = Form("gemini-3.8-flash")
):
    if api_key.strip():
        runtime_config["api_key"] = api_key.strip()
    if groq_key.strip():
        runtime_config["groq_key"] = groq_key.strip()
    if model.strip():
        runtime_config["model"] = model.strip()
    return {"status": "success", "message": "Configuration updated successfully"}


@app.get("/api/sample-receipt")
async def get_sample_receipt():
    sample_path = os.path.join(STATIC_DIR, "sample_whole_foods.png")
    if os.path.exists(sample_path):
        return {"url": "/sample_whole_foods.png", "filename": "sample_whole_foods.png"}
    return {"url": None}


@app.post("/api/extract", response_model=ExpenseRecord)
async def extract_receipt_endpoint(
    file: UploadFile = File(...),
    custom_model: Optional[str] = Form(None)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image (JPEG, PNG, WebP)")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty")

    file_ext = os.path.splitext(file.filename)[1] or ".jpg"
    safe_filename = f"{uuid.uuid4().hex}{file_ext}"
    saved_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(saved_path, "wb") as f:
        f.write(image_bytes)

    image_url = f"/uploads/{safe_filename}"

    active_gemini_key = runtime_config["api_key"] or os.getenv("GEMINI_API_KEY")
    active_groq_key = runtime_config["groq_key"] or os.getenv("GROQ_API_KEY")
    active_model = custom_model or runtime_config["model"]

    try:
        receipt, is_valid, discrepancy = extract_receipt(
            image_bytes=image_bytes,
            mime_type=file.content_type,
            gemini_key=active_gemini_key,
            groq_key=active_groq_key,
            model_name=active_model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed across Gemini and Groq: {str(e)}")

    # Store in database
    record = save_expense(
        receipt=receipt,
        arithmetic_valid=is_valid,
        discrepancy=discrepancy,
        image_url=image_url
    )
    return record


@app.get("/api/expenses")
async def list_expenses(category: Optional[str] = None, search: Optional[str] = None):
    return get_all_expenses(category=category, search=search)


@app.delete("/api/expenses/{expense_id}")
async def remove_expense(expense_id: int):
    success = delete_expense(expense_id)
    if not success:
        raise HTTPException(status_code=404, detail="Expense record not found")
    return {"status": "deleted", "id": expense_id}


@app.get("/api/stats")
async def stats_endpoint():
    return get_stats()


@app.get("/api/export")
async def export_csv_endpoint():
    csv_content = generate_csv_data()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=expenses_export.csv"}
    )


# Serve uploaded receipts
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Serve UI static assets
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Receipt & Expense Tracker on http://127.0.0.1:{port}")
    uvicorn.run("src.app:app", host="0.0.0.0", port=port, reload=True)
