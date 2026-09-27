import os
import shutil
import uuid
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .schemas import Receipt, ExpenseRecord, Category
from .extractor import extract_receipt_with_gemini
from .storage import (
    init_db,
    save_expense,
    get_all_expenses,
    get_expense_by_id,
    delete_expense,
    get_stats,
    generate_csv_data,
)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
init_db()

app = FastAPI(
    title="Gemini Receipt & Expense Tracker",
    description="Multimodal GenAI Expense Tracking with Pydantic Validation & Guardrails",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory runtime override for API Key (if provided via UI modal)
runtime_config = {
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
}


@app.get("/api/config")
async def get_configuration():
    key = runtime_config["api_key"] or os.getenv("GEMINI_API_KEY", "")
    is_live = bool(key and key.strip() not in ("your_gemini_api_key_here", "your_key_here", ""))
    return {
        "is_configured": is_live,
        "masked_key": f"••••{key[-4:]}" if is_live and len(key) >= 4 else "Not Configured (Demo Mode)",
        "model": runtime_config["model"],
        "categories": [c.value for c in Category]
    }


@app.post("/api/config")
async def update_configuration(api_key: str = Form(""), model: str = Form("gemini-2.5-flash")):
    if api_key.strip():
        runtime_config["api_key"] = api_key.strip()
    if model.strip():
        runtime_config["model"] = model.strip()
    return {"status": "success", "message": "Configuration updated successfully"}


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

    # Save original uploaded image
    file_ext = os.path.splitext(file.filename)[1] or ".jpg"
    safe_filename = f"{uuid.uuid4().hex}{file_ext}"
    saved_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(saved_path, "wb") as f:
        f.write(image_bytes)

    image_url = f"/uploads/{safe_filename}"

    # Extract via Gemini Multimodal pipeline
    active_key = runtime_config["api_key"] or os.getenv("GEMINI_API_KEY")
    active_model = custom_model or runtime_config["model"]

    try:
        receipt, is_valid, discrepancy = extract_receipt_with_gemini(
            image_bytes=image_bytes,
            mime_type=file.content_type,
            api_key=active_key,
            model_name=active_model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Extraction failed: {str(e)}")

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
    print(f"🚀 Starting Receipt & Expense Tracker on http://127.0.0.1:{port}")
    uvicorn.run("src.app:app", host="0.0.0.0", port=port, reload=True)
