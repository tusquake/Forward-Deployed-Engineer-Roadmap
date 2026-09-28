import sqlite3
import json
import os
import io
import csv
from datetime import datetime
from typing import List, Optional, Dict, Any
from .schemas import Receipt, ExpenseRecord, LineItem, Category

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "expenses.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                merchant_name TEXT NOT NULL,
                purchase_date TEXT NOT NULL,
                category TEXT NOT NULL,
                subtotal REAL DEFAULT 0.0,
                tax REAL DEFAULT 0.0,
                total REAL NOT NULL,
                currency TEXT DEFAULT 'USD',
                line_items_json TEXT NOT NULL,
                confidence_notes TEXT,
                arithmetic_valid INTEGER DEFAULT 1,
                discrepancy_amount REAL DEFAULT 0.0,
                image_url TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def save_expense(
    receipt: Receipt,
    arithmetic_valid: bool,
    discrepancy: float,
    image_url: Optional[str] = None
) -> ExpenseRecord:
    init_db()
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    items_json = json.dumps([item.model_dump() for item in receipt.line_items])

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO expenses (
                merchant_name, purchase_date, category, subtotal, tax, total,
                currency, line_items_json, confidence_notes, arithmetic_valid,
                discrepancy_amount, image_url, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            receipt.merchant_name,
            receipt.purchase_date,
            receipt.category.value,
            receipt.subtotal,
            receipt.tax,
            receipt.total,
            receipt.currency,
            items_json,
            receipt.confidence_notes,
            1 if arithmetic_valid else 0,
            discrepancy,
            image_url,
            now_str
        ))
        conn.commit()
        record_id = cursor.lastrowid

    return ExpenseRecord(
        id=record_id,
        merchant_name=receipt.merchant_name,
        purchase_date=receipt.purchase_date,
        line_items=receipt.line_items,
        subtotal=receipt.subtotal,
        tax=receipt.tax,
        total=receipt.total,
        category=receipt.category,
        currency=receipt.currency,
        confidence_notes=receipt.confidence_notes,
        arithmetic_valid=arithmetic_valid,
        discrepancy_amount=discrepancy,
        image_url=image_url,
        created_at=now_str
    )


def row_to_record(row: sqlite3.Row) -> ExpenseRecord:
    try:
        raw_items = json.loads(row["line_items_json"])
        line_items = [LineItem(**item) for item in raw_items]
    except Exception:
        line_items = []

    try:
        cat = Category(row["category"])
    except Exception:
        cat = Category.OTHER

    return ExpenseRecord(
        id=row["id"],
        merchant_name=row["merchant_name"],
        purchase_date=row["purchase_date"],
        line_items=line_items,
        subtotal=row["subtotal"],
        tax=row["tax"],
        total=row["total"],
        category=cat,
        currency=row["currency"],
        confidence_notes=row["confidence_notes"] or "",
        arithmetic_valid=bool(row["arithmetic_valid"]),
        discrepancy_amount=row["discrepancy_amount"] or 0.0,
        image_url=row["image_url"],
        created_at=row["created_at"]
    )


def get_all_expenses(category: Optional[str] = None, search: Optional[str] = None) -> List[ExpenseRecord]:
    init_db()
    query = "SELECT * FROM expenses WHERE 1=1"
    params = []

    if category and category.lower() != "all":
        query += " AND category = ?"
        params.append(category.lower())

    if search and search.strip():
        query += " AND (merchant_name LIKE ? OR confidence_notes LIKE ?)"
        term = f"%{search.strip()}%"
        params.extend([term, term])

    query += " ORDER BY purchase_date DESC, id DESC"

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [row_to_record(r) for r in rows]


def get_expense_by_id(expense_id: int) -> Optional[ExpenseRecord]:
    init_db()
    with get_db() as conn:
        row = conn.execute("SELECT * FROM expenses WHERE id = ?", (expense_id,)).fetchone()
        if row:
            return row_to_record(row)
        return None


def delete_expense(expense_id: int) -> bool:
    init_db()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_stats() -> Dict[str, Any]:
    init_db()
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM expenses").fetchall()

    records = [row_to_record(r) for r in rows]
    total_spend = sum(r.total for r in records)
    total_count = len(records)
    flags_count = sum(1 for r in records if not r.arithmetic_valid)

    category_totals: Dict[str, float] = {}
    for r in records:
        cat = r.category.value
        category_totals[cat] = category_totals.get(cat, 0.0) + r.total

    top_category = "None"
    if category_totals:
        top_category = max(category_totals.items(), key=lambda x: x[1])[0].capitalize()

    return {
        "total_spend": round(total_spend, 2),
        "total_receipts": total_count,
        "flagged_discrepancies": flags_count,
        "top_category": top_category,
        "category_breakdown": {k: round(v, 2) for k, v in category_totals.items()}
    }


def generate_csv_data() -> str:
    records = get_all_expenses()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Purchase Date", "Merchant", "Category",
        "Subtotal", "Tax", "Total", "Currency",
        "Items Count", "Arithmetic Valid", "Discrepancy", "Confidence Notes"
    ])
    for r in records:
        writer.writerow([
            r.id, r.purchase_date, r.merchant_name, r.category.value,
            r.subtotal, r.tax, r.total, r.currency,
            len(r.line_items), r.arithmetic_valid, r.discrepancy_amount, r.confidence_notes
        ])
    return output.getvalue()
