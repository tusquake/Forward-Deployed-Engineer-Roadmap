from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Category(str, Enum):
    GROCERIES = "groceries"
    DINING = "dining"
    TRANSPORT = "transport"
    UTILITIES = "utilities"
    SHOPPING = "shopping"
    HEALTH = "health"
    ENTERTAINMENT = "entertainment"
    BUSINESS = "business"
    OTHER = "other"


class LineItem(BaseModel):
    description: str = Field(description="Description or item name on the receipt")
    quantity: float = Field(default=1.0, description="Quantity of items purchased")
    unit_price: float = Field(default=0.0, description="Unit price per item")
    line_total: float = Field(description="Total price for this line item")


class Receipt(BaseModel):
    merchant_name: str = Field(description="Name of store or merchant")
    purchase_date: str = Field(description="Purchase date in ISO YYYY-MM-DD format, or best estimate")
    line_items: List[LineItem] = Field(default_factory=list, description="Extracted individual purchased items")
    subtotal: float = Field(default=0.0, description="Pre-tax subtotal")
    tax: float = Field(default=0.0, description="Tax amount")
    total: float = Field(description="Total final bill amount")
    category: Category = Field(default=Category.OTHER, description="Primary spending category")
    currency: str = Field(default="USD", description="Currency symbol or 3-letter code (e.g. USD, EUR, INR)")
    confidence_notes: str = Field(default="", description="Any notes about blur, illegible text, or math reconciliation")


class ExpenseRecord(Receipt):
    id: Optional[int] = None
    arithmetic_valid: bool = True
    discrepancy_amount: float = 0.0
    image_url: Optional[str] = None
    created_at: Optional[str] = None
