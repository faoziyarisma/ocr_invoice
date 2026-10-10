"""
Skema field yang diekstrak dari dokumen invoice/receipt.

Prinsip desain:
- Setiap field punya `value` + `confidence` (0-1) + `source_snippet` (potongan teks OCR
  yang jadi dasar keputusan). Ini yang membuat model "tahu kapan ia tidak yakin".
- Jika field tidak ditemukan di dokumen, value = None dan confidence = 0.0 (abstain),
  BUKAN menebak. Validator menolak value yang tidak berasal dari teks sumber (anti-halusinasi
  dicek di extractor.py dengan grounding check, bukan di sini).
- line_items adalah tabel, karena banyak invoice punya baris produk yang harus
  dijumlahkan silang (cross-section reasoning) untuk verifikasi total.
"""
from __future__ import annotations
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class FieldValue(BaseModel):
    """Wrapper generik: setiap field hasil ekstraksi membawa confidence-nya sendiri."""
    value: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    source_snippet: Optional[str] = None  # potongan teks OCR asal nilai ini (utk audit)

    @field_validator("confidence")
    @classmethod
    def clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    line_total: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class InvoiceExtraction(BaseModel):
    """Hasil akhir ekstraksi satu dokumen invoice."""

    document_id: str
    vendor_name: FieldValue = Field(default_factory=FieldValue)
    invoice_number: FieldValue = Field(default_factory=FieldValue)
    invoice_date: FieldValue = Field(default_factory=FieldValue)
    due_date: FieldValue = Field(default_factory=FieldValue)
    bill_to: FieldValue = Field(default_factory=FieldValue)
    subtotal: FieldValue = Field(default_factory=FieldValue)
    tax_amount: FieldValue = Field(default_factory=FieldValue)
    total_amount: FieldValue = Field(default_factory=FieldValue)
    currency: FieldValue = Field(default_factory=FieldValue)
    line_items: list[LineItem] = Field(default_factory=list)

    # metadata proses — penting untuk observability produksi
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    abstained_fields: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    ocr_engine: str = "tesseract"
    llm_provider: str = "mock"
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None

    def compute_review_flag(self, threshold: float = 0.6) -> None:
        """Tandai dokumen untuk review manusia jika confidence keseluruhan rendah,
        atau jika field kritikal (total_amount) diabstain."""
        critical = ["total_amount", "invoice_number"]
        critical_missing = any(
            getattr(self, f).value is None for f in critical
        )
        self.needs_human_review = (
            self.overall_confidence < threshold or critical_missing
        )
