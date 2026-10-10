from __future__ import annotations

import json
import time
from typing import Any

from .ocr import run_ocr
from .llm_client import LLMClient
from .confidence import (
    ground_and_score_field,
    compute_overall_confidence,
)
from .schema import InvoiceExtraction, FieldValue, LineItem

FIELD_NAMES = [
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "bill_to",
    "subtotal",
    "tax_amount",
    "total_amount",
    "currency",
]

_llm = LLMClient()


def extract_document(
    file_bytes: bytes,
    filename: str,
    document_id: str,
) -> InvoiceExtraction:
    """
    Extract invoice information from a document.

    Pipeline:
        1. OCR
        2. Abstain if OCR returns no text
        3. LLM extraction
        4. Ground and score extracted fields
        5. Parse line items
        6. Calculate confidence
        7. Determine human-review requirement
    """

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. OCR
    # ------------------------------------------------------------------
    ocr_result = run_ocr(file_bytes, filename)

    full_text = (ocr_result.full_text or "").strip()
    print(f"\n=== OCR TEXT [{document_id}] ===\n{full_text or '<empty>'}\n", flush=True)

    # Empty document / OCR completely failed.
    # Abstain from all fields and don't call the LLM.
    if not full_text:
        result = InvoiceExtraction(document_id=document_id)

        result.abstained_fields = FIELD_NAMES.copy()
        result.needs_human_review = True
        result.ocr_engine = "tesseract"
        result.latency_ms = round(
            (time.perf_counter() - t0) * 1000,
            1,
        )
        print(
            f"=== FINAL EXTRACTED FIELDS [{document_id}] ===\n"
            f"{json.dumps(result.model_dump(), ensure_ascii=False, indent=2)}",
            flush=True,
        )

        return result

    # ------------------------------------------------------------------
    # 2. LLM extraction
    # ------------------------------------------------------------------
    llm_resp = _llm.extract(full_text)
    print(
        f"=== RAW LLM FIELD EXTRACTION [{document_id}] ===\n"
        f"{json.dumps(llm_resp.raw_json, ensure_ascii=False, indent=2)}",
        flush=True,
    )

    # Make sure raw_json is always a dictionary.
    raw: dict[str, Any] = llm_resp.raw_json or {}

    if not isinstance(raw, dict):
        raw = {}

    # ------------------------------------------------------------------
    # 3. Extract and score fields
    # ------------------------------------------------------------------
    kwargs: dict[str, Any] = {
        "document_id": document_id,
    }

    abstained: list[str] = []
    confidences: list[float] = []

    mean_word_conf = getattr(
        ocr_result,
        "mean_word_conf",
        0.0,
    )

    for name in FIELD_NAMES:
        raw_field = raw.get(name)

        # LLM may return:
        #   null
        #   string
        #   malformed object
        # Normalize it before passing to confidence logic.
        if not isinstance(raw_field, dict):
            raw_field = {}

        try:
            final = ground_and_score_field(
                raw_field,
                full_text,
                mean_word_conf,
            )
        except Exception:
            # If confidence calculation fails for one field,
            # abstain only that field instead of failing the document.
            final = {
                "value": None,
                "confidence": 0.0,
            }

        # Protect against malformed output from ground_and_score_field.
        value = final.get("value")
        confidence = final.get("confidence", 0.0)

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        field_data = {
            **final,
            "confidence": confidence,
        }

        try:
            kwargs[name] = FieldValue(**field_data)
        except Exception:
            # Invalid field schema -> abstain this field.
            kwargs[name] = FieldValue(
                value=None,
                confidence=0.0,
            )

            value = None
            confidence = 0.0

        confidences.append(confidence)

        if value is None:
            abstained.append(name)

    # ------------------------------------------------------------------
    # 4. Parse line items
    # ------------------------------------------------------------------
    line_items: list[LineItem] = []

    raw_line_items = raw.get("line_items", [])

    if isinstance(raw_line_items, list):
        for li in raw_line_items:
            if not isinstance(li, dict):
                continue

            try:
                line_items.append(LineItem(**li))
            except Exception:
                # Malformed line item should not fail the entire document.
                continue

    # ------------------------------------------------------------------
    # 5. Build final result
    # ------------------------------------------------------------------
    result = InvoiceExtraction(**kwargs)

    result.line_items = line_items
    result.abstained_fields = abstained

    # ------------------------------------------------------------------
    # 6. Overall confidence
    # ------------------------------------------------------------------
    result.overall_confidence = compute_overall_confidence(confidences)

    # ------------------------------------------------------------------
    # 7. Review flag
    # ------------------------------------------------------------------
    result.compute_review_flag()

    # ------------------------------------------------------------------
    # 8. Metadata
    # ------------------------------------------------------------------
    result.ocr_engine = "tesseract"

    provider = getattr(llm_resp, "provider", "unknown")
    model = getattr(llm_resp, "model", "unknown")

    result.llm_provider = f"{provider}:{model}"

    result.latency_ms = round(
        (time.perf_counter() - t0) * 1000,
        1,
    )

    cost_usd = getattr(llm_resp, "cost_usd", 0.0)

    try:
        result.cost_usd = round(float(cost_usd), 6)
    except (TypeError, ValueError):
        result.cost_usd = 0.0

    extracted_fields = {
        name: getattr(result, name).model_dump() for name in FIELD_NAMES
    }
    extracted_fields["line_items"] = [item.model_dump() for item in result.line_items]
    print(
        f"=== FINAL EXTRACTED FIELDS [{document_id}] ===\n"
        f"{json.dumps(extracted_fields, ensure_ascii=False, indent=2)}",
        flush=True,
    )

    return result
