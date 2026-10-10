"""
Client LLM yang provider-agnostic.

Kenapa penting: soal minta "keep the provider configurable (any OpenAI-compatible
endpoint)". Jadi satu class ini dipakai untuk Google AI Studio, OpenRouter, Ollama lokal,
ataupun OpenAI asli — cukup ganti base_url/model di .env.

Mode "mock": ekstraksi berbasis regex atas teks OCR, dipakai sebagai default supaya
`docker compose run --rm eval` tetap bisa jalan out-of-the-box tanpa API key (sesuai
requirement soal: "Ship sample documents and ground truth so the eval runs out of the box").
Saat API key diisi di .env, sistem otomatis pakai LLM sungguhan.
"""

from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

# EXTRACTION_SYSTEM_PROMPT = """You are a strict document-extraction engine.
# You will be given raw OCR text from an invoice/receipt. Extract ONLY fields that are
# explicitly present in the text. Never invent values.

# Return ONLY a JSON object (no markdown fences, no prose) with this exact shape:
# {
#   "vendor_name": {"value": string|null, "confidence": float 0-1, "source_snippet": string|null},
#   "invoice_number": {"value": string|null, "confidence": float 0-1, "source_snippet": string|null},
#   "invoice_date": {"value": string|null (YYYY-MM-DD if determinable), "confidence": float, "source_snippet": string|null},
#   "due_date": {"value": string|null, "confidence": float, "source_snippet": string|null},
#   "bill_to": {"value": string|null, "confidence": float, "source_snippet": string|null},
#   "subtotal": {"value": string|null, "confidence": float, "source_snippet": string|null},
#   "tax_amount": {"value": string|null, "confidence": float, "source_snippet": string|null},
#   "total_amount": {"value": string|null, "confidence": float, "source_snippet": string|null},
#   "currency": {"value": string|null, "confidence": float, "source_snippet": string|null},
#   "line_items": [{"description": string, "quantity": float|null, "unit_price": float|null, "line_total": float|null, "confidence": float}]
# }

# Rules:
# - If a field is not present or you are not confident, set value=null and confidence<=0.3. This is called "abstaining" — it is REQUIRED behavior, not a failure.
# - confidence must reflect how directly the text supports the value (1.0 = verbatim & unambiguous, 0.3 = inferred/uncertain).
# - source_snippet must be a short verbatim excerpt from the OCR text (max ~12 words) that justifies the value. If you cannot point to supporting text, you must not output that value.
# - total_amount should equal subtotal + tax_amount when both are present; if it does not reconcile, lower total_amount's confidence and note it implicitly via a lower score.
# - Do not fabricate line items that are not in the text.
# """
# """
# Client LLM yang provider-agnostic.

# Kenapa penting: soal minta "keep the provider configurable (any OpenAI-compatible
# endpoint)". Jadi satu class ini dipakai untuk Google AI Studio, OpenRouter, Ollama lokal,
# ataupun OpenAI asli — cukup ganti base_url/model di .env.

# Mode "mock": ekstraksi berbasis regex atas teks OCR, dipakai sebagai default supaya
# `docker compose run --rm eval` tetap bisa jalan out-of-the-box tanpa API key (sesuai
# requirement soal: "Ship sample documents and ground truth so the eval runs out of the box").
# Saat API key diisi di .env, sistem otomatis pakai LLM sungguhan.
# """

EXTRACTION_SYSTEM_PROMPT = """You are a strict document-extraction engine.
You will be given raw OCR text from an invoice/receipt. Please extract all variable and its value that are explicitly present in the text. Never invent values."""


@dataclass
class LLMResponse:
    raw_json: dict
    latency_ms: float
    cost_usd: float
    provider: str
    model: str


class LLMClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "mock")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        # perkiraan biaya per 1K token input/output, dipakai untuk visibilitas cost.
        # Angka ini indikatif — override via PRICE_IN_PER_1K / PRICE_OUT_PER_1K di .env.
        self.price_in = float(os.getenv("PRICE_IN_PER_1K", "0.00015"))
        self.price_out = float(os.getenv("PRICE_OUT_PER_1K", "0.0006"))

    def extract(self, ocr_text: str) -> LLMResponse:
        t0 = time.perf_counter()
        if self.provider == "mock" or not self.api_key:
            data = _mock_extract(ocr_text)
            latency = (time.perf_counter() - t0) * 1000
            return LLMResponse(
                raw_json=data,
                latency_ms=latency,
                cost_usd=0.0,
                provider="mock",
                model="regex-mock-v1",
            )
        return self._call_openai_compatible(ocr_text, t0)

    def _call_openai_compatible(self, ocr_text: str, t0: float) -> LLMResponse:
        # Import lokal supaya paket `openai` hanya dibutuhkan saat benar-benar dipakai.
        # Import dynamically so static analysis does not require the optional
        # `openai` dependency when running in mock mode.
        from importlib import import_module

        OpenAI = import_module("openai").OpenAI

        client = OpenAI(base_url=self.base_url or None, api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"OCR TEXT:\n{ocr_text}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        content = resp.choices[0].message.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = _extract_json_loose(content) or {}

        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        cost = (in_tok / 1000) * self.price_in + (out_tok / 1000) * self.price_out

        return LLMResponse(
            raw_json=data,
            latency_ms=latency_ms,
            cost_usd=cost,
            provider=self.provider,
            model=self.model,
        )


def _extract_json_loose(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Mock extractor: regex-based, tidak menebak jika pola tidak ketemu (abstain).
# Ini BUKAN pengganti LLM sungguhan — cakupannya sengaja terbatas, hanya agar
# pipeline end-to-end (OCR -> extract -> validate -> eval) bisa didemokan dan
# diuji tanpa API key/biaya.
# ---------------------------------------------------------------------------
#   Sekarang simbol multi-karakter ditulis sebagai alternation literal.
_MONEY = r"(?:(?:Rp\.?|IDR|USD|\$|€|£)\s?)?\d(?:[\d.,]*\d)?"


def _find(
    pattern: str, text: str, flags=re.IGNORECASE
) -> tuple[Optional[str], Optional[str]]:
    m = re.search(pattern, text, flags)
    if not m:
        return None, None
    val = m.group(1).strip() if m.groups() else m.group(0).strip()
    snippet = m.group(0)[:120]
    return val, snippet


def _field(value, snippet, conf_if_found=0.75):
    if value is None:
        return {"value": None, "confidence": 0.0, "source_snippet": None}
    return {"value": value, "confidence": conf_if_found, "source_snippet": snippet}


def _mock_extract(text: str) -> dict:
    # Pola di bawah ini mengikuti label PERSIS seperti pada dokumen invoice
    # "kledo" yang dikirim (bukan generalisasi ke format lain):
    #   "Invoice: INV/00006" | "Date: 20/07/2020" | "Due Date: 08/08/2020"
    #   "BILL TO ..." | "Subtotal Rp ..." | "Tax Rp ..." | "TOTAL Rp ..."
    inv_no, s1 = _find(r"invoice\s*:\s*([A-Za-z0-9\/\-]+)", text)
    inv_date, s2 = _find(
        r"(?<!due\s)date\s*:\s*([\d]{1,4}[\/\-][\d]{1,2}[\/\-][\d]{1,4})", text
    )
    due_date, s3 = _find(
        r"due\s*date\s*:\s*([\d]{1,4}[\/\-][\d]{1,2}[\/\-][\d]{1,4})", text
    )
    # (?<!sub) mencegah "total" ini ke-match di dalam kata "Subtotal"
    total, s4 = _find(rf"(?<!sub)total\s*({_MONEY})", text)
    subtotal, s5 = _find(rf"subtotal\s*({_MONEY})", text)
    tax, s6 = _find(rf"tax\s*({_MONEY})", text)
    # Vendor: kata TEPAT SEBELUM "Invoice:" (bukan dijangkar ke awal teks/^) —
    # karena urutan hasil OCR bisa meloncat (posisi blok di halaman, bukan urutan
    # baca manusia), nama vendor "kledo" bisa muncul di tengah teks, bukan di awal.
    # Hanya kata berisi huruf (bukan tanggal/angka) yang diambil, dan harus diikuti
    # literal ":" supaya tidak ke-match ke kata "Invoice" lain yang nyasar.
    vendor, s7 = _find(r"\b([A-Za-z][A-Za-z\.]{1,30})\s+Invoice\s*:", text)
    # Bill To: tangkap sampai sebelum header tabel "PRODUCT ..." pada dokumen ini.
    bill_to, s8 = _find(r"bill\s*to\s*(.+?)\s*(?=product\b)", text)
    if bill_to:
        bill_to = bill_to.strip()
    if "Rp" in text:
        currency, currency_snippet = "IDR", "Rp"
    elif "$" in text:
        currency, currency_snippet = "USD", "$"
    else:
        currency, currency_snippet = None, None

    return {
        "vendor_name": _field(vendor, s7, 0.55),
        "invoice_number": _field(inv_no, s1, 0.8),
        "invoice_date": _field(inv_date, s2, 0.7),
        "due_date": _field(due_date, s3, 0.7),
        "bill_to": _field(bill_to, s8, 0.5),
        "subtotal": _field(subtotal, s5, 0.7),
        "tax_amount": _field(tax, s6, 0.65),
        "total_amount": _field(total, s4, 0.75),
        "currency": _field(currency, currency_snippet, 0.6),
        "line_items": [],
    }
