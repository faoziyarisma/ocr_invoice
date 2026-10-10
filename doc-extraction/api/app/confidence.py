"""
Menggabungkan tiga sinyal jadi satu confidence akhir per field, dan melakukan
grounding check untuk mencegah halusinasi:

1. confidence yang dilaporkan LLM sendiri
2. kualitas OCR di sekitar area itu (mean word confidence dokumen)
3. grounding check: apakah source_snippet yang diklaim LLM benar-benar muncul
   (secara fuzzy) di teks OCR asli? Jika TIDAK, field dipaksa abstain (value=None,
   confidence=0) — ini pagar utama anti-halusinasi karena LLM tidak bisa "mengarang"
   fakta tanpa jejak yang bisa diverifikasi di teks sumber.
"""
from __future__ import annotations
from difflib import SequenceMatcher


def _fuzzy_contains(haystack: str, needle: str, threshold: float = 0.6) -> bool:
    """Cek apakah `needle` (source_snippet klaim LLM) kemungkinan besar berasal
    dari `haystack` (teks OCR asli), toleran terhadap noise OCR kecil."""
    if not needle:
        return False
    needle = needle.strip().lower()
    haystack = haystack.lower()
    if needle in haystack:
        return True
    # fallback fuzzy: cari window sepanjang needle di haystack dengan similarity ratio
    n = len(needle)
    if n == 0:
        return False
    step = max(1, n // 2)
    best = 0.0
    for i in range(0, max(1, len(haystack) - n), step):
        window = haystack[i : i + n]
        ratio = SequenceMatcher(None, window, needle).ratio()
        best = max(best, ratio)
        if best >= threshold:
            return True
    return best >= threshold


def ground_and_score_field(field_dict: dict, ocr_full_text: str, ocr_mean_conf: float) -> dict:
    """Terapkan grounding check + gabungkan sinyal confidence. Mengembalikan dict
    field yang sudah final (siap dimasukkan ke schema Pydantic)."""
    value = field_dict.get("value")
    llm_conf = float(field_dict.get("confidence") or 0.0)
    snippet = field_dict.get("source_snippet")

    if value is None:
        return {"value": None, "confidence": 0.0, "source_snippet": None}

    grounded = _fuzzy_contains(ocr_full_text, snippet or str(value))
    if not grounded:
        # LLM mengklaim sebuah nilai tapi tidak bisa dijejak balik ke teks sumber
        # -> kemungkinan halusinasi -> paksa abstain.
        return {"value": None, "confidence": 0.0, "source_snippet": None}

    # OCR quality factor: dokumen dengan OCR jelek menurunkan confidence maksimum yang
    # bisa dicapai, walau LLM-nya sendiri "yakin".
    ocr_quality_factor = max(0.4, min(1.0, ocr_mean_conf / 90.0))
    final_conf = round(llm_conf * ocr_quality_factor, 3)

    return {"value": value, "confidence": final_conf, "source_snippet": snippet}


def compute_overall_confidence(field_confidences: list[float]) -> float:
    non_null = [c for c in field_confidences if c is not None]
    if not non_null:
        return 0.0
    return round(sum(non_null) / len(non_null), 3)
