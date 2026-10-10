"""
OCR layer: PDF/gambar -> teks + bounding box + skor kepercayaan per kata dari Tesseract.

Kenapa OCR + LLM (bukan VLM langsung)?
- Tesseract murah & cepat, jalan lokal tanpa API cost.
- Confidence per-kata dari Tesseract dipakai sebagai salah satu sinyal confidence
  akhir (lihat confidence.py) — LLM yang membaca teks dengan word-confidence rendah
  di sekitar sebuah field akan menurunkan confidence field itu.
- Untuk dokumen yang sangat miring/rusak, sistem ini bisa diganti VLM (llm_client.py
  sudah disiapkan provider-agnostic) tanpa mengubah schema/eval harness.
"""

from __future__ import annotations
import io
from dataclasses import dataclass, field
from typing import Optional

import pytesseract
from PIL import Image, ImageOps, ImageFilter

try:
    from pdf2image import convert_from_bytes

    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


@dataclass
class OCRWord:
    text: str
    conf: float  # 0-100 dari tesseract
    left: int
    top: int
    width: int
    height: int
    page: int


@dataclass
class OCRResult:
    full_text: str
    words: list[OCRWord] = field(default_factory=list)
    n_pages: int = 1
    mean_word_conf: float = 0.0

    def low_conf_ratio(self, threshold: float = 60.0) -> float:
        """Proporsi kata dengan confidence OCR rendah — sinyal dokumen 'kotor'."""
        if not self.words:
            return 1.0
        low = sum(1 for w in self.words if w.conf < threshold)
        return low / len(self.words)


def _preprocess(img: Image.Image) -> Image.Image:
    """Preprocessing ringan untuk scan yang miring/gelap/noisy."""
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")  # grayscale
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))  # kurangi noise scan
    return img


def _load_pages(file_bytes: bytes, filename: str) -> list[Image.Image]:
    if filename.lower().endswith(".pdf"):
        if not HAS_PDF2IMAGE:
            raise RuntimeError(
                "pdf2image/poppler tidak terpasang; install poppler-utils untuk dukungan PDF multi-halaman."
            )
        return convert_from_bytes(file_bytes, dpi=300)
    return [Image.open(io.BytesIO(file_bytes))]


def run_ocr(file_bytes: bytes, filename: str) -> OCRResult:
    pages = _load_pages(file_bytes, filename)
    all_words: list[OCRWord] = []
    texts: list[str] = []

    for page_idx, page_img in enumerate(pages):
        proc = _preprocess(page_img)
        data = pytesseract.image_to_data(
            proc,
            # lang="ind+eng",
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )
        page_text_parts = []
        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            conf_raw = data["conf"][i]
            try:
                conf = float(conf_raw)
            except (ValueError, TypeError):
                conf = -1.0
            if not word or conf < 0:
                continue
            all_words.append(
                OCRWord(
                    text=word,
                    conf=conf,
                    left=data["left"][i],
                    top=data["top"][i],
                    width=data["width"][i],
                    height=data["height"][i],
                    page=page_idx,
                )
            )
            page_text_parts.append(word)
        texts.append(" ".join(page_text_parts))

    full_text = "\n\n--- PAGE BREAK ---\n\n".join(texts)
    mean_conf = sum(w.conf for w in all_words) / len(all_words) if all_words else 0.0
    return OCRResult(
        full_text=full_text,
        words=all_words,
        n_pages=len(pages),
        mean_word_conf=mean_conf,
    )
