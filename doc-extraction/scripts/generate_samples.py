"""
Generator sample invoice sintetis + ground truth.

Ini juga berfungsi sebagai "extension" agentic-development kustom: skrip yang
dipakai berulang selama pengembangan untuk membuat dataset uji baru dengan variasi
layout/noise, tanpa perlu mencari scan asli setiap kali menambah test case.

Bukan pengganti scan dokumen dunia nyata — hanya membuat eval harness bisa
didemokan end-to-end secara reproducible (deterministic seed).
"""
from __future__ import annotations
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

random.seed(42)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

OUT_IMG = Path("/home/claude/doc-extraction/data/samples")
OUT_GT = Path("/home/claude/doc-extraction/data/ground_truth")
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_GT.mkdir(parents=True, exist_ok=True)

VENDORS = ["Nusantara Supplies Co", "Bluewave Logistics Ltd", "Green Valley Traders"]


def add_noise(img: Image.Image, level: str) -> Image.Image:
    if level == "clean":
        return img
    if level == "medium":
        img = img.rotate(random.uniform(-1.2, 1.2), fillcolor="white", expand=False)
        img = img.filter(ImageFilter.GaussianBlur(0.6))
        return img
    if level == "messy":
        img = img.rotate(random.uniform(-3, 3), fillcolor="white", expand=False)
        img = img.filter(ImageFilter.GaussianBlur(1.1))
        # tambahkan bercak noise ringan
        px = img.load()
        w, h = img.size
        for _ in range(int(w * h * 0.002)):
            x, y = random.randrange(w), random.randrange(h)
            px[x, y] = random.choice([(180, 180, 180), (60, 60, 60)])
        return img
    return img


def make_invoice(doc_id: str, noise_level: str, missing_due_date: bool, currency_symbol: str):
    W, H = 900, 1200
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 20)
    font_small = ImageFont.truetype(FONT_PATH, 16)
    font_bold = ImageFont.truetype(FONT_BOLD, 26)

    vendor = random.choice(VENDORS)
    inv_no = f"INV-{random.randint(1000,9999)}"
    inv_date = f"2026-0{random.randint(1,9)}-{random.randint(10,28)}"
    due_date = f"2026-0{random.randint(1,9)}-{random.randint(10,28)}"
    bill_to = f"PT Client {random.randint(1,50)} Indonesia"

    n_items = random.randint(2, 4)
    items = []
    subtotal = 0.0
    for i in range(n_items):
        qty = random.randint(1, 5)
        price = round(random.uniform(10, 200), 2)
        total = round(qty * price, 2)
        subtotal += total
        items.append((f"Product {chr(65+i)}", qty, price, total))
    tax = round(subtotal * 0.11, 2)
    total_amount = round(subtotal + tax, 2)

    y = 40
    d.text((40, y), vendor, font=font_bold, fill="black"); y += 50
    d.text((40, y), f"Invoice #: {inv_no}", font=font, fill="black"); y += 30
    d.text((40, y), f"Invoice Date: {inv_date}", font=font, fill="black"); y += 30
    if not missing_due_date:
        d.text((40, y), f"Due Date: {due_date}", font=font, fill="black"); y += 30
    d.text((40, y), f"Bill To: {bill_to}", font=font, fill="black"); y += 50

    d.text((40, y), "Description", font=font, fill="black")
    d.text((420, y), "Qty", font=font, fill="black")
    d.text((520, y), "Unit Price", font=font, fill="black")
    d.text((700, y), "Line Total", font=font, fill="black")
    y += 30
    d.line((40, y, 860, y), fill="black"); y += 15

    for desc, qty, price, total in items:
        d.text((40, y), desc, font=font_small, fill="black")
        d.text((420, y), str(qty), font=font_small, fill="black")
        d.text((520, y), f"{currency_symbol}{price}", font=font_small, fill="black")
        d.text((700, y), f"{currency_symbol}{total}", font=font_small, fill="black")
        y += 28

    y += 20
    d.line((500, y, 860, y), fill="black"); y += 10
    d.text((500, y), f"Subtotal: {currency_symbol}{subtotal:.2f}", font=font, fill="black"); y += 28
    d.text((500, y), f"Tax (11%): {currency_symbol}{tax:.2f}", font=font, fill="black"); y += 28
    d.text((500, y), f"Total Amount: {currency_symbol}{total_amount:.2f}", font=font_bold, fill="black"); y += 40

    img = add_noise(img, noise_level)
    img_path = OUT_IMG / f"{doc_id}.png"
    img.save(img_path)

    gt = {
        "document_id": doc_id,
        "vendor_name": vendor,
        "invoice_number": inv_no,
        "invoice_date": inv_date,
        "due_date": None if missing_due_date else due_date,
        "bill_to": bill_to,
        "subtotal": f"{subtotal:.2f}",
        "tax_amount": f"{tax:.2f}",
        "total_amount": f"{total_amount:.2f}",
        "currency": "USD" if currency_symbol == "$" else "IDR",
        "line_items": [
            {"description": d_, "quantity": q, "unit_price": p, "line_total": t}
            for d_, q, p, t in items
        ],
        "noise_level": noise_level,
    }
    (OUT_GT / f"{doc_id}.json").write_text(json.dumps(gt, indent=2))
    return doc_id


if __name__ == "__main__":
    configs = [
        ("doc_001_clean", "clean", False, "$"),
        ("doc_002_medium_noise", "medium", False, "$"),
        ("doc_003_messy_scan", "messy", False, "$"),
        ("doc_004_missing_duedate", "medium", True, "$"),
        ("doc_005_idr_currency", "clean", False, "Rp"),
        ("doc_006_messy_missing_field", "messy", True, "$"),
    ]
    for doc_id, noise, missing_due, cur in configs:
        make_invoice(doc_id, noise, missing_due, cur)
        print(f"generated {doc_id}")
