"""
Optimisasi produksi yang diterapkan: pemrosesan batch PARALEL (ThreadPoolExecutor,
lihat api/app/main.py::extract_batch) dibanding SEKUENSIAL.

Tesseract OCR adalah operasi yang melepas GIL (C extension) selama sebagian besar
waktunya, dan panggilan LLM adalah I/O-bound (network wait) — keduanya idle-CPU-friendly
untuk threading, sehingga ThreadPoolExecutor memberi speedup nyata tanpa kompleksitas
multiprocessing.

Skrip ini mengukur dampaknya di dataset sample (6 dokumen) sebagai bukti kuantitatif.
"""
import sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
from app.ocr import run_ocr
from app.llm_client import _mock_extract

samples = sorted((ROOT / "data" / "samples").glob("*.png"))
files = [(p.read_bytes(), p.name) for p in samples]


def process_one(item):
    data, name = item
    r = run_ocr(data, name)
    _mock_extract(r.full_text)
    return True


# --- sequential ---
t0 = time.perf_counter()
for f in files:
    process_one(f)
seq_time = time.perf_counter() - t0

# --- parallel (4 workers, sama seperti main.py) ---
t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(process_one, files))
par_time = time.perf_counter() - t0

speedup = seq_time / par_time if par_time else float("inf")
print(f"Dokumen diproses      : {len(files)}")
print(f"Sekuensial            : {seq_time:.3f}s  ({seq_time/len(files)*1000:.1f} ms/doc)")
print(f"Paralel (4 workers)   : {par_time:.3f}s  ({par_time/len(files)*1000:.1f} ms/doc)")
print(f"Speedup               : {speedup:.2f}x")
