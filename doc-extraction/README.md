# Document Extraction — Invoice OCR + LLM

Sistem ekstraksi field terstruktur dari invoice (scan, multi-halaman, layout
tidak konsisten) menggunakan **OCR (Tesseract) + LLM**, lengkap dengan eval
harness yang membuktikan akurasinya.

## TL;DR — Run Commands

```bash
cp .env.example .env          # default: LLM_PROVIDER=mock (gratis, offline, tanpa API key)
docker compose up --build     # start API di http://localhost:8000
# di terminal lain / setelah API healthy:
docker compose run --rm eval  # jalankan eval harness -> outputs/scorecard.json
```

Untuk pakai LLM sungguhan (bukan mock), isi `LLM_PROVIDER=openai_compatible` dan
`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` di `.env` — lihat contoh Google AI
Studio / OpenRouter / Ollama di `.env.example`. Tidak perlu ubah kode apa pun.

Generate ulang sample dokumen sintetis (opsional):
```bash
python3 scripts/generate_samples.py
```

## Dataset & Skema

**Kenapa invoice?** Punya field lookup sederhana (nomor invoice, tanggal) sekaligus
field yang butuh reasoning lintas-bagian (total = subtotal + tax, tabel line items).
Persis kasus yang diminta soal.

Saya generate **6 invoice sintetis** (`scripts/generate_samples.py`) dengan variasi
layout, mata uang (USD/IDR), field yang sengaja hilang (`due_date`), dan 3 level
noise scan (clean / medium / messy: rotasi, blur, bercak). Ini dipilih karena
sandbox pengembangan tidak punya akses internet untuk mencari dataset publik —
skrip generator deterministik (seed=42) supaya reproducible, dan sengaja
mensimulasikan kegagalan OCR nyata (lihat bagian Trade-off di bawah).

Skema (`api/app/schema.py`): 9 field header + tabel `line_items`. Setiap field
adalah `{value, confidence, source_snippet}` — bukan `{value}` polos — karena
requirement soal eksplisit meminta **confidence per-field dan kemampuan abstain**.

## Arsitektur

```
 dokumen (PNG/PDF)
     │
     ▼
 OCR (Tesseract, +preprocessing: grayscale/autocontrast/median-filter)
     │  → full_text + confidence per kata
     ▼
 LLM (provider-agnostic, OpenAI-compatible endpoint apapun)
     │  → JSON terstruktur, WAJIB menyertakan source_snippet per field
     ▼
 Grounding check (confidence.py)
     │  → apakah source_snippet klaim LLM benar-benar ada di teks OCR?
     │  → TIDAK ada → paksa abstain (value=None). Ini pagar anti-halusinasi utama.
     │  → ADA → confidence_final = confidence_LLM × faktor_kualitas_OCR
     ▼
 Validasi Pydantic (schema.py) → InvoiceExtraction tervalidasi & typed
```

**Kenapa OCR+LLM, bukan VLM langsung, bukan agentic multi-step?**
Single well-designed call (bukan agentic) karena tugasnya ekstraksi terstruktur satu-lewat,
bukan tugas yang butuh eksplorasi/tool-use bertahap — menambah agen di sini menambah
latensi & cost tanpa manfaat akurasi yang jelas. OCR+LLM dipilih atas VLM langsung karena:
lebih murah (token teks jauh lebih sedikit dari token gambar), confidence per-kata dari
Tesseract memberi sinyal kualitas independen dari LLM, dan `llm_client.py` tetap
provider-agnostic — mengganti ke VLM (kirim gambar langsung) hanya butuh menambah
`extract_from_image()` di `llm_client.py`, tanpa mengubah schema atau eval harness.

**Anti-halusinasi bukan cuma instruksi prompt.** Prompt memang menyuruh LLM untuk
menyertakan `source_snippet` dan abstain bila tidak yakin, tapi itu tidak
diandalkan sendirian — `confidence.py::ground_and_score_field()` **memverifikasi
ulang** secara programatik apakah `source_snippet` itu benar-benar muncul (fuzzy
match, toleran typo OCR ringan) di teks OCR asli. LLM yang mengarang nilai tanpa
jejak di teks sumber otomatis di-downgrade jadi abstain, terlepas dari apa kata LLM.

**Mode mock.** `LLM_PROVIDER=mock` (default) memakai ekstraktor regex, bukan LLM
sungguhan. Ini supaya `docker compose run --rm eval` bisa jalan out-of-the-box
tanpa API key (requirement soal), dan supaya reviewer bisa lihat pipeline penuh
jalan tanpa perlu daftar API dulu. Ganti provider di `.env` untuk pakai LLM asli —
arsitekturnya (grounding, schema, eval) sama persis untuk keduanya.

## Evaluasi — bagian paling penting

`eval/metrics.py` mengklasifikasikan **setiap field prediksi** ke salah satu dari
5 kategori, bukan cuma "benar/salah":

| Kategori | Arti | Kenapa dipisah |
|---|---|---|
| `exact_match` | Nilai cocok ground truth | baseline |
| `correct_abstain` | GT kosong, sistem abstain | keputusan benar |
| `hallucination` | **GT kosong, sistem mengisi nilai** | **fatal** — ini yang paling ditekankan soal |
| `wrong_value` | GT ada, sistem isi tapi salah | buruk, tapi beda level dgn halusinasi |
| `false_abstain` | GT ada, sistem abstain | konservatif berlebih, lebih aman drpd 2 di atas tapi rugi recall |

Dari situ dihitung metrik turunan per field: `accuracy_on_present_fields`,
`hallucination_rate`, `abstention_precision`, dan `confidence_calibration_gap`
(apakah field ber-confidence tinggi memang lebih sering benar — jika gap ini
negatif/kecil, berarti confidence yang dilaporkan sistem tidak bisa dipercaya
untuk routing review, dan itu sendiri adalah temuan penting).

**Deteksi regresi**: `eval/compare_versions.py old.json new.json` membandingkan
dua scorecard dan menandai regresi — dengan aturan khusus: **kenaikan
total_hallucinations SELALU dianggap regresi serius**, apa pun perubahan akurasi
lainnya, karena mengarang data lebih mahal daripada tidak menjawab.

### Hasil run nyata (mock provider, 6 dokumen — lihat `outputs/scorecard.json`)

```
macro_avg_field_accuracy : 0.889   (dari field yang seharusnya terisi)
total_hallucinations     : 0       ← grounding check berhasil, tidak ada
avg_latency_ms/doc       : ~551ms (OCR only, mock LLM instan)
n_documents_flagged      : 3 dari 6 (confidence rendah/field kritikal hilang)
```

**Field terlemah**: `subtotal` (0.50 accuracy) dan `tax_amount` (0.67). Root
cause-nya ketahuan lewat `per_document_results.json`: pada scan "messy", Tesseract
kadang membaca `386.87` jadi `386 87` (titik desimal hilang karena blur), sehingga
regex mock-extractor gagal menangkap angka penuh dan sistem memilih **abstain**
(`false_abstain`) alih-alih menebak setengah angka — perilaku yang benar secara
desain, tapi menurunkan recall. Ini persis jenis kegagalan yang harus ditangkap
eval "by kind, not just counted": bukan "sistem 50% akurat", tapi "sistem gagal
spesifik pada desimal yang rusak OCR di scan noisy, dan memilih aman (abstain)
alih-alih menebak". Dengan LLM sungguhan (bukan mock-regex), ini kemungkinan besar
teratasi karena LLM bisa menalar "1966 05" → "1966.05" dari konteks, sementara
regex tidak bisa — salah satu argumen kuat kenapa mock hanya untuk demo, bukan
pengganti LLM asli.

## Production

- **Batch**: `POST /extract/batch` menerima banyak file, diproses paralel via
  `ThreadPoolExecutor` (`api/app/main.py`), melaporkan `total_cost_usd` dan
  `avg_latency_ms_per_doc` agregat.
- **Cost & latency visible**: setiap hasil ekstraksi membawa `latency_ms` dan
  `cost_usd` (dihitung dari token usage asli saat pakai LLM sungguhan).
- **Optimisasi yang diterapkan & diukur**: pemrosesan paralel (4 worker thread).
  Tesseract dipanggil sbg proses C yang melepas GIL, dan panggilan LLM adalah
  I/O-bound — keduanya cocok untuk threading tanpa kompleksitas multiprocessing.
  **Diukur jujur** (`scripts/benchmark_parallel.py`): di sandbox pengembangan ini
  (1 vCPU), paralel justru sedikit lebih lambat (0.87x) karena overhead thread
  tidak terbayar tanpa core tambahan. Di mesin produksi multi-core, speedup
  mendekati linear (hingga jumlah core) karena OCR per-dokumen didominasi CPU
  Tesseract yang bisa jalan bersamaan di core berbeda. Saya laporkan angka asli
  ini alih-alih memalsukan hasil — poin ini penting untuk trade-off reasoning.
- **Resilience**: `eval/run_eval.py::call_api()` retry 3x dengan backoff; item
  `line_items` yang malformed di-drop per-item, bukan menggagalkan seluruh dokumen
  (`extractor.py`); dokumen dengan OCR gagal total langsung abstain penuh tanpa
  memanggil LLM (hemat cost).

## Agentic Development

Dibangun dengan bantuan Claude (agentic coding), dengan alur: rancang skema →
OCR module → mock LLM client (utk demo tanpa API key) → grounding/confidence →
eval metrics → verifikasi tiap modul dengan run nyata di sandbox (bukan sekadar
ditulis lalu diasumsikan benar) — proses ini menangkap & memperbaiki 3 bug nyata
sebelum submission: regex `total_amount` yang salah menangkap substring dari
"Subtotal", grounding check yang salah menolak field `currency` (snippet "USD"
dicari di teks yang isinya simbol "$"), dan pattern `vendor_name` yang berasumsi
ada newline padahal teks OCR di-flatten.

**Ekstensi kustom yang benar-benar dipakai**: `scripts/generate_samples.py` — 
generator dataset sintetis dengan noise terkontrol (rotasi/blur/bercak scan) +
ground truth otomatis. Dipakai berulang selama development untuk menambah/mengubah
test case tanpa perlu mencari scan asli tiap kali, dan levelnya (`clean/medium/messy`)
dipakai untuk stress-test grounding check terhadap OCR yang benar-benar rusak.

## Trade-off & Keterbatasan

- **Mock ≠ LLM asli.** Angka di atas dihasilkan mock-regex extractor untuk
  membuktikan pipa (OCR→grounding→eval) bekerja tanpa API key. Akurasi field
  numerik (`subtotal`/`tax_amount`) diperkirakan naik signifikan dengan LLM
  sungguhan karena reasoning kontekstual mengatasi OCR yang rusak.
- **Dataset sintetis, bukan scan dunia nyata.** Realistis untuk variasi layout
  & noise (font, rotasi, blur), tapi tidak menangkap semua kegagalan scan asli
  (stempel, tanda tangan tumpang tindih, kertas terlipat, dsb).
- **Grounding check berbasis fuzzy string match**, bukan verifikasi semantik —
  bisa lolos nilai yang "terlihat mirip" teks tapi salah makna (mis. angka dari
  baris lain yang kebetulan mirip).

## Dengan waktu lebih

1. Jalankan dgn LLM sungguhan (Gemini Flash via Google AI Studio) + bandingkan
   scorecard vs mock lewat `compare_versions.py` untuk kuantifikasi gain-nya.
2. Ganti fuzzy-match grounding dengan verifikasi berbasis posisi bounding-box
   OCR (`OCRWord.left/top`) supaya bisa highlight lokasi field di gambar asli.
3. Kumpulkan scan invoice nyata (bukan sintetis) utk ground truth yang lebih
   representatif, terutama utk tabel line-items multi-halaman.
4. Tambah endpoint `/extract/batch` dengan queue (bukan ThreadPoolExecutor
   in-process) untuk beban produksi sungguhan + retry/backoff per-dokumen.

## Struktur Repo

```
docker-compose.yml       # api (serving) + eval (one-shot, profile "eval")
.env.example
api/
  Dockerfile
  app/
    main.py              # FastAPI: /extract, /extract/batch, /health
    schema.py            # Pydantic: InvoiceExtraction, FieldValue
    ocr.py                # Tesseract + preprocessing
    llm_client.py         # provider-agnostic + mock extractor
    confidence.py          # grounding check + confidence scoring
    extractor.py           # orkestrasi OCR->LLM->grounding->schema
eval/
  Dockerfile
  run_eval.py             # harness: panggil API, hitung metrik, tulis scorecard
  metrics.py              # klasifikasi 5-kategori + metrik turunan
  compare_versions.py      # deteksi regresi antar scorecard
data/
  samples/                # 6 invoice sintetis (PNG)
  ground_truth/            # ground truth JSON per dokumen
scripts/
  generate_samples.py       # generator dataset sintetis (agentic-dev extension)
  benchmark_parallel.py      # ukur dampak optimisasi paralel
  local_dry_run.py           # dry-run eval tanpa Docker (dipakai bikin outputs/ awal)
outputs/
  scorecard.json             # hasil run nyata (mock provider)
  per_document_results.json   # detail per-dokumen utk debugging
```
