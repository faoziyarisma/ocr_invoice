"""
Dry-run end-to-end TANPA docker/fastapi/pydantic (dipakai untuk menghasilkan
outputs/scorecard.json awal yang di-commit sbg bukti "a run you actually did",
karena sandbox pengembangan ini tidak dapat menjalankan Docker/network).

Logikanya identik dengan alur asli: api/app/extractor.py -> api/app/main.py,
hanya tanpa pembungkus Pydantic/FastAPI (dict biasa) supaya bisa jalan dengan
dependency minimal (pytesseract + Pillow saja, yang memang tersedia).

Menjalankan `docker compose up` + `docker compose run --rm eval` yang sebenarnya
akan melalui jalur HTTP asli (main.py) dan menghasilkan angka yang setara.
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "eval"))

from app.ocr import run_ocr
from app.llm_client import _mock_extract
from app.confidence import ground_and_score_field, compute_overall_confidence
from metrics import classify_field, FieldStats

FIELD_NAMES = [
    "vendor_name", "invoice_number", "invoice_date", "due_date",
    "bill_to", "subtotal", "tax_amount", "total_amount", "currency",
]

samples_dir = ROOT / "data" / "samples"
gt_dir = ROOT / "data" / "ground_truth"
out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)


def extract_document_dry(file_bytes, filename, doc_id):
    t0 = time.perf_counter()
    ocr_result = run_ocr(file_bytes, filename)
    raw = _mock_extract(ocr_result.full_text)

    fields, confs, abstained = {}, [], []
    for name in FIELD_NAMES:
        final = ground_and_score_field(raw.get(name, {}), ocr_result.full_text, ocr_result.mean_word_conf)
        fields[name] = final
        confs.append(final["confidence"])
        if final["value"] is None:
            abstained.append(name)

    overall_conf = compute_overall_confidence(confs)
    critical_missing = fields["total_amount"]["value"] is None or fields["invoice_number"]["value"] is None
    needs_review = overall_conf < 0.6 or critical_missing

    return {
        "document_id": doc_id,
        **fields,
        "overall_confidence": overall_conf,
        "abstained_fields": abstained,
        "needs_human_review": needs_review,
        "ocr_engine": "tesseract",
        "llm_provider": "mock:regex-mock-v1",
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "cost_usd": 0.0,
    }


def load_ground_truth():
    gts = {}
    for f in sorted(gt_dir.glob("*.json")):
        d = json.loads(f.read_text())
        gts[d["document_id"]] = d
    return gts


def main():
    gts = load_ground_truth()
    field_stats = {name: FieldStats() for name in FIELD_NAMES}
    per_doc_results = []
    total_latency = 0.0
    n_review = 0

    for sample_path in sorted(samples_dir.glob("*.png")):
        doc_id = sample_path.stem
        gt = gts.get(doc_id)
        if gt is None:
            continue
        pred = extract_document_dry(sample_path.read_bytes(), sample_path.name, doc_id)

        doc_field_results = {}
        for name in FIELD_NAMES:
            pf = pred[name]
            category = classify_field(pf["value"], pf["confidence"], gt.get(name))
            stats = field_stats[name]
            setattr(stats, category, getattr(stats, category) + 1)
            is_correct = category in ("exact_match", "correct_abstain")
            (stats.confidences_correct if is_correct else stats.confidences_incorrect).append(pf["confidence"])
            doc_field_results[name] = {
                "predicted": pf["value"], "confidence": pf["confidence"],
                "ground_truth": gt.get(name), "category": category,
            }

        total_latency += pred["latency_ms"]
        if pred["needs_human_review"]:
            n_review += 1

        per_doc_results.append({
            "document_id": doc_id,
            "noise_level": gt.get("noise_level"),
            "overall_confidence": pred["overall_confidence"],
            "needs_human_review": pred["needs_human_review"],
            "latency_ms": pred["latency_ms"],
            "cost_usd": pred["cost_usd"],
            "fields": doc_field_results,
        })

    n_docs = len(per_doc_results)
    scorecard = {
        "note": "DRY RUN (mock LLM, no Docker) — dihasilkan di sandbox pengembangan tanpa akses network/Docker. Jalankan `docker compose run --rm eval` untuk hasil via jalur API HTTP asli.",
        "n_documents_evaluated": n_docs,
        "n_documents_errored": 0,
        "n_documents_flagged_for_review": n_review,
        "avg_latency_ms_per_doc": round(total_latency / n_docs, 1) if n_docs else None,
        "total_cost_usd": 0.0,
        "avg_cost_usd_per_doc": 0.0,
        "per_field_metrics": {name: stats.to_dict() for name, stats in field_stats.items()},
    }
    accs = [v["accuracy_on_present_fields"] for v in scorecard["per_field_metrics"].values()
            if v["accuracy_on_present_fields"] is not None]
    total_halluc = sum(v["hallucination"] for v in scorecard["per_field_metrics"].values())
    scorecard["aggregate"] = {
        "macro_avg_field_accuracy": round(sum(accs) / len(accs), 4) if accs else None,
        "total_hallucinations": total_halluc,
    }

    (out_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2))
    (out_dir / "per_document_results.json").write_text(json.dumps(per_doc_results, indent=2))
    print(json.dumps(scorecard, indent=2))


if __name__ == "__main__":
    main()
