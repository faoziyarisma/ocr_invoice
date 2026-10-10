"""
Eval harness one-shot.

Dijalankan sebagai `docker compose run --rm eval` — client HTTP terpisah dari
API (soal secara eksplisit minta API tetap serving, eval sebagai command terpisah
dalam compose project yang sama).

Output:
- outputs/scorecard.json         -> ringkasan metrik per field + agregat, utk dibandingkan antar versi
- outputs/per_document_results.json -> detail per dokumen (utk debugging kegagalan)

Cara pakai untuk mendeteksi regresi antar versi (requirement "compare two versions
and catch a regression"):
    python run_eval.py --out /outputs/v1
    # ... ubah kode/prompt ...
    python run_eval.py --out /outputs/v2
    python compare_versions.py /outputs/v1/scorecard.json /outputs/v2/scorecard.json
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import requests

from metrics import classify_field, FieldStats

FIELD_NAMES = [
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "customer_name",
    "customer_email",
    "customer_phone",
    "subtotal",
    "tax_amount",
    "total_amount",
    "payment_received",
    "outstanding_amount",
    "currency",
    "payment_instruction",
    "line_items",
]


def load_ground_truth(gt_dir: Path) -> dict[str, dict]:
    gts = {}
    for f in sorted(gt_dir.glob("*.json")):
        data = json.loads(f.read_text())
        gts[data["document_id"]] = data
    return gts


def call_api(api_url: str, sample_path: Path, retries: int = 3) -> dict:
    """Panggil endpoint /extract dengan retry sederhana -> ketahanan terhadap
    kegagalan sementara (requirement 'resilience to bad output')."""
    last_err = None
    for attempt in range(retries):
        try:
            with open(sample_path, "rb") as f:
                resp = requests.post(
                    f"{api_url}/extract",
                    files={"file": (sample_path.name, f, "image/png")},
                    timeout=60,
                )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"API call failed after {retries} retries for {sample_path.name}: {last_err}"
    )


def evaluate(api_url: str, samples_dir: Path, gt_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    gts = load_ground_truth(gt_dir)

    sample_files = sorted(samples_dir.glob("*.png")) + sorted(samples_dir.glob("*.pdf"))
    if not sample_files:
        raise SystemExit(f"Tidak ada sample dokumen di {samples_dir}")

    field_stats = {name: FieldStats() for name in FIELD_NAMES}
    per_doc_results = []
    total_cost = 0.0
    total_latency = 0.0
    n_needs_review = 0
    n_errors = 0

    for sample_path in sample_files:
        doc_id = sample_path.stem
        gt = gts.get(doc_id)
        if gt is None:
            print(f"[skip] tidak ada ground truth untuk {doc_id}")
            continue

        try:
            pred = call_api(api_url, sample_path)
        except Exception as e:  # noqa: BLE001
            n_errors += 1
            per_doc_results.append({"document_id": doc_id, "error": str(e)})
            continue

        doc_field_results = {}
        for name in FIELD_NAMES:
            pred_field = pred.get(name) or {}
            pred_val = pred_field.get("value")
            pred_conf = pred_field.get("confidence", 0.0)
            gt_val = gt.get(name)

            category = classify_field(pred_val, pred_conf, gt_val)
            stats = field_stats[name]
            setattr(stats, category, getattr(stats, category) + 1)
            is_correct = category in ("exact_match", "correct_abstain")
            (
                stats.confidences_correct if is_correct else stats.confidences_incorrect
            ).append(pred_conf)

            doc_field_results[name] = {
                "predicted": pred_val,
                "confidence": pred_conf,
                "ground_truth": gt_val,
                "category": category,
            }

        total_cost += pred.get("cost_usd") or 0
        total_latency += pred.get("latency_ms") or 0
        if pred.get("needs_human_review"):
            n_needs_review += 1

        per_doc_results.append(
            {
                "document_id": doc_id,
                "noise_level": gt.get("noise_level"),
                "overall_confidence": pred.get("overall_confidence"),
                "needs_human_review": pred.get("needs_human_review"),
                "latency_ms": pred.get("latency_ms"),
                "cost_usd": pred.get("cost_usd"),
                "fields": doc_field_results,
            }
        )

    n_docs = len([r for r in per_doc_results if "error" not in r])
    scorecard = {
        "n_documents_evaluated": n_docs,
        "n_documents_errored": n_errors,
        "n_documents_flagged_for_review": n_needs_review,
        "avg_latency_ms_per_doc": round(total_latency / n_docs, 1) if n_docs else None,
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd_per_doc": round(total_cost / n_docs, 6) if n_docs else None,
        "per_field_metrics": {
            name: stats.to_dict() for name, stats in field_stats.items()
        },
    }

    # agregat lintas-field: rata-rata accuracy_on_present_fields dan total hallucination
    accs = [
        v["accuracy_on_present_fields"]
        for v in scorecard["per_field_metrics"].values()
        if v["accuracy_on_present_fields"] is not None
    ]
    total_halluc = sum(
        v["hallucination"] for v in scorecard["per_field_metrics"].values()
    )
    scorecard["aggregate"] = {
        "macro_avg_field_accuracy": round(sum(accs) / len(accs), 4) if accs else None,
        "total_hallucinations": total_halluc,
    }

    (out_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2))
    (out_dir / "per_document_results.json").write_text(
        json.dumps(per_doc_results, indent=2)
    )
    return scorecard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=os.getenv("API_URL", "http://api:8000"))
    parser.add_argument("--samples-dir", default="/data/samples")
    parser.add_argument("--gt-dir", default="/data/ground_truth")
    parser.add_argument("--out", default="/outputs")
    args = parser.parse_args()

    # tunggu API siap (compose startup race)
    for _ in range(30):
        try:
            requests.get(f"{args.api_url}/health", timeout=3)
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)

    scorecard = evaluate(
        args.api_url, Path(args.samples_dir), Path(args.gt_dir), Path(args.out)
    )
    print(json.dumps(scorecard, indent=2))


if __name__ == "__main__":
    main()
