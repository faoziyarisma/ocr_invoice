"""
Bandingkan dua scorecard.json dan tandai regresi.

Usage: python compare_versions.py old_scorecard.json new_scorecard.json [--fail-on-regression]

Aturan regresi (bisa dipakai sebagai CI gate sebelum ship versi baru):
- macro_avg_field_accuracy turun > 2 poin persentase, ATAU
- total_hallucinations naik (berapa pun) -> selalu dianggap regresi serius, karena
  soal eksplisit: "must not hallucinate". Kita tidak toleransi kenaikan halusinasi
  sama sekali, walau accuracy keseluruhan naik.
"""
from __future__ import annotations
import argparse
import json
import sys


def compare(old: dict, new: dict) -> tuple[list[str], list[str]]:
    regressions, improvements = [], []

    old_acc = old["aggregate"]["macro_avg_field_accuracy"] or 0
    new_acc = new["aggregate"]["macro_avg_field_accuracy"] or 0
    delta = new_acc - old_acc
    if delta < -0.02:
        regressions.append(f"macro_avg_field_accuracy turun {abs(delta)*100:.1f}pp ({old_acc} -> {new_acc})")
    elif delta > 0.02:
        improvements.append(f"macro_avg_field_accuracy naik {delta*100:.1f}pp ({old_acc} -> {new_acc})")

    old_h = old["aggregate"]["total_hallucinations"]
    new_h = new["aggregate"]["total_hallucinations"]
    if new_h > old_h:
        regressions.append(f"total_hallucinations NAIK dari {old_h} ke {new_h} — REGRESI SERIUS")
    elif new_h < old_h:
        improvements.append(f"total_hallucinations turun dari {old_h} ke {new_h}")

    for field, new_stats in new["per_field_metrics"].items():
        old_stats = old["per_field_metrics"].get(field, {})
        oa = old_stats.get("accuracy_on_present_fields")
        na = new_stats.get("accuracy_on_present_fields")
        if oa is not None and na is not None and na < oa - 0.05:
            regressions.append(f"field '{field}': accuracy turun {oa} -> {na}")

    return regressions, improvements


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old_scorecard")
    ap.add_argument("new_scorecard")
    ap.add_argument("--fail-on-regression", action="store_true")
    args = ap.parse_args()

    old = json.loads(open(args.old_scorecard).read())
    new = json.loads(open(args.new_scorecard).read())

    regressions, improvements = compare(old, new)

    print("=== IMPROVEMENTS ===")
    for i in improvements:
        print(f"  + {i}")
    print("=== REGRESSIONS ===")
    for r in regressions:
        print(f"  - {r}")
    if not regressions:
        print("  (tidak ada regresi terdeteksi)")

    if regressions and args.fail_on_regression:
        sys.exit(1)


if __name__ == "__main__":
    main()
