"""
Metrik evaluasi.

Prinsip dari soal: "pick metrics that are honest about how the system fails,
not just how often". Jadi selain accuracy, kita hitung per field & per dokumen:

- exact_match          : nilai persis sama (setelah normalisasi ringan)
- correct_abstain      : field memang kosong di ground truth, sistem juga abstain -> BENAR
- hallucination        : ground truth kosong, tapi sistem mengisi nilai -> FATAL, paling mahal
- wrong_value          : ground truth ada, sistem mengisi tapi salah -> buruk, tapi beda level
                          dengan hallucination karena field-nya memang seharusnya diisi
- false_abstain        : ground truth ada, sistem malah abstain -> conservative miss,
                          lebih aman daripada wrong_value/hallucination tapi tetap kerugian recall
- calibration          : apakah confidence yang dilaporkan berkorelasi dengan benar/salah
                          (field ber-confidence tinggi seharusnya lebih sering benar)

Kita PISAHKAN hallucination dari wrong_value secara eksplisit karena keduanya
punya biaya bisnis yang sangat berbeda (soal secara eksplisit menekankan "must not
hallucinate values that are not there").
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


def _normalize(v) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = re.sub(r"[\$,\s]", "", s)  # buang simbol mata uang, koma ribuan, spasi
    s = s.rstrip(".0") if re.match(r"^\d+\.0+$", s) else s
    return s


def _numeric_close(a: str, b: str, tol: float = 0.01) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except ValueError:
        return False


def classify_field(pred_value, pred_conf: float, gt_value) -> str:
    """Klasifikasikan satu prediksi field ke salah satu kategori kegagalan/keberhasilan."""
    gt_empty = gt_value is None or str(gt_value).strip() == ""
    pred_empty = pred_value is None or pred_conf <= 0.0

    if gt_empty and pred_empty:
        return "correct_abstain"
    if gt_empty and not pred_empty:
        return "hallucination"
    if not gt_empty and pred_empty:
        return "false_abstain"

    # keduanya terisi -> bandingkan
    pn, gn = _normalize(pred_value), _normalize(gt_value)
    if pn == gn or _numeric_close(pn, gn):
        return "exact_match"
    return "wrong_value"


@dataclass
class FieldStats:
    exact_match: int = 0
    correct_abstain: int = 0
    hallucination: int = 0
    wrong_value: int = 0
    false_abstain: int = 0
    confidences_correct: list = field(default_factory=list)
    confidences_incorrect: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return (self.exact_match + self.correct_abstain + self.hallucination
                + self.wrong_value + self.false_abstain)

    @property
    def accuracy_on_present_fields(self) -> float:
        """Dari field yang SEHARUSNYA ada nilainya di ground truth, berapa persen benar.
        Ini metrik utama kualitas ekstraksi (bukan dihitung dari total termasuk abstain)."""
        denom = self.exact_match + self.wrong_value + self.false_abstain
        return round(self.exact_match / denom, 4) if denom else None

    @property
    def hallucination_rate(self) -> float:
        """Dari field yang SEHARUSNYA kosong, berapa persen malah diisi (fatal)."""
        denom = self.hallucination + self.correct_abstain
        return round(self.hallucination / denom, 4) if denom else None

    @property
    def abstention_precision(self) -> float:
        """Ketika sistem abstain, seberapa sering itu memang keputusan tepat
        (ground truth memang kosong) vs keputusan konservatif berlebihan."""
        denom = self.correct_abstain + self.false_abstain
        return round(self.correct_abstain / denom, 4) if denom else None

    def calibration_gap(self) -> float | None:
        """Selisih rata-rata confidence antara prediksi benar vs salah.
        Idealnya positif besar: field benar harus punya confidence lebih tinggi
        daripada field salah -> confidence bisa dipakai sbg sinyal untuk routing review."""
        if not self.confidences_correct or not self.confidences_incorrect:
            return None
        avg_c = sum(self.confidences_correct) / len(self.confidences_correct)
        avg_i = sum(self.confidences_incorrect) / len(self.confidences_incorrect)
        return round(avg_c - avg_i, 4)

    def to_dict(self) -> dict:
        return {
            "n": self.total,
            "exact_match": self.exact_match,
            "correct_abstain": self.correct_abstain,
            "hallucination": self.hallucination,
            "wrong_value": self.wrong_value,
            "false_abstain": self.false_abstain,
            "accuracy_on_present_fields": self.accuracy_on_present_fields,
            "hallucination_rate": self.hallucination_rate,
            "abstention_precision": self.abstention_precision,
            "confidence_calibration_gap": self.calibration_gap(),
        }
