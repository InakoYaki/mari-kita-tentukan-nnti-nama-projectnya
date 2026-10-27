"""
Regex baseline (B1) for the masking thesis. Standard library only.

  python baseline_regex.py --data data/test.jsonl --out preds_regex.jsonl           # format only
  python baseline_regex.py --data data/test.jsonl --out preds_regex_cues.jsonl --cues  # + cue words

Format-only cannot tell NIK / KK / NPWP_NIK / SIM_NIK apart (all 16 digits), so it
labels them NIK. --cues looks ~25 characters back for words like npwp / sim / kk /
rekening, which is the FAIR regex baseline to compare your model against.
Names and addresses cannot be found by format, so this baseline never predicts them.
"""
import argparse
import json
import re

CANDIDATES = [  # (label, regex, priority: lower wins ties of equal length)
    ("EMAIL", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", 0),
    ("API_KEY_TOKEN", r"(?:sk-[A-Za-z0-9]{32}|ghp_[A-Za-z0-9]{36}|AKIA[A-Z0-9]{16})", 0),
    ("AKTA_LAHIR", r"(?<!\d)\d{4}-(?:LU|LT)-\d{8}-\d{4}(?!\d)", 0),
    ("IP_ADDRESS", r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d)(?!\.\d)", 0),
    ("PHONE", r"\(\d{3}\) \d{3}-\d{4}", 0),
    ("NPWP_15", r"(?<!\d)\d{2}\.\d{3}\.\d{3}\.\d-\d{3}\.\d{3}(?!\d)", 0),
    ("SIM_LEGACY", r"(?<!\d)\d{4}[- ]\d{4}[- ]\d{6}(?!\d)", 1),
    ("SIM_LEGACY", r"(?<!\d)\d{4} \d{2} \d{2} \d{4}(?!\d)", 1),
    ("NIK", r"(?<!\d)(?:\d{4} \d{4} \d{4} \d{4}|\d{16})(?!\d)", 2),   # 16-digit family
    ("NPWP_15", r"(?<!\d)\d{15}(?!\d)", 2),
    ("SIM_LEGACY", r"(?<!\d)\d{14}(?!\d)", 2),
    ("BPJS", r"(?<!\d)\d{13}(?!\d)", 2),
    ("SIM_LEGACY", r"(?<!\d)\d{12}(?!\d)", 3),                        # ambiguous with bank account
    ("BANK_ACCOUNT", r"(?<!\d)\d{10,11}(?!\d)", 4),
    ("STNK_SERIAL", r"(?<!\d)\d{8}(?!\d)", 4),
    ("PHONE_ID", r"(?<![\d+])(?:\+62|62|0)\s?8\d{2}[\s-]?\d{3,4}[\s-]?\d{3,5}(?!\d)", 2.5),
    ("PASSPORT", r"(?<![A-Za-z0-9])[A-Z]\d{7}(?!\d)", 5),
]
CUE_SPANS = [  # cue mode only: value captured in group 1
    ("PASSWORD", re.compile(r"(?i)(?:password|pass)\s*(?:=|:|to)?\s*'?([^\s',]+)")),
    ("USERNAME", re.compile(r"(?i)(?:username|user)\s*:?\s*([\w.@]+)")),
]
COMPILED = [(lab, re.compile(rx), pr) for lab, rx, pr in CANDIDATES]


def luhn_ok(num):
    total = 0
    for i, ch in enumerate(reversed(num)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def relabel(label, text, start, end, cues):
    val = re.sub(r"\D", "", text[start:end])
    if label == "NIK" and len(val) == 16:
        if val.startswith("4") and luhn_ok(val):
            return "CREDIT_CARD"
        if cues:
            back = text[max(0, start - 25):start].lower()
            for word, lab in (("npwp", "NPWP_NIK"), (r"sim", "SIM_NIK"), ("kk", "KK")):
                if re.search(rf"\b{word}\b", back):
                    return lab
    if cues and label in ("SIM_LEGACY", "BANK_ACCOUNT", "STNK_SERIAL", "BPJS") and len(val) in (8, 10, 11, 12, 13):
        back = text[max(0, start - 25):start].lower()
        if re.search(r"\b(?:rekening|rek|akun)\b", back) and len(val) in (10, 11, 12):
            return "BANK_ACCOUNT"
    return label


def predict(text, cues=False):
    found = []
    for label, rx, pr in COMPILED:
        for m in rx.finditer(text):
            found.append((m.start(), m.end(), relabel(label, text, m.start(), m.end(), cues), pr))
    if cues:
        for label, rx in CUE_SPANS:
            for m in rx.finditer(text):
                found.append((m.start(1), m.end(1), label, -1))
    found.sort(key=lambda t: (-(t[1] - t[0]), t[3]))  # longest first, then priority
    taken, out = [], []
    for s, e, lab, _ in found:
        if all(e <= a or s >= b for a, b in taken):
            taken.append((s, e))
            out.append({"label": lab, "start": s, "end": e})
    return sorted(out, key=lambda x: x["start"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cues", action="store_true")
    args = ap.parse_args()
    with open(args.data, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as g:
        for line in f:
            row = json.loads(line)
            g.write(json.dumps({"id": row["id"], "spans": predict(row["text"], args.cues)},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
