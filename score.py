"""
Span-level scoring for the masking thesis. Standard library only.

Gold and prediction files are JSONL; every row has "id" and "spans":
  [{"label": "NIK", "start": 38, "end": 54, "attr": "FULL"?}]
Predictions only need id + spans (attr is not required).

Run:
  python score.py --gold data/test.jsonl --pred preds.jsonl [--report report.json]

Metrics
  exact    : label AND start/end must match exactly
  lenient  : same label and any character overlap (one-to-one matching)
  recall by attr  (PERSON FULL/PARTIAL, ADDRESS_ID FORMAL/INFORMAL)
  recall by style (formal/informal prompts)
  confusion: for every gold span that was not matched exactly, what happened
             (MISSED, BOUNDARY = right label but wrong edges, or another label)
Recall is your leakage proxy: a missed span is data that would be sent out.
"""
import argparse
import json
from collections import Counter, defaultdict


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score_rows(gold_rows, pred_rows):
    preds = {r["id"]: r["spans"] for r in pred_rows}
    exact = defaultdict(lambda: [0, 0, 0])    # tp, fp, fn per label
    lenient = defaultdict(lambda: [0, 0, 0])
    attr = defaultdict(lambda: [0, 0, 0])     # gold, exact_hit, lenient_hit
    style = defaultdict(lambda: [0, 0, 0])
    confusion = Counter()

    for row in gold_rows:
        gold, pred = row["spans"], preds.get(row["id"], [])
        gset = {(g["label"], g["start"], g["end"]) for g in gold}
        pset = {(p["label"], p["start"], p["end"]) for p in pred}
        for k in gset & pset:
            exact[k[0]][0] += 1
        for k in pset - gset:
            exact[k[0]][1] += 1
        for k in gset - pset:
            exact[k[0]][2] += 1

        used = set()
        for g in gold:
            hit = False
            for pi, p in enumerate(pred):
                if pi not in used and p["label"] == g["label"] and _overlap(g, p):
                    used.add(pi)
                    hit = True
                    break
            lenient[g["label"]][0 if hit else 2] += 1
            g["_lhit"] = hit
        for pi, p in enumerate(pred):
            if pi not in used:
                lenient[p["label"]][1] += 1

        for g in gold:
            e_hit = (g["label"], g["start"], g["end"]) in pset
            l_hit = g.pop("_lhit")
            for key, table in (((g["label"], g.get("attr")), attr),
                               (row.get("style", "?"), style)):
                if table is attr and g.get("attr") is None:
                    continue
                table[key][0] += 1
                table[key][1] += e_hit
                table[key][2] += l_hit
            if not e_hit:
                same = [p["label"] for p in pred if (p["start"], p["end"]) == (g["start"], g["end"])]
                over = [p["label"] for p in pred if _overlap(g, p)]
                if same:
                    confusion[(g["label"], same[0])] += 1
                elif g["label"] in over:
                    confusion[(g["label"], "BOUNDARY")] += 1
                elif over:
                    confusion[(g["label"], over[0])] += 1
                else:
                    confusion[(g["label"], "MISSED")] += 1

    labels = sorted(set(exact) | set(lenient))
    per_label = {}
    for l in labels:
        pe, re_, fe = prf(*exact[l])
        pl, rl, fl = prf(*lenient[l])
        per_label[l] = {"support": exact[l][0] + exact[l][2],
                        "exact": {"p": pe, "r": re_, "f1": fe},
                        "lenient": {"p": pl, "r": rl, "f1": fl}}

    def micro(table):
        tp, fp, fn = (sum(v[i] for v in table.values()) for i in range(3))
        p, r, f = prf(tp, fp, fn)
        return {"p": p, "r": r, "f1": f}

    supported = [l for l in labels if per_label[l]["support"] > 0]
    macro = {k: sum(per_label[l][k]["f1"] for l in supported) / max(len(supported), 1)
             for k in ("exact", "lenient")}
    return {
        "per_label": per_label,
        "micro": {"exact": micro(exact), "lenient": micro(lenient)},
        "macro_f1": macro,
        "recall_by_attr": {f"{k[0]}:{k[1]}": {"n": v[0], "exact": v[1] / v[0], "lenient": v[2] / v[0]}
                           for k, v in attr.items()},
        "recall_by_style": {k: {"n": v[0], "exact": v[1] / v[0], "lenient": v[2] / v[0]}
                            for k, v in style.items()},
        "confusion": {f"{a} -> {b}": n for (a, b), n in confusion.most_common()},
    }


def print_report(rep, title=""):
    print(f"\n=== {title} ===")
    print(f"{'label':<14}{'n':>6}  {'P':>6}{'R':>6}{'F1':>6}  {'F1-len':>7}")
    for l, v in rep["per_label"].items():
        print(f"{l:<14}{v['support']:>6}  {v['exact']['p']:>6.2f}{v['exact']['r']:>6.2f}"
              f"{v['exact']['f1']:>6.2f}  {v['lenient']['f1']:>7.2f}")
    m, ml = rep["micro"]["exact"], rep["micro"]["lenient"]
    print(f"{'MICRO':<14}{'':>6}  {m['p']:>6.2f}{m['r']:>6.2f}{m['f1']:>6.2f}  {ml['f1']:>7.2f}")
    print(f"{'MACRO F1':<14}{'':>6}  exact {rep['macro_f1']['exact']:.2f} | lenient {rep['macro_f1']['lenient']:.2f}")
    print("\nrecall by attr:")
    for k, v in sorted(rep["recall_by_attr"].items()):
        print(f"  {k:<26} n={v['n']:<5} exact {v['exact']:.2f}  lenient {v['lenient']:.2f}")
    print("recall by style:")
    for k, v in sorted(rep["recall_by_style"].items()):
        print(f"  {k:<26} n={v['n']:<5} exact {v['exact']:.2f}  lenient {v['lenient']:.2f}")
    print("top errors (gold -> what happened):")
    for k, n in list(rep["confusion"].items())[:12]:
        print(f"  {k:<34} {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()
    rep = score_rows(load(args.gold), load(args.pred))
    print_report(rep, args.pred)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
