"""
Fine-tune a transformer for token classification on the masking taxonomy.

  # 1) smoke test (a few minutes, checks everything runs)
  python train_ner.py --data data --model xlm-roberta-base --out runs/smoke --epochs 1 --limit 200
  # 2) full run
  python train_ner.py --data data --model xlm-roberta-base --out runs/xlmr --epochs 5
  # 3) comparison model (needs a FAST tokenizer; the script checks)
  python train_ner.py --data data --model indobenchmark/indobert-base-p1 --out runs/indobert

Needs: pip install transformers accelerate   (torch is preinstalled on Colab)

Pipeline: JSONL char spans -> BIO tags per token (via tokenizer offsets) -> Trainer ->
predictions decoded back to character spans -> scored with score.py (exact + lenient).
Only the pure functions at the top are imported lazily-safe; heavy imports are inside main().

v2 changes (after the first real run): the best epoch is chosen by dev F1 (not dev loss),
the best weights are kept in memory, and no checkpoints are written during training
(the first run spent ~25 of its 31 minutes writing 1.1 GB checkpoints to Drive).
"""
import argparse
import inspect
import json
import os


# ---------------------------------------------------------------- pure functions
def load_rows(path, limit=None):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def align_labels(offsets, spans, label2id):
    """Per-token label ids from character spans. Special tokens (offset s==e) -> -100."""
    labels, started = [], set()
    for s, e in offsets:
        if s == e:
            labels.append(-100)
            continue
        tag = "O"
        for idx, sp in enumerate(spans):
            if s < sp["end"] and e > sp["start"]:
                tag = f"{'I' if idx in started else 'B'}-{sp['label']}"
                started.add(idx)
                break
        labels.append(label2id[tag])
    return labels


def decode(text, offsets, pred_ids, id2label):
    """Token tag ids -> character spans (whitespace trimmed at the edges)."""
    spans, cur = [], None
    for (s, e), pid in zip(offsets, pred_ids):
        if s == e:
            continue
        tag = id2label[int(pid)]
        if tag == "O":
            if cur:
                spans.append(cur)
                cur = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or cur is None or cur["label"] != label:
            if cur:
                spans.append(cur)
            cur = {"label": label, "start": s, "end": e}
        else:
            cur["end"] = e
    if cur:
        spans.append(cur)
    out = []
    for sp in spans:
        a, b = sp["start"], sp["end"]
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            out.append({"label": sp["label"], "start": a, "end": b})
    return out


# ------------------------------------------------------------------------ main
def main():
    import torch
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, Trainer, TrainerCallback,
                              TrainingArguments, set_seed)
    from score import print_report, score_rows

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--out", default="runs/xlmr")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="use only N rows per split (smoke test)")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tags = json.load(open(os.path.join(args.data, "labels.json"), encoding="utf-8"))["bio_tags"]
    label2id = {t: i for i, t in enumerate(tags)}
    id2label = {i: t for t, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tok.is_fast:
        raise SystemExit("This model has no fast tokenizer, so character offsets are unavailable. "
                         "Pick another model.")

    def encode(rows):
        enc = tok([r["text"] for r in rows], truncation=True, max_length=args.max_len,
                  return_offsets_mapping=True)
        items, offsets = [], []
        for i, r in enumerate(rows):
            off = enc["offset_mapping"][i]
            items.append({"input_ids": enc["input_ids"][i],
                          "attention_mask": enc["attention_mask"][i],
                          "labels": align_labels(off, r["spans"], label2id)})
            offsets.append(off)
        return items, offsets

    class DS(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    rows = {s: load_rows(os.path.join(args.data, f"{s}.jsonl"), args.limit)
            for s in ("train", "dev", "test")}
    enc = {s: encode(rows[s]) for s in rows}

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(tags), id2label=id2label, label2id=label2id)

    targs = TrainingArguments(
        output_dir=os.path.join(args.out, "tmp"), learning_rate=args.lr,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs * 2,
        num_train_epochs=args.epochs, weight_decay=0.01, save_strategy="no",
        logging_steps=50, report_to="none", seed=args.seed, fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=targs, train_dataset=DS(enc["train"][0]),
                      data_collator=DataCollatorForTokenClassification(tok))

    def predict_split(split):
        items, offsets = enc[split]
        logits = trainer.predict(DS(items)).predictions
        logits = logits[0] if isinstance(logits, tuple) else logits
        pred_ids = logits.argmax(-1)
        return [{"id": r["id"],
                 "spans": decode(r["text"], offsets[i], pred_ids[i][:len(offsets[i])], id2label)}
                for i, r in enumerate(rows[split])]

    best = {"f1": -1.0, "epoch": 0, "state": None}

    class DevF1(TrainerCallback):
        def on_epoch_end(self, targs_, state, control, **kwargs):
            rep = score_rows(rows["dev"], predict_split("dev"))
            f1, lf1 = rep["micro"]["exact"]["f1"], rep["micro"]["lenient"]["f1"]
            print(f"\n[epoch {state.epoch:.0f}] dev micro F1: exact {f1:.3f} | lenient {lf1:.3f}")
            if f1 > best["f1"]:
                best.update(f1=f1, epoch=int(round(state.epoch)),
                            state={k: v.detach().cpu().clone()
                                   for k, v in trainer.model.state_dict().items()})

    trainer.add_callback(DevF1())
    trainer.train()
    if best["state"] is not None:
        trainer.model.load_state_dict(best["state"])
        print(f"\nUsing best epoch {best['epoch']} (dev micro F1 exact {best['f1']:.3f})")
    trainer.save_model(os.path.join(args.out, "best"))
    tok.save_pretrained(os.path.join(args.out, "best"))

    for split in ("dev", "test"):
        preds = predict_split(split)
        with open(os.path.join(args.out, f"preds_{split}.jsonl"), "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        rep = score_rows(rows[split], preds)
        print_report(rep, f"{args.model} on {split}")
        with open(os.path.join(args.out, f"report_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
