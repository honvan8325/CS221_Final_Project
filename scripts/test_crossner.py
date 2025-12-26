import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import random
import numpy as np
from gliner import GLiNER


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

THRESHOLD = 0.5
BATCH_SIZE = 32

CROSSNER_ROOT = "../data/crossner"
CROSSNER_SUBSETS = ["ai", "literature", "music", "politics", "science"]

MODEL_PATHS = {
    "s": "urchade/gliner_small-v2.1",
    "m": "urchade/gliner_medium-v2.1",
    "l": "urchade/gliner_large-v2.1",
}


def bio_to_spans(tokens, tags):
    spans = []
    start, label = None, None

    for i, t in enumerate(tags):
        if t.startswith("B-"):
            if label is not None:
                spans.append((start, i - 1, label))
            start = i
            label = t[2:]
        elif t.startswith("I-"):
            continue
        else:
            if label is not None:
                spans.append((start, i - 1, label))
                start, label = None, None

    if label is not None:
        spans.append((start, len(tokens) - 1, label))

    return spans


def load_crossner_test(subset):
    path = Path(CROSSNER_ROOT) / subset / "test.txt"
    data, tokens, tags = [], [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if tokens:
                    data.append({"tokens": tokens, "spans": bio_to_spans(tokens, tags)})
                    tokens, tags = [], []
                continue

            tok, tag = line.split()
            tokens.append(tok)
            tags.append(tag)

    return data


def token_boundaries(tokens):
    bounds, pos = [], 0
    for t in tokens:
        bounds.append((pos, pos + len(t)))
        pos += len(t) + 1
    return bounds


def char_to_token(preds, boundaries):
    out = set()

    for p in preds:
        cs, ce, label = p["start"], p["end"], p["label"]
        s = e = None

        for i, (a, b) in enumerate(boundaries):
            if a <= cs < b:
                s = i
            if a < ce <= b:
                e = i

        if s is not None and e is not None and s <= e:
            out.add((s, e, label))

    return out


def run_inference(model, dataset, all_labels, desc):
    texts = [" ".join(ex["tokens"]) for ex in dataset]
    boundaries = [token_boundaries(ex["tokens"]) for ex in dataset]

    all_preds = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_bounds = boundaries[i : i + BATCH_SIZE]

        preds_batch = model.inference(
            batch_texts,
            all_labels,
            threshold=THRESHOLD,
            flat_ner=True,
            batch_size=BATCH_SIZE,
        )

        for preds, bounds in zip(preds_batch, batch_bounds):
            all_preds.append(char_to_token(preds, bounds))

    return all_preds


def compute_f1(dataset, preds):
    tp = fp = fn = 0

    for ex, pred in zip(dataset, preds):
        gold = set(ex["spans"])

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0

    return round(f1 * 100, 1)


def main():
    parser = argparse.ArgumentParser(
        description="GLiNER Zero-shot NER Evaluation on CrossNER Test Set"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="l",
        choices=["s", "m", "l"],
        help="Model size: s (small), m (medium), l (large). Default: l",
    )

    parser.add_argument(
        "--subset",
        type=str,
        nargs="*",
        default=None,
        help="Subset(s) to evaluate: ai, literature, music, politics, science. "
        "Leave empty to run all subsets.",
    )

    parser.add_argument(
        "--save-csv", action="store_true", help="Save results to CSV file"
    )

    args = parser.parse_args()

    if args.subset is None or len(args.subset) == 0 or "all" in args.subset:
        subsets_to_eval = CROSSNER_SUBSETS
    else:
        subsets_to_eval = [s.lower() for s in args.subset]
        invalid = [s for s in subsets_to_eval if s not in CROSSNER_SUBSETS]
        if invalid:
            print(f"❌ Invalid subset(s): {invalid}")
            print(f"   Valid options: {CROSSNER_SUBSETS}")
            return

    print(f"📊 Loading datasets: {subsets_to_eval}")
    datasets = {sub.capitalize(): load_crossner_test(sub) for sub in subsets_to_eval}

    label_sets = {
        name: sorted({l for ex in ds for _, _, l in ex["spans"]})
        for name, ds in datasets.items()
    }

    model_size = args.model.lower()
    model_path = MODEL_PATHS[model_size]
    model_name = f"GLiNER-{model_size.upper()}"

    print(f"\n🚀 Loading {model_name} from {model_path}")
    model = GLiNER.from_pretrained(model_path).to(DEVICE)

    row = {"Model": model_name}
    scores = []

    for dname, dset in datasets.items():
        preds = run_inference(
            model, dset, label_sets[dname], desc=f"{model_name} | {dname}"
        )

        f1 = compute_f1(dset, preds)
        row[dname] = f1
        scores.append(f1)

    row["Average"] = round(sum(scores) / len(scores), 1)

    df = pd.DataFrame([row])

    cols = (
        ["Model"]
        + sorted([c for c in df.columns if c not in ["Model", "Average"]])
        + ["Average"]
    )
    df = df[[c for c in cols if c in df.columns]]

    print("\n" + "=" * 70)
    print("📈 RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    if args.save_csv:
        output_file = f"crossner_results_{model_size}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
