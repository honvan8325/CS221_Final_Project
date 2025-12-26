import argparse
import random
import torch
import pandas as pd
import requests
from pathlib import Path
from datasets import load_dataset
from datasets.features import ClassLabel, Sequence
from gliner import GLiNER


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

MODEL_PATHS = {
    "s": "urchade/gliner_small-v2.1",
    "m": "urchade/gliner_medium-v2.1",
    "l": "urchade/gliner_large-v2.1",
}

MANUAL_LABEL_MAPS = {
    "WikiANN": {
        0: "o",
        1: "b-person",
        2: "i-person",
        3: "b-organization",
        4: "i-organization",
        5: "b-location",
        6: "i-location",
    },
    "BC5CDR": {
        0: "o",
        1: "b-chemical",
        2: "b-disease",
        3: "i-disease",
        4: "i-chemical",
    },
    "BC2GM": {0: "o", 1: "b-gene", 2: "i-gene"},
    "NCBI": {0: "o", 1: "b-disease", 2: "i-disease"},
    "TweetNER7": {
        0: "b-corporation",
        1: "b-creative work",
        2: "b-event",
        3: "b-group",
        4: "b-location",
        5: "b-person",
        6: "b-product",
        7: "i-corporation",
        8: "i-creative work",
        9: "i-event",
        10: "i-group",
        11: "i-location",
        12: "i-person",
        13: "i-product",
        14: "o",
    },
    "Broad-Tweet": {
        0: "o",
        1: "b-person",
        2: "i-person",
        3: "b-organization",
        4: "i-organization",
        5: "b-location",
        6: "i-location",
    },
}

DATASETS_CONFIG = [
    {"name": "wikiann", "config": "en", "col": "ner_tags", "alias": "WikiANN"},
    {"name": "tner/bc5cdr", "col": "tags", "alias": "BC5CDR"},
    {"name": "ncbi_disease", "col": "ner_tags", "alias": "NCBI"},
    {"name": "Aunderline/genia", "alias": "GENIA"},
    {"name": "spyysalo/bc2gm_corpus", "col": "ner_tags", "alias": "BC2GM"},
    {"name": "tner/tweetner7", "col": "tags", "alias": "TweetNER7"},
]


def load_hf_dataset(name, config=None, split=None):
    kwargs = {"split": split} if split else {}
    try:
        return (
            load_dataset(name, config, revision="refs/convert/parquet", **kwargs)
            if config
            else load_dataset(name, revision="refs/convert/parquet", **kwargs)
        )
    except:
        return (
            load_dataset(name, config, **kwargs)
            if config
            else load_dataset(name, **kwargs)
        )


def get_label_map(hf_dataset, label_col, alias):
    if alias in MANUAL_LABEL_MAPS:
        return MANUAL_LABEL_MAPS[alias]

    if label_col in hf_dataset.features:
        feature = hf_dataset.features[label_col]
        if isinstance(feature, Sequence) and isinstance(feature.feature, ClassLabel):
            return {i: name.lower() for i, name in enumerate(feature.feature.names)}
        if isinstance(feature, ClassLabel):
            return {i: name.lower() for i, name in enumerate(feature.names)}
    return None


def convert_bio_to_gliner(hf_dataset, label_col, alias):
    id2label = get_label_map(hf_dataset, label_col, alias)
    if id2label is None:
        print(f"⚠️ Warning: No label map found for {alias}")
        return []

    gliner_data = []
    for ex in hf_dataset:
        tokens = ex["tokens"]
        tags = ex[label_col]
        if not tokens:
            continue

        spans, start, cur_type = [], None, None
        for i, tag_id in enumerate(tags):
            tag_str = id2label.get(tag_id, "o").lower()

            if tag_str == "o":
                if start is not None:
                    spans.append([start, i - 1, cur_type])
                    start, cur_type = None, None
                continue

            prefix, ent_type = tag_str.split("-", 1) if "-" in tag_str else ("o", None)

            if prefix == "s":
                if start is not None:
                    spans.append([start, i - 1, cur_type])
                    start, cur_type = None, None
                spans.append([i, i, ent_type])
            elif prefix == "b":
                if start is not None:
                    spans.append([start, i - 1, cur_type])
                start, cur_type = i, ent_type
            elif prefix == "i":
                if start is None or cur_type != ent_type:
                    if start is not None:
                        spans.append([start, i - 1, cur_type])
                    start, cur_type = i, ent_type
            elif prefix == "e":
                if start is not None and cur_type == ent_type:
                    spans.append([start, i, cur_type])
                else:
                    spans.append([i, i, ent_type])
                start, cur_type = None, None

        if start is not None:
            spans.append([start, len(tags) - 1, cur_type])

        gliner_data.append({"tokenized_text": tokens, "ner": spans})
    return gliner_data


def convert_genia_to_gliner(hf_dataset):
    gliner_data = []
    for ex in hf_dataset:
        spans = [
            [ent["start"], ent["end"] - 1, ent["type"].lower()]
            for ent in ex["entities"]
        ]
        gliner_data.append({"tokenized_text": ex["tokens"], "ner": spans})
    return gliner_data


def load_harveyner():
    data_dir = Path("harveyner")
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "tweets.test.bio"

    if not path.exists():
        try:
            url = "https://raw.githubusercontent.com/brickee/HarveyNER/main/data/tweets/tweets.test.bio"
            path.write_text(requests.get(url).text, encoding="utf-8")
        except Exception as e:
            print(f"❌ Failed to download HarveyNER: {e}")
            return []

    gliner_data, tokens, tags = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if tokens:
                spans, start, cur_type = [], None, None
                for i, tag in enumerate(tags):
                    if tag == "O":
                        if start is not None:
                            spans.append([start, i - 1, cur_type])
                            start, cur_type = None, None
                    elif tag.startswith("B-"):
                        if start is not None:
                            spans.append([start, i - 1, cur_type])
                        start, cur_type = i, tag[2:].lower()
                    elif tag.startswith("I-"):
                        ent_type = tag[2:].lower()
                        if cur_type != ent_type:
                            if start is not None:
                                spans.append([start, i - 1, cur_type])
                            start, cur_type = i, ent_type

                if start is not None:
                    spans.append([start, len(tags) - 1, cur_type])
                gliner_data.append({"tokenized_text": tokens, "ner": spans})
                tokens, tags = [], []
        else:
            parts = line.split()
            tokens.append(parts[0])
            tags.append(parts[-1])
    return gliner_data


def load_datasets(dataset_names=None):
    test_sets = {}

    configs_to_load = DATASETS_CONFIG
    if dataset_names and "all" not in dataset_names:
        configs_to_load = [
            cfg for cfg in DATASETS_CONFIG if cfg["alias"] in dataset_names
        ]
        if not configs_to_load:
            print(f"⚠️ No matching datasets found for: {dataset_names}")
            return test_sets

    print("⏳ Loading test datasets...\n")

    for cfg in configs_to_load:
        print(f"📂 {cfg['alias']}...", end=" ", flush=True)
        try:
            ds_test = load_hf_dataset(cfg["name"], cfg.get("config"), split="test")

            if "filter_lang" in cfg:
                ds_test = ds_test.filter(lambda x: x["lang"] == cfg["filter_lang"])

            converted = (
                convert_genia_to_gliner(ds_test)
                if cfg["alias"] == "GENIA"
                else convert_bio_to_gliner(ds_test, cfg["col"], cfg["alias"])
            )

            converted = [x for x in converted if x["ner"]]

            if len(converted) > 10000:
                converted = random.sample(converted, 10000)

            test_sets[cfg["alias"]] = converted
            print(f"✅ {len(converted)} samples")
        except Exception as e:
            print(f"❌ {e}")

    if not dataset_names or "all" in dataset_names or "HarveyNER" in dataset_names:
        harvey = load_harveyner()
        if harvey:
            test_sets["HarveyNER"] = harvey
            print(f"📂 HarveyNER... ✅ {len(harvey)} samples")

    print(f"\n✨ Loaded {len(test_sets)} datasets successfully\n")
    return test_sets


def get_token_boundaries(tokens):
    boundaries = []
    pos = 0
    for token in tokens:
        boundaries.append((pos, pos + len(token)))
        pos += len(token) + 1
    return boundaries


def align_spans_to_tokens(pred_spans, token_boundaries):
    token_spans = set()

    for pred in pred_spans:
        char_start, char_end = pred["start"], pred["end"]
        label = pred["label"].lower()

        start_idx = end_idx = -1

        for idx, (t_start, t_end) in enumerate(token_boundaries):
            if t_start <= char_start < t_end:
                start_idx = idx
                break
            if char_start == t_end and idx + 1 < len(token_boundaries):
                start_idx = idx + 1
                break

        if start_idx != -1:
            for idx in range(start_idx, len(token_boundaries)):
                t_start, t_end = token_boundaries[idx]
                if t_start < char_end <= t_end:
                    end_idx = idx
                    break

        if start_idx != -1 and end_idx != -1:
            token_spans.add((start_idx, end_idx, label))

    return token_spans


def evaluate_model(model, test_sets):
    results = []
    print(f"{'Dataset':<20} {'F1 Score':<12} Labels")
    print("-" * 60)

    for name, dataset in test_sets.items():
        if not dataset:
            continue

        labels = list(set(label for ex in dataset for _, _, label in ex["ner"]))
        if not labels:
            continue

        tp = fp = fn = 0

        for ex in dataset:
            tokens = ex["tokenized_text"]
            gold_spans = set((s, e, l) for s, e, l in ex["ner"])
            text = " ".join(tokens)

            preds = model.predict_entities(
                text, labels=labels, flat_ner=True, threshold=0.5
            )
            boundaries = get_token_boundaries(tokens)
            pred_spans = align_spans_to_tokens(preds, boundaries)

            tp += len(gold_spans & pred_spans)
            fp += len(pred_spans - gold_spans)
            fn += len(gold_spans - pred_spans)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results.append([name, round(f1 * 100, 2), len(labels)])
        print(f"{name:<20} {f1*100:>6.2f}%       {len(labels)} types")

    return pd.DataFrame(results, columns=["Dataset", "F1 Score", "Label Count"])


def main():
    parser = argparse.ArgumentParser(
        description="GLiNER Zero-shot NER Evaluation on 7 NER Benchmarks"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="l",
        choices=["s", "m", "l"],
        help="Model size: s (small), m (medium), l (large). Default: l",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        default=None,
        help="Dataset(s) to evaluate. Available: WikiANN, BC5CDR, NCBI, GENIA, "
        "BC2GM, TweetNER7, HarveyNER. ",
    )

    args = parser.parse_args()

    dataset_names = None
    if args.dataset and len(args.dataset) > 0:
        dataset_names = args.dataset

    test_sets = load_datasets(dataset_names)

    if not test_sets:
        print("❌ No datasets loaded. Exiting.")
        return

    model_size = args.model.lower()
    model_path = MODEL_PATHS[model_size]
    model_name = f"GLiNER-{model_size.upper()}"

    print(f"🚀 Loading GLiNER model: {model_name}...\n")
    model = GLiNER.from_pretrained(
        model_path, local_files_only=False, _attn_implementation="eager"
    ).to(DEVICE)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60 + "\n")

    df_results = evaluate_model(model, test_sets)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(df_results[["Dataset", "F1 Score"]].to_string(index=False))
    print(f"\nAverage F1: {df_results['F1 Score'].mean():.2f}%")
    print("=" * 60)

    if args.save_csv:
        output_file = f"7ner_results_{model_size}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
