"""Evaluate a separate, human-labeled customer-review CSV. No fabricated metrics."""
import argparse
import json
import pandas as pd
from data import read_csv, analyze_frame
from inference import SentimentEngine
from config import LABELS


def evaluate(predicted, actual):
    matrix = {label: {other: 0 for other in LABELS} for label in LABELS}
    for truth, prediction in zip(actual, predicted):
        matrix[truth][prediction] += 1
    metrics = {}
    for label in LABELS:
        tp = matrix[label][label]
        predicted_count = sum(matrix[other][label] for other in LABELS)
        actual_count = sum(matrix[label].values())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / actual_count if actual_count else 0.0
        metrics[label] = dict(precision=precision, recall=recall,
                             f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                             support=actual_count)
    return dict(rows=len(actual), accuracy=sum(matrix[v][v] for v in LABELS) / len(actual),
                macro_f1=sum(v["f1"] for v in metrics.values()) / 3,
                per_class=metrics, confusion_matrix_actual_rows_predicted_columns=matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate real predictions against independently human-labeled reviews.")
    parser.add_argument("csv")
    parser.add_argument("--review-column", default="review")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()
    from pathlib import Path
    source = read_csv(Path(args.csv).read_bytes())
    if args.review_column not in source or args.label_column not in source:
        parser.error("CSV must include the review and label columns specified.")
    actual = source[args.label_column].str.strip().str.capitalize().tolist()
    if not set(actual).issubset(set(LABELS)):
        parser.error("Human labels must be Positive, Neutral, or Negative.")
    result, prefix, _ = analyze_frame(source, args.review_column, SentimentEngine())
    if result[prefix + "sentiment"].isna().any():
        parser.error("Evaluation reviews must all be valid non-empty text within the character limit.")
    print(json.dumps(evaluate(result[prefix + "sentiment"].tolist(), actual), indent=2))
