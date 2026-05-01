"""
Dysgraphia Detection — Standalone Evaluation Script
Prints a detailed comparison of all three trained models.
Run: python evaluate.py
"""

import os
import json

METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")


def print_table(results):
    header = f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}"
    separator = "-" * 65
    print(separator)
    print(header)
    print(separator)

    best_f1 = max(r["f1_score"] for r in results)
    for r in results:
        marker = "  ← BEST" if r["f1_score"] == best_f1 else ""
        print(f"{r['name']:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1_score']:>10.4f}{marker}")
    print(separator)


def print_confusion_matrix(matrix, model_name):
    print(f"\nConfusion Matrix — {model_name}:")
    print(f"  {'':>15} Predicted Normal  Predicted Dysgraphia")
    print(f"  {'Actual Normal':>15}      {matrix[0][0]:^12}        {matrix[0][1]:^12}")
    print(f"  {'Actual Dysgraphia':>15}      {matrix[1][0]:^12}        {matrix[1][1]:^12}")


def evaluate():
    if not os.path.exists(METRICS_PATH):
        print("No metrics file found. Run train.py first.")
        from train import train
        train()

    with open(METRICS_PATH) as f:
        data = json.load(f)

    results = data["models"]
    best_name = data["best_model"]

    print("\n" + "=" * 65)
    print("  DYSGRAPHIA DETECTION — MODEL EVALUATION REPORT")
    print("=" * 65)
    print(f"\n  Dataset:  {data['dataset_size']} total samples "
          f"({data['train_size']} train / {data['test_size']} test)")
    print(f"  Best Model: {best_name}\n")

    print_table(results)

    for r in results:
        print_confusion_matrix(r["confusion_matrix"], r["name"])

    print("\n" + "=" * 65)
    print(f"  CONCLUSION: {best_name} achieved the highest F1-Score,")
    print("  making it the best classifier for dysgraphia detection.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    evaluate()
