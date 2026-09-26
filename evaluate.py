from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate(scored_path: str | Path, labels_path: str | Path) -> None:
    scored_path = Path(scored_path)
    labels_path = Path(labels_path)

    scored = pd.read_csv(scored_path)
    labels = pd.read_csv(labels_path)
    
    missing_from_scored = set(labels["row_id"]) - set(scored["row_id"])
    if missing_from_scored:
        raise ValueError(
            f"{len(missing_from_scored):,} row_ids in {labels_path} have no "
            f"matching row in {scored_path} -- was scored_transactions.csv "
            f"regenerated from a different transactions file?"
        )

    results = labels.merge(scored, on="row_id", how="left", validate="one_to_one")

    results = results.rename(columns={"label": "actual_label"})
    results["actual_label"] = results["actual_label"].astype(int)

    if "Timestamp" in results.columns:
        ts = pd.to_datetime(results["Timestamp"])
        print(f"Evaluating on {len(results):,} rows spanning "
              f"{ts.min()} -> {ts.max()}")
        print("(Check this range looks like what you intended -- e.g. a")
        print(" genuine held-out test window, not the whole dataset.)\n")

    y_true = results["actual_label"]
    y_score = results["risk_score"]
    y_pred = results["flagged_for_review"].astype(int)

    pr_auc = average_precision_score(y_true, y_score)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    alert_rate = y_pred.mean()

    print("\nAML Model Evaluation")
    print(f"Transactions: {len(results):,}")
    print(f"Actual laundering: {y_true.sum():,}")
    print(f"Flagged: {y_pred.sum():,}")
    print(f"Alert rate: {alert_rate:.2%}")
    print()

    print("Metrics")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print()

    print("Confusion Matrix")
    print(f"True negatives: {tn:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True positives: {tp:,}")

    results["result"] = "TN"
    results.loc[
        (results["actual_label"] == 0)
        & (results["flagged_for_review"] == 1),
        "result",
    ] = "FP"

    results.loc[
        (results["actual_label"] == 1)
        & (results["flagged_for_review"] == 0),
        "result",
    ] = "FN"

    results.loc[
        (results["actual_label"] == 1)
        & (results["flagged_for_review"] == 1),
        "result",
    ] = "TP"

    print("\nPrediction Breakdown")
    print(results["result"].value_counts().to_string())

    print("\nHighest-risk transactions:")
    print(
        results.sort_values("risk_score", ascending=False)[
            [
                "Timestamp",
                "sender_node",
                "receiver_node",
                "Amount Paid",
                "risk_score",
                "flagged_for_review",
                "actual_label",
                "result",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", default="outputs/scored_transactions.csv",)
    parser.add_argument("--labels", default="outputs/labels.csv",)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluate(
        args.scored,
        args.labels,
    )


if __name__ == "__main__":
    main()