from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from features import build_feature_table
from encoding import fit_encoders as _fit_encoders, apply_encoders
from plotting import (
    plot_feature_importance,
    plot_precision_recall_curve,
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "model_bundle"
PLOTS_DIR = OUTPUT_DIR / "plots"
LABELS_PATH = OUTPUT_DIR / "labels.csv"

TARGET_COL = "Is Laundering"

CATEGORICAL_FREQ_COLS = [
    "Payment Currency", "Receiving Currency", "sender_country", "receiver_country",
]
CATEGORICAL_CODE_COLS = [
    "Payment Format", "sender_entity_type", "receiver_entity_type",
]
DROP_COLS = [
    "Timestamp", TARGET_COL, "row_id",
    "From Bank", "To Bank", "Sender Account", "Receiver Account",
    "sender_node", "receiver_node",
    "sender_Entity ID", "receiver_Entity ID",
    "sender_Bank Name", "receiver_Bank Name",
]

CLEAN_CUTOFF = pd.Timestamp("2022-09-11")
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
)


def restrict_to_clean_regime(feats: pd.DataFrame) -> pd.DataFrame:
    clean = feats[feats["Timestamp"] < CLEAN_CUTOFF].copy()
    logger.info(
        "Restricted to clean regime (Timestamp < %s): %s / %s rows kept",
        CLEAN_CUTOFF.date(), f"{len(clean):,}", f"{len(feats):,}",
    )
    return clean


def time_split_train_val_test(
    df: pd.DataFrame, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("Timestamp").reset_index(drop=True)
    n = len(df)
    train_cut = int(n * train_frac)
    val_cut = int(n * (train_frac + val_frac))
    return df.iloc[:train_cut].copy(), df.iloc[train_cut:val_cut].copy(), df.iloc[val_cut:].copy()


def fit_encoders(train_df: pd.DataFrame) -> dict:
    return _fit_encoders(train_df, CATEGORICAL_FREQ_COLS, CATEGORICAL_CODE_COLS)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    raw_categorical = CATEGORICAL_FREQ_COLS + CATEGORICAL_CODE_COLS
    return [
        c for c in df.columns
        if c not in DROP_COLS and c not in raw_categorical
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _scale_pos_weight(y: pd.Series) -> float:
    pos = max(1, y.sum())
    neg = max(1, len(y) - pos)
    return neg / pos


def choose_threshold(y_true: np.ndarray, proba: np.ndarray, alert_budget: float) -> float:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)

    target_count = max(1, int(len(proba) * alert_budget))
    threshold = float(np.sort(proba)[-target_count])
    preds = (proba >= threshold).astype(int)
    precision = y_true[preds == 1].mean() if preds.sum() else 0.0
    recall = (preds[y_true == 1].sum() / y_true.sum()) if y_true.sum() else 0.0
    logger.info(
        "Threshold for top %.1f%% alerts: %.6f (precision=%.4f, recall=%.4f)",
        alert_budget * 100, threshold, precision, recall,
    )
    return threshold


def _precision_recall(y_true: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    precision = y_true[preds == 1].mean() if preds.sum() else 0.0
    recall = (preds[y_true == 1].sum() / y_true.sum()) if y_true.sum() else 0.0
    return float(precision), float(recall)


def export_labels(test_df: pd.DataFrame) -> None:
    holdout = test_df[["row_id", TARGET_COL]].rename(columns={TARGET_COL: "label"})
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(LABELS_PATH, index=False)


def main(alert_budget: float, trans_path: str | Path, accounts_path: str | Path) -> None:
    t0 = time.time()

    trans_path = Path(trans_path)
    accounts_path = Path(accounts_path)
    if not trans_path.is_absolute():
        trans_path = ROOT_DIR / trans_path
    if not accounts_path.is_absolute():
        accounts_path = ROOT_DIR / accounts_path

    feats = build_feature_table(trans_path, accounts_path)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    feats = restrict_to_clean_regime(feats)

    train_df, val_df, test_df = time_split_train_val_test(feats)
    logger.info("train: %s  val: %s  test: %s", f"{len(train_df):,}", f"{len(val_df):,}", f"{len(test_df):,}")
    logger.info(
        "positive rates -- train: %.4f%%  val: %.4f%%  test: %.4f%%",
        train_df[TARGET_COL].mean() * 100, val_df[TARGET_COL].mean() * 100, test_df[TARGET_COL].mean() * 100,
    )

    encoders = fit_encoders(train_df)
    train_enc = apply_encoders(train_df, encoders)
    val_enc = apply_encoders(val_df, encoders)
    test_enc = apply_encoders(test_df, encoders)
    feature_cols = get_feature_cols(train_enc)
    logger.info("Feature count: %d", len(feature_cols))

    tuning_model = xgb.XGBClassifier(
        scale_pos_weight=_scale_pos_weight(train_enc[TARGET_COL]), **XGB_PARAMS
    )
    tuning_model.fit(train_enc[feature_cols], train_enc[TARGET_COL])
    val_proba = tuning_model.predict_proba(val_enc[feature_cols])[:, 1]
    val_y_true = val_enc[TARGET_COL].to_numpy(dtype=int)
    val_pr_auc = average_precision_score(val_y_true, val_proba)
    logger.info("Tuning-model validation PR-AUC: %.4f", val_pr_auc)
    threshold = choose_threshold(val_y_true, val_proba, alert_budget)

    plot_precision_recall_curve(
        val_y_true, val_proba, "XGBoost validation",
        PLOTS_DIR / "precision_recall_validation.png",
    )

    trainval_enc = pd.concat([train_enc, val_enc], ignore_index=True)
    model = xgb.XGBClassifier(
        scale_pos_weight=_scale_pos_weight(trainval_enc[TARGET_COL]), **XGB_PARAMS
    )
    model.fit(trainval_enc[feature_cols], trainval_enc[TARGET_COL])
    plot_feature_importance(
        model, feature_cols, "XGBoost model",
        PLOTS_DIR / "feature_importance.png",
    )

    test_proba = model.predict_proba(test_enc[feature_cols])[:, 1]
    test_preds = (test_proba >= threshold).astype(int)
    y_test = test_enc[TARGET_COL].to_numpy(dtype=int)
    test_precision, test_recall = _precision_recall(y_test, test_preds)
    test_pr_auc = float(average_precision_score(y_test, test_proba))

    logger.info("ONE-TIME TEST EVALUATION")
    logger.info("PR-AUC:  %.4f", test_pr_auc)
    logger.info("ROC-AUC: %.4f", roc_auc_score(y_test, test_proba))
    logger.info("Alerts: %s (%.3f%%)", f"{test_preds.sum():,}", test_preds.mean() * 100)
    logger.info("Precision: %.4f", test_precision)
    logger.info("Recall: %.4f", test_recall)

    export_labels(test_df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_DIR / "xgboost_model.json")
    joblib.dump(encoders, MODEL_DIR / "encoders.joblib")
    metadata = {
        "feature_cols": feature_cols,
        "threshold": threshold,
        "alert_budget": alert_budget,
        "trained_on": "Sept 1-10 2022 (clean regime, tail excluded)",
        "val_pr_auc": float(val_pr_auc),
        "test_pr_auc": test_pr_auc,
        "test_precision": test_precision,
        "test_recall": test_recall,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model bundle to %s/ ", MODEL_DIR)
    logger.info("Done in %.1fs", time.time() - t0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alert-budget", type=float, default=0.01, help="Fraction of transactions to flag")
    parser.add_argument("--trans-path", default="data/HI-Small_Trans.csv")
    parser.add_argument("--accounts-path", default="data/HI-Small_accounts.csv")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    main(args.alert_budget, args.trans_path, args.accounts_path)