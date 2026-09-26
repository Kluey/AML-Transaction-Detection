from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

from features import build_feature_table
from encoding import apply_encoders

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "model_bundle"

RESULT_DISPLAY_COLS = ["Timestamp", "sender_node", "receiver_node", "Amount Paid", "Payment Format"]


def load_bundle(model_dir: Path = MODEL_DIR) -> tuple[xgb.XGBClassifier, dict, dict]:
    model_path = model_dir / "xgboost_model.json"
    encoders_path = model_dir / "encoders.joblib"
    metadata_path = model_dir / "metadata.json"
    for path in (model_path, encoders_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Model bundle file not found: {path}.")

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    encoders = joblib.load(encoders_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    return model, encoders, metadata


def score(trans_path: str | Path, accounts_path: str | Path) -> pd.DataFrame:
    trans_path = Path(trans_path)
    accounts_path = Path(accounts_path)
    if not trans_path.is_absolute():
        trans_path = ROOT_DIR / trans_path
    if not accounts_path.is_absolute():
        accounts_path = ROOT_DIR / accounts_path

    model, encoders, metadata = load_bundle()
    feature_cols = metadata["feature_cols"]
    threshold = metadata["threshold"]

    feats = build_feature_table(trans_path, accounts_path)
    feats_enc = apply_encoders(feats, encoders)

    missing = [c for c in feature_cols if c not in feats_enc.columns]
    if missing:
        raise ValueError(f"New data is missing expected feature columns: {missing}")

    proba = model.predict_proba(feats_enc[feature_cols])[:, 1]
    flagged = (proba >= threshold).astype(int)

    result = feats[RESULT_DISPLAY_COLS].copy()

    result["row_id"] = feats["row_id"]
    result["risk_score"] = proba
    result["flagged_for_review"] = flagged

    return result.sort_values("risk_score", ascending=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trans-path", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--accounts-path", default=str(DATA_DIR / "HI-Small_accounts.csv"))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "scored_transactions.csv"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    result = score(args.trans_path, args.accounts_path)

    n_flagged = result["flagged_for_review"].sum()
    logger.info(
        "Scored %s transactions -- %s flagged for review (%.2f%%)",
        f"{len(result):,}", f"{n_flagged:,}", n_flagged / len(result) * 100,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Saved full results to %s", output_path)

    print("\nTop highest-risk transactions:")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()