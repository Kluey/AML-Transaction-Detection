from __future__ import annotations
import pandas as pd

def fit_encoders(train_df: pd.DataFrame, freq_cols: list[str], code_cols: list[str]) -> dict:
    encoders = {"freq_maps": {}, "code_maps": {}}
    for col in freq_cols:
        if col in train_df.columns:
            encoders["freq_maps"][col] = train_df[col].value_counts(normalize=True).to_dict()
    for col in code_cols:
        if col in train_df.columns:
            categories = pd.Index(train_df[col].dropna().unique())
            encoders["code_maps"][col] = {value: code for code, value in enumerate(categories)}
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, freq_map in encoders["freq_maps"].items():
        df[col + "_freq"] = df[col].map(freq_map).fillna(0.0)
    for col, code_map in encoders["code_maps"].items():
        unknown_code = len(code_map)
        df[col + "_code"] = df[col].map(code_map).fillna(unknown_code).astype("int32")
    return df
