from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

ENTITY_TYPE_KEYWORDS = ("Sole Proprietorship", "Corporation", "Partnership", "Individual")
BANK_NAME_COUNTRY_RE = re.compile(r"^([A-Za-z ]+?) Bank")
DEFAULT_COUNTRY = "USA"

NEAR_THRESHOLD_LOW = 9000
NEAR_THRESHOLD_HIGH = 9999.99
NIGHT_HOUR_START = 0
NIGHT_HOUR_END = 5

PAGERANK_MAX_ROWS = 100_000 
PAGERANK_RECOMPUTE_FRACTION = 10 
CYCLE_HOP_LIMIT = 4 

def load_transactions(path: str | Path) -> pd.DataFrame:
    logger.info("Loading transactions from %s", path)
    df = pd.read_csv(path)

    cols = list(df.columns)
    account_col_idxs = [i for i, c in enumerate(cols) if c.split(".")[0] == "Account"]
    if len(account_col_idxs) == 2:
        cols[account_col_idxs[0]] = "Sender Account"
        cols[account_col_idxs[1]] = "Receiver Account"
        df.columns = cols

    df["row_id"] = df.index
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    df["sender_node"] = df["From Bank"].astype(str) + "_" + df["Sender Account"].astype(str)
    df["receiver_node"] = df["To Bank"].astype(str) + "_" + df["Receiver Account"].astype(str)
    return df


def load_accounts(path: str | Path) -> pd.DataFrame:
    logger.info("Loading accounts from %s", path)
    accounts = pd.read_csv(path)
    accounts["node"] = accounts["Bank ID"].astype(str) + "_" + accounts["Account Number"].astype(str)
    accounts["entity_type"] = accounts["Entity Name"].apply(_parse_entity_type)
    return accounts


def _parse_entity_type(name: object) -> str:
    if not isinstance(name, str):
        return "Unknown"
    lowered = name.lower()
    for kind in ENTITY_TYPE_KEYWORDS:
        if kind.lower() in lowered:
            return kind
    return "Other"


def _parse_country(bank_name: object) -> str:
    if not isinstance(bank_name, str):
        return "Unknown"
    match = BANK_NAME_COUNTRY_RE.match(bank_name)
    return match.group(1).strip() if match else DEFAULT_COUNTRY


def add_edge_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding edge features")
    df = df.copy()

    df["log_amount_paid"] = np.log1p(df["Amount Paid"])
    df["log_amount_received"] = np.log1p(df["Amount Received"])
    df["amount_diff"] = df["Amount Paid"] - df["Amount Received"]
    df["amount_ratio"] = (
        df["Amount Received"] / df["Amount Paid"].replace(0, np.nan)
    ).fillna(1.0)

    df["currency_mismatch"] = (df["Payment Currency"] != df["Receiving Currency"]).astype(int)
    df["same_bank"] = (df["From Bank"] == df["To Bank"]).astype(int)
    df["is_self_loop"] = (df["sender_node"] == df["receiver_node"]).astype(int)

    df["is_micro_amount"] = (df["Amount Paid"] < 1.0).astype(int)
    df["is_round_100"] = (df["Amount Paid"] % 100 == 0).astype(int)
    df["is_round_1000"] = (df["Amount Paid"] % 1000 == 0).astype(int)
    df["near_threshold_10k"] = df["Amount Paid"].between(
        NEAR_THRESHOLD_LOW, NEAR_THRESHOLD_HIGH
    ).astype(int)

    df["hour"] = df["Timestamp"].dt.hour
    df["dayofweek"] = df["Timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_night"] = df["hour"].between(NIGHT_HOUR_START, NIGHT_HOUR_END).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def _prior_unique_counterparty_counts(group: pd.Series) -> pd.Series:
    return (~group.duplicated()).cumsum().shift(fill_value=0)


def add_account_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding account rolling features")
    df = df.copy()

    # sender-side history
    g_send = df.groupby("sender_node")
    df["sender_prior_txn_count"] = g_send.cumcount()
    df["sender_prior_sent_sum"] = g_send["Amount Paid"].cumsum() - df["Amount Paid"]
    df["sender_prior_sent_mean"] = (
        df["sender_prior_sent_sum"] / df["sender_prior_txn_count"].replace(0, np.nan)
    ).fillna(0.0)
    df["sender_prior_unique_counterparties"] = g_send["receiver_node"].transform(
        _prior_unique_counterparty_counts
    )
    df["sender_self_loop_rate"] = g_send["is_self_loop"].transform(
        lambda s: s.shift().expanding().mean()
    ).fillna(0.0)
    df["sender_micro_amount_rate"] = g_send["is_micro_amount"].transform(
        lambda s: s.shift().expanding().mean()
    ).fillna(0.0)

    sender_prev_ts = g_send["Timestamp"].shift(1)
    df["sender_time_since_prev_s"] = (
        (df["Timestamp"] - sender_prev_ts).dt.total_seconds()
    ).fillna(-1)

    # receiver-side history
    g_recv = df.groupby("receiver_node")
    df["receiver_prior_txn_count"] = g_recv.cumcount()
    df["receiver_prior_received_sum"] = g_recv["Amount Received"].cumsum() - df["Amount Received"]
    df["receiver_prior_unique_counterparties"] = g_recv["sender_node"].transform(
        _prior_unique_counterparty_counts
    )

    df["sender_pass_through_ratio"] = (
        df["sender_prior_sent_sum"] / df["receiver_prior_received_sum"].replace(0, np.nan)
    ).fillna(0.0)

    return df


def add_pairwise_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding pairwise features")
    df = df.copy()

    g_pair = df.groupby(["sender_node", "receiver_node"])
    df["pair_prior_count"] = g_pair.cumcount()
    df["pair_is_first_interaction"] = (df["pair_prior_count"] == 0).astype(int)

    seen_pairs: set[tuple[str, str]] = set()
    reciprocal_flags = np.zeros(len(df), dtype=int)
    for i, (sender, receiver) in enumerate(zip(df["sender_node"], df["receiver_node"])):
        if (receiver, sender) in seen_pairs:
            reciprocal_flags[i] = 1
        seen_pairs.add((sender, receiver))
    df["pair_is_reciprocated"] = reciprocal_flags

    return df


def _closes_short_cycle(
    source: str,
    target: str,
    successors: dict[str, set[str]],
    hop_limit: int = CYCLE_HOP_LIMIT,
) -> int:
    if target not in successors:
        return 0
    frontier = {target}
    visited = {target}
    for _ in range(hop_limit):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in successors.get(node, ()):
                if neighbor == source:
                    return 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return 0


def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding graph features")

    sender_degree, receiver_degree = [], []
    sender_pagerank, receiver_pagerank = [], []
    cycle_closure = []

    out_degree: dict[str, int] = {}
    in_degree: dict[str, int] = {}
    successors: dict[str, set[str]] = {}

    use_pagerank = len(df) <= PAGERANK_MAX_ROWS
    pagerank_cache: dict[str, float] = {}
    recompute_every = max(1, len(df) // PAGERANK_RECOMPUTE_FRACTION)

    for i, (u, v) in enumerate(zip(df["sender_node"], df["receiver_node"])):
        sender_degree.append(out_degree.get(u, 0) + in_degree.get(u, 0))
        receiver_degree.append(out_degree.get(v, 0) + in_degree.get(v, 0))

        if use_pagerank and (i % recompute_every == 0 or not pagerank_cache):
            graph = nx.DiGraph()
            for source, targets in successors.items():
                graph.add_edges_from((source, target) for target in targets)
            pagerank_cache = nx.pagerank(graph, alpha=0.85) if graph.number_of_edges() else {}
        sender_pagerank.append(pagerank_cache.get(u, 0.0))
        receiver_pagerank.append(pagerank_cache.get(v, 0.0))

        cycle_closure.append(_closes_short_cycle(u, v, successors))

        successors.setdefault(u, set()).add(v)
        out_degree[u] = out_degree.get(u, 0) + 1
        in_degree[v] = in_degree.get(v, 0) + 1

    df = df.copy()
    df["sender_degree_before"] = sender_degree
    df["receiver_degree_before"] = receiver_degree
    df["sender_pagerank_before"] = sender_pagerank
    df["receiver_pagerank_before"] = receiver_pagerank
    df["closes_short_cycle"] = cycle_closure
    return df


def add_entity_bank_features(df: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding entity/bank metadata features")
    df = df.copy()

    meta = accounts[["node", "Entity ID", "entity_type", "Bank Name"]].copy()
    meta["country"] = meta["Bank Name"].apply(_parse_country)

    accounts_per_entity = (
        accounts.groupby("Entity ID")["node"].nunique().rename("accounts_per_entity")
    )
    meta = meta.merge(accounts_per_entity, on="Entity ID", how="left")
    meta_by_node = meta.set_index("node")

    meta_columns = ["Entity ID", "entity_type", "Bank Name", "country", "accounts_per_entity"]
    for role, node_col in (("sender", "sender_node"), ("receiver", "receiver_node")):
        for column in meta_columns:
            df[f"{role}_{column}"] = df[node_col].map(meta_by_node[column])

    fill_defaults = {
        "sender_entity_type": "Unknown",
        "receiver_entity_type": "Unknown",
        "sender_country": "Unknown",
        "receiver_country": "Unknown",
        "sender_accounts_per_entity": 1,
        "receiver_accounts_per_entity": 1,
    }
    df = df.fillna(fill_defaults)

    df["is_cross_border"] = (df["sender_country"] != df["receiver_country"]).astype(int)
    same_entity_id = (
        df["sender_Entity ID"].notna()
        & df["receiver_Entity ID"].notna()
        & (df["sender_Entity ID"] == df["receiver_Entity ID"])
    )
    df["same_entity"] = (same_entity_id | (df["is_self_loop"] == 1)).astype(int)

    assert (df.loc[df["is_self_loop"] == 1, "same_entity"] == 1).all(), (
        "Invariant violated: self-loop rows must have same_entity == 1"
    )

    return df


def build_feature_table(trans_path: str | Path, accounts_path: str | Path) -> pd.DataFrame:
    df = load_transactions(trans_path)
    accounts = load_accounts(accounts_path)

    df = add_edge_features(df)
    df = add_account_rolling_features(df)
    df = add_pairwise_features(df)
    df = add_graph_features(df)
    df = add_entity_bank_features(df, accounts)

    logger.info("Feature table built with shape: %s", df.shape)
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--accounts", default=str(DATA_DIR / "HI-Small_accounts.csv"),)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=10_000,
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    trans_path = Path(args.transactions)
    accounts_path = Path(args.accounts)
    output_dir = Path(args.output_dir)
    if not trans_path.is_absolute():
        trans_path = ROOT_DIR / trans_path
    if not accounts_path.is_absolute():
        accounts_path = ROOT_DIR / accounts_path
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    features = build_feature_table(trans_path, accounts_path)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(features.shape)
    print(features.head(3))

    features.to_parquet(output_dir / "feature_table.parquet", index=False, compression="zstd")
    features.head(args.sample_rows).to_csv(output_dir / "feature_table_sample.csv", index=False)


if __name__ == "__main__":
    main()