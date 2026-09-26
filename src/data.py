from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

ALL_COLUMNS = [*FEATURE_COLUMNS, "attack_type", "difficulty"]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
ENGINEERED_COLUMNS = [
    "bytes_ratio",
    "error_ratio",
    "log_src_bytes",
    "log_dst_bytes",
    "log_duration",
]


def load_nsl_kdd(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load one NSL-KDD file and return the 41 features and binary target."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"NSL-KDD file not found: {source}")

    frame = pd.read_csv(source, header=None, names=ALL_COLUMNS)
    if frame.shape[1] != len(ALL_COLUMNS):
        raise ValueError(
            f"Expected {len(ALL_COLUMNS)} columns, received {frame.shape[1]}"
        )

    target = frame["attack_type"].ne("normal").astype("int8")
    features = frame[FEATURE_COLUMNS].copy()
    return features, target

