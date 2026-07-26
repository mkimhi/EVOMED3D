#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold

ONLINE_METADATA = [
    "frame_number", "info_frame_idx", "is_scene_start", "dt_s",
    "ego_x_m", "ego_y_m", "ego_z_m", "ego_yaw_deg",
    "delta_ego_translation_m", "ego_speed_mps", "ego_accel_mps2",
    "yaw_rate_deg_s", "abs_yaw_rate_deg_s", "yaw_accel_deg_s2", "abs_yaw_accel_deg_s2",
]
LABELS = ["easy", "medium", "hard"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
ACTIONS = {"easy": "L8", "medium": "L12", "hard": "L24"}


class PolicyMlp(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_table(args):
    meta = pd.read_csv(args.metadata_csv)
    sim = pd.read_csv(args.similarity_csv)
    df = meta.merge(sim, on=["token", "scene_token"], suffixes=("", "_sim"))
    df = df[(df["has_previous_frame"] == 1) & df[args.target].notna()].copy()
    return meta, df


def feature_names(variant, meta, df):
    non_feature = {"index", "token", "scene_token", "timestamp", "has_previous_frame", "similarity_L24_mean"}
    offline = [c for c in meta.columns if c not in non_feature and pd.api.types.is_numeric_dtype(meta[c])]
    l8 = [c for c in df.columns if c.startswith("similarity_L8_")]
    if variant == "online":
        return ONLINE_METADATA
    if variant == "offline":
        return offline
    if variant == "cascade_l8":
        return ONLINE_METADATA + l8
    raise ValueError(f"unknown variant: {variant}")


def labels_from_target(values, easy_threshold, hard_threshold):
    labels = np.full(len(values), LABEL_TO_INDEX["medium"], dtype=np.int64)
    labels[values >= easy_threshold] = LABEL_TO_INDEX["easy"]
    labels[values < hard_threshold] = LABEL_TO_INDEX["hard"]
    return labels


def fit_model(x, y, hidden_dim, epochs, lr, weight_decay, seed):
    torch.manual_seed(seed)
    model = PolicyMlp(x.shape[1], hidden_dim=hidden_dim, out_dim=len(LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(x_t), y_t)
        loss.backward()
        opt.step()
    return model


def evaluate_cv(df, features, y, args):
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=args.cv_folds).split(df[features], y, df["scene_token"]), 1):
        x_train = df.iloc[tr][features].replace([np.inf, -np.inf], np.nan)
        x_test = df.iloc[te][features].replace([np.inf, -np.inf], np.nan)
        mean = x_train.mean(numeric_only=True).fillna(0.0)
        x_train = x_train.fillna(mean)
        x_test = x_test.fillna(mean)
        scale = x_train.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        xtr = ((x_train - mean) / scale).to_numpy(np.float32)
        xte = ((x_test - mean) / scale).to_numpy(np.float32)
        model = fit_model(xtr, y[tr], args.hidden_dim, args.epochs, args.lr, args.weight_decay, args.seed + fold)
        with torch.no_grad():
            logits = model(torch.as_tensor(xte, dtype=torch.float32))
            prob = torch.softmax(logits, dim=1).numpy()
        pred = prob.argmax(axis=1)
        rows.append(dict(
            fold=fold,
            cross_entropy=log_loss(y[te], prob, labels=[0, 1, 2]),
            accuracy=accuracy_score(y[te], pred),
            balanced_accuracy=balanced_accuracy_score(y[te], pred),
            macro_f1=f1_score(y[te], pred, average="macro"),
        ))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Train/export a tiny torch MLP temporal-memory policy from policy CSV tables.")
    parser.add_argument("--variant", required=True, choices=["online", "offline", "cascade_l8"])
    parser.add_argument("--metadata-csv", default="outputs/policy_signal_stats/val/policy_signal_stats.csv")
    parser.add_argument("--similarity-csv", default="outputs/policy_signal_stats/val/policy_feature_similarity.csv")
    parser.add_argument("--target", default="similarity_L24_mean")
    parser.add_argument("--easy-threshold", type=float, default=0.80)
    parser.add_argument("--hard-threshold", type=float, default=0.65)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    meta, df = load_table(args)
    features = feature_names(args.variant, meta, df)
    y = labels_from_target(df[args.target].to_numpy(), args.easy_threshold, args.hard_threshold)
    cv = evaluate_cv(df, features, y, args)
    print(cv.to_string(index=False))
    print(cv.mean(numeric_only=True).to_string())

    x_df = df[features].replace([np.inf, -np.inf], np.nan)
    mean = x_df.mean(numeric_only=True).fillna(0.0)
    x_df = x_df.fillna(mean)
    scale = x_df.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    x = ((x_df - mean) / scale).to_numpy(np.float32)
    model = fit_model(x, y, args.hidden_dim, args.epochs, args.lr, args.weight_decay, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        feature_names=features,
        mean=mean.reindex(features).fillna(0.0).astype(float).tolist(),
        scale=scale.reindex(features).fillna(1.0).astype(float).tolist(),
        hidden_dim=args.hidden_dim,
        labels=LABELS,
        actions=ACTIONS,
        requires_l8_similarity=args.variant == "cascade_l8",
        state_dict=model.net.state_dict(),
        variant=args.variant,
        target=args.target,
        easy_threshold=args.easy_threshold,
        hard_threshold=args.hard_threshold,
        cv_summary=cv.mean(numeric_only=True).to_dict(),
    ), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
