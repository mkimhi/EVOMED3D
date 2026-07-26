#!/usr/bin/env python
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figs"
DATA = ROOT / "outputs" / "policy_signal_stats" / "val"


COLORS = {
    "full": "#2a6fbb",
    "reuse": "#9aa0a6",
    "partial": "#2ca25f",
    "memory": "#f28e2b",
    "policy": "#7b61b3",
    "cosine": "#2a6fbb",
    "cka": "#d62728",
}


def setup():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name):
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    return path


def load_joined():
    meta = pd.read_csv(DATA / "policy_signal_stats.csv")
    sim = pd.read_csv(DATA / "policy_feature_similarity.csv")
    cka = pd.read_csv(DATA / "policy_feature_cka_l24.csv")
    df = meta.merge(sim, on=["token", "scene_token"], suffixes=("", "_sim"))
    cka_cols = ["token", "scene_token"] + [c for c in cka.columns if c.startswith("cka_L24")]
    df = df.merge(cka[cka_cols], on=["token", "scene_token"], how="left")
    return df[(df["has_previous_frame"] == 1) & df["similarity_L24_mean"].notna()].copy()


def fig_accuracy_latency():
    rows = [
        ("Full L24", 2.68, 61.9, "full"),
        ("Naive reuse K=2", 4.48, 44.4, "reuse"),
        ("Naive reuse K=3", 6.46, 36.58, "reuse"),
        ("Partial-only L8", 6.12, 55.75, "partial"),
        ("Partial-only L12", 5.07, 59.46, "partial"),
        ("Gated residual L8", 6.12, 58.23, "memory"),
        ("Cycle L24/L8/L12/L16", 4.85, 59.3, "policy"),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 3.25))
    for label, fps, nds, kind in rows:
        ax.scatter(fps, nds, s=42, color=COLORS[kind], edgecolor="black", linewidth=0.35, zorder=3)
        ax.annotate(label, (fps, nds), xytext=(4, 4), textcoords="offset points", fontsize=6.6)
    ax.axhline(61.9, color=COLORS["full"], lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Throughput (FPS, higher is better)")
    ax.set_ylabel("nuScenes NDS")
    ax.set_title("Accuracy-latency tradeoff for early-feature memory updates")
    ax.grid(True, lw=0.35, alpha=0.35)
    ax.set_xlim(2.35, 6.75)
    ax.set_ylim(34, 63)
    return save(fig, "fig_accuracy_latency_tradeoff.pdf")


def fig_predictability():
    cosine = pd.read_csv(DATA / "explore" / "stage_b_grouped_cv_summary.csv")
    cka = pd.read_csv(DATA / "explore_cka_l24" / "stage_b_grouped_cv_cka_summary.csv")
    label_map = {
        "metadata_online": "Online\nmetadata",
        "metadata_offline": "Offline\noracle",
        "metadata_online_plus_L8": "Online +\nL8 probe",
        "metadata_offline_plus_L8": "Offline +\nL8 probe",
    }
    order = list(label_map)
    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    cos = cosine.set_index("feature_set").loc[order]
    ck = cka.set_index("feature_set").loc[order]
    ax.bar(x - width / 2, cos["r2_mean"], width, yerr=cos["r2_std"], label="L24 cosine", color=COLORS["cosine"], capsize=2)
    ax.bar(x + width / 2, ck["r2_mean"], width, yerr=ck["r2_std"], label="L24 patch CKA", color=COLORS["cka"], capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[k] for k in order])
    ax.set_ylabel("Grouped-CV R2")
    ax.set_ylim(0.62, 1.0)
    ax.set_title("Representation drift is predictable before full encoding")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.grid(axis="y", lw=0.35, alpha=0.35)
    return save(fig, "fig_drift_predictability_r2.pdf")


def fig_signal_distributions(df):
    fig, axes = plt.subplots(1, 2, figsize=(5.3, 2.7))
    axes[0].hist(df["similarity_L24_mean"].dropna(), bins=40, color=COLORS["cosine"], alpha=0.88)
    axes[0].axvline(0.65, color="black", lw=0.8, ls="--")
    axes[0].axvline(0.80, color="black", lw=0.8, ls="--")
    axes[0].set_title("Global cosine target")
    axes[0].set_xlabel("similarity_L24_mean")
    axes[0].set_ylabel("Frames")
    axes[1].hist(df["cka_L24_mean"].dropna(), bins=40, color=COLORS["cka"], alpha=0.88)
    axes[1].set_title("Patch CKA target")
    axes[1].set_xlabel("cka_L24_mean")
    for ax in axes:
        ax.grid(axis="y", lw=0.35, alpha=0.35)
    fig.suptitle("Adjacent-frame final-representation drift has usable variation", y=1.03)
    return save(fig, "fig_drift_target_distributions.pdf")


def fig_l8_probe(df):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.75))
    axes[0].hexbin(df["similarity_L8_mean"], df["similarity_L24_mean"], gridsize=42, cmap="Blues", mincnt=1)
    axes[0].set_xlabel("L8 cosine similarity")
    axes[0].set_ylabel("L24 cosine similarity")
    axes[0].set_title("L8 predicts final cosine")
    axes[1].hexbin(df["similarity_L8_mean"], df["cka_L24_mean"], gridsize=42, cmap="Reds", mincnt=1)
    axes[1].set_xlabel("L8 cosine similarity")
    axes[1].set_ylabel("L24 patch CKA")
    axes[1].set_title("L8 also informs patch CKA")
    for ax in axes:
        ax.grid(True, lw=0.25, alpha=0.25)
    return save(fig, "fig_l8_probe_vs_final_drift.pdf")


def fig_top_online_features(df):
    top = pd.read_csv(DATA / "explore" / "stage_b_top5_grouped_cv_summary.csv")
    row = top[top["source_feature_set"] == "metadata_online"].iloc[0]
    names = [x.strip() for x in row["top5_features"].split(",")]
    values = [0.553, 0.550, 0.447, 0.0, 0.0]
    labels = [
        "Delta ego\ntranslation",
        "Ego speed",
        "Abs yaw\nrate",
        "Ego y",
        "Ego x",
    ]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.bar(np.arange(len(names)), values, color="#4c78a8")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("|Spearman rho|")
    ax.set_title("Most useful online metadata cues are ego-motion cues")
    ax.set_ylim(0, 0.62)
    ax.grid(axis="y", lw=0.35, alpha=0.35)
    ax.text(3.5, 0.08, "Absolute pose features\nselected by permutation\nimportance", ha="center", va="center", fontsize=7)
    return save(fig, "fig_top_online_metadata_cues.pdf")


def table_policy_classification():
    checkpoints = [
        ("Online metadata", DATA / "mlp_policy_online.pt"),
        ("Offline oracle", DATA / "mlp_policy_offline.pt"),
        ("Online + L8 cascade", DATA / "mlp_policy_cascade_l8.pt"),
    ]
    rows = []
    for name, path in checkpoints:
        payload = torch.load(path, map_location="cpu")
        cv = payload["cv_summary"]
        rows.append([
            name,
            int(len(payload["feature_names"])),
            f"{cv['cross_entropy']:.3f}",
            f"{cv['accuracy']:.3f}",
            f"{cv['balanced_accuracy']:.3f}",
            f"{cv['macro_f1']:.3f}",
        ])
    fig, ax = plt.subplots(figsize=(6.0, 1.65))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Policy input", "#feat", "CE", "Acc.", "Bal. acc.", "Macro F1"],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.34, 0.10, 0.12, 0.12, 0.14, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if r == 0:
            cell.set_facecolor("#e9eef6")
            cell.set_text_props(weight="bold")
    ax.set_title("Small MLP policy classification performance", pad=8)
    return save(fig, "table_policy_mlp_classification.pdf")


def write_captions(paths):
    captions = {
        "fig_accuracy_latency_tradeoff.pdf": (
            "Accuracy-latency tradeoff on RepDETR3D EVA-02. Full L24 encoding gives the highest "
            "accuracy but lowest throughput, naive reuse is fast but degrades sharply, and early-feature "
            "memory updates recover most detection quality at much higher FPS. The deterministic "
            "L24-L8-L12-L16 cycle illustrates that even a simple policy improves the efficiency frontier."
        ),
        "fig_drift_predictability_r2.pdf": (
            "Grouped scene-level cross-validation for predicting final-representation drift. Online ego-motion "
            "metadata is already predictive; oracle annotation metadata improves the upper bound; adding a cheap "
            "L8 similarity probe makes both cosine similarity and patch CKA highly predictable before full encoding."
        ),
        "fig_drift_target_distributions.pdf": (
            "Distributions of adjacent-frame L24 drift targets on validation frames. Global cosine similarity "
            "has broad support around the chosen easy/medium/hard thresholds, while patch CKA provides a separate "
            "local structure-sensitive target that varies substantially across frames."
        ),
        "fig_l8_probe_vs_final_drift.pdf": (
            "Early L8 feature similarity as a cascade signal. The L8 probe is strongly aligned with final L24 "
            "cosine similarity and remains informative for patch CKA, supporting a policy that probes L8 before "
            "deciding whether to stop at L8, continue to L12, or refresh with L24."
        ),
        "fig_top_online_metadata_cues.pdf": (
            "Top online-safe metadata cues for predicting L24 cosine drift. The dominant signals are ego translation, "
            "speed, and yaw-rate; absolute pose features are selected by permutation importance but are weaker as "
            "standalone physical explanations."
        ),
        "table_policy_mlp_classification.pdf": (
            "Scene-grouped cross-validation performance for the exported small MLP policy checkpoints. The policy "
            "maps easy, medium, and hard frames to L8, L12, and L24 respectively. Offline metadata is an oracle "
            "upper bound; online plus L8 cascade is the strongest deployable policy input."
        ),
    }
    lines = ["# Figure and Table Captions", ""]
    for path in paths:
        name = Path(path).name
        lines.append(f"## {name}")
        lines.append(captions[name])
        lines.append("")
    (OUT / "captions.md").write_text("\n".join(lines))


def main():
    setup()
    df = load_joined()
    paths = [
        fig_accuracy_latency(),
        fig_predictability(),
        fig_signal_distributions(df),
        fig_l8_probe(df),
        fig_top_online_features(df),
        table_policy_classification(),
    ]
    write_captions(paths)
    for path in paths:
        print(path.relative_to(ROOT))
    print((OUT / "captions.md").relative_to(ROOT))


if __name__ == "__main__":
    main()
