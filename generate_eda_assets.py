"""
generate_eda_assets.py
-----------------------
Xom train_signals.csv / train_transactions.parquet fayllaridan EDA
grafiklari va agregatlangan statistikalarni hisoblab, `eda_assets/`
papkasiga saqlaydi.

MUHIM (NDA): bu skript FAQAT sizning local kompyuteringizda ishga
tushirilishi kerak. Chiqadigan `eda_assets/` papkasi (PNG rasmlar +
summary.json) — xom mijoz/tranzaksiya ma'lumotini o'z ichiga olmaydi,
faqat agregatlangan (jamlangan) sonlar va grafiklar. Shuning uchun bu
papkani GitHub'ga (EDA website uchun) xavfsiz push qilish mumkin.

Ishga tushirish:
    python generate_eda_assets.py --data-dir data --out-dir eda_assets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("eda_assets"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    signals = pd.read_csv(args.data_dir / "train_signals.csv", parse_dates=["signal_sanasi"])
    tx = pd.read_parquet(args.data_dir / "train_transactions.parquet")
    tx["tranzaksiya_vaqti"] = pd.to_datetime(tx["tranzaksiya_vaqti"])
    merged = tx.merge(signals[["signal_id", "eskalatsiya"]], on="signal_id")

    summary = {}

    # --- 1) Dataset overview ---
    summary["n_signals"] = int(len(signals))
    summary["n_transactions"] = int(len(tx))
    summary["escalation_rate"] = float(signals["eskalatsiya"].mean())
    summary["date_min"] = str(signals["signal_sanasi"].min().date())
    summary["date_max"] = str(signals["signal_sanasi"].max().date())
    summary["avg_tx_per_signal"] = float(tx.groupby("signal_id").size().mean())

    # --- 2) Target distribution chart ---
    fig, ax = plt.subplots(figsize=(5, 4))
    signals["eskalatsiya"].map({0: "Dismissed", 1: "Escalated"}).value_counts().plot(
        kind="bar", ax=ax, color=["#4C72B0", "#DD8452"]
    )
    ax.set_title("Target Distribution")
    ax.set_ylabel("Number of signals")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(args.out_dir / "target_distribution.png", dpi=120)
    plt.close(fig)

    # --- 3) Direction and transaction type (by target) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(
        data=merged.assign(target=merged["eskalatsiya"].map({0: "Dismissed", 1: "Escalated"})),
        x="kirim_chiqim", hue="target", ax=axes[0],
    )
    axes[0].set_title("Direction (by target)")
    axes[0].set_xlabel("Direction")
    axes[0].set_xticklabels(["Outgoing", "Incoming"])

    sns.countplot(
        data=merged.assign(target=merged["eskalatsiya"].map({0: "Dismissed", 1: "Escalated"})),
        x="tranzaksiya_turi", hue="target", ax=axes[1],
    )
    axes[1].set_title("Transaction Type (by target)")
    axes[1].set_xlabel("Transaction type")
    axes[1].set_xticklabels(["Bank transfer", "Card", "Cash", "International"])
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(args.out_dir / "direction_type_distribution.png", dpi=120)
    plt.close(fig)

    # --- 4) Transaction amount distribution ---
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=merged.assign(target=merged["eskalatsiya"].map({0: "Dismissed", 1: "Escalated"})),
        x="target", y="miqdor_indeksi", ax=ax,
    )
    ax.set_title("Transaction Amount: Escalated vs Dismissed")
    ax.set_xlabel("")
    ax.set_ylabel("Standardized amount")
    plt.tight_layout()
    fig.savefig(args.out_dir / "amount_distribution.png", dpi=120)
    plt.close(fig)

    # --- 5) Vaqt oynalari bo'yicha faollik (signal oldidan N kun) ---
    m2 = tx.merge(signals[["signal_id", "signal_sanasi", "eskalatsiya"]], on="signal_id")
    m2["days_before"] = (m2["signal_sanasi"] - m2["tranzaksiya_vaqti"]).dt.total_seconds() / 86400.0
    m2 = m2[m2["days_before"] >= 0]

    window_stats = []
    for w in [1, 3, 7, 14, 30]:
        sub = m2[m2["days_before"] <= w]
        cnt = sub.groupby(["signal_id", "eskalatsiya"]).size().reset_index(name="cnt")
        means = cnt.groupby("eskalatsiya")["cnt"].mean()
        window_stats.append({"window": w, "dismissed": float(means.get(0, 0)), "escalated": float(means.get(1, 0))})
    summary["window_activity"] = window_stats

    fig, ax = plt.subplots(figsize=(7, 4))
    df_win = pd.DataFrame(window_stats)
    ax.plot(df_win["window"], df_win["dismissed"], marker="o", label="Dismissed")
    ax.plot(df_win["window"], df_win["escalated"], marker="o", label="Escalated")
    ax.set_xlabel("Days before signal")
    ax.set_ylabel("Average transaction count")
    ax.set_title("Pre-signal Activity: Escalated vs Dismissed")
    ax.legend()
    plt.tight_layout()
    fig.savefig(args.out_dir / "window_activity.png", dpi=120)
    plt.close(fig)

    # --- Feature importance (agar models/ papkasida mavjud bo'lsa, ixtiyoriy) ---
    try:
        import joblib
        import numpy as np
        import glob

        feature_cols = joblib.load("models/feature_cols.pkl")
        model_paths = sorted(glob.glob("models/lgbm_fold*.pkl"))
        if model_paths:
            models = [joblib.load(p) for p in model_paths]
            importances = np.mean([m.feature_importances_ for m in models], axis=0)
            order = np.argsort(importances)[::-1][:15]

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh(
                [feature_cols[i] for i in order][::-1],
                [importances[i] for i in order][::-1],
            )
            ax.set_title("Top-15 Most Important Features (LightGBM)")
            plt.tight_layout()
            fig.savefig(args.out_dir / "feature_importance.png", dpi=120)
            plt.close(fig)
    except Exception as e:
        print(f"Feature importance chart skipped: {e}")

    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Tayyor! Barcha fayllar '{args.out_dir}/' papkasida.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()