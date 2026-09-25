"""
app.py
------
Team Galaxy — WIUT Hackathon 2026, AML Alert Prioritization — EDA website.

This app only displays PRE-COMPUTED assets from `eda_assets/` (charts and
summary statistics) — it does not read the raw train/test CSV or parquet
files and has no dependency on them. This makes it safe to push to a
public GitHub repo and deploy on Streamlit Community Cloud.

Local run:
    streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).parent / "eda_assets"

st.set_page_config(page_title="Team Galaxy — AML Alert Prioritization EDA", layout="wide")

# ---------------------------------------------------------------- Header
st.title("🔍 AML Alert Prioritization — Exploratory Data Analysis")
st.caption("Team Galaxy · 9C953F22 · WIUT Hackathon 2026 · FinTech / AI in Finance track")

st.markdown(
    """
**Our approach in brief:** we transformed the relational dataset (one alert
linked to many historical transactions) into a signal-level feature table,
then trained LightGBM, CatBoost and XGBoost models with 5-fold
cross-validation and blended their predictions. Below is the EDA process
that shaped these modeling decisions.
"""
)

# ---------------------------------------------------------------- Overview
summary = {}
summary_path = ASSETS_DIR / "summary.json"
if summary_path.exists():
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

st.header("📊 Dataset Structure")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total signals (train)", f"{summary.get('n_signals', '—'):,}")
col2.metric("Total transactions (train)", f"{summary.get('n_transactions', '—'):,}")
col3.metric("Escalation rate", f"{summary.get('escalation_rate', 0):.1%}")
col4.metric("Avg. transactions/signal", f"{summary.get('avg_tx_per_signal', 0):.0f}")

st.markdown(
    f"""
**Data schema** *(original column names shown in parentheses)*:
- `signals`: `signal_id`, signal date (`signal_sanasi`), target (`eskalatsiya`: 0 = dismissed, 1 = escalated)
- `transactions` *(relational — many rows per signal)*: `signal_id`,
  transaction timestamp (`tranzaksiya_vaqti`), direction (`kirim_chiqim`: incoming/outgoing),
  transaction type (`tranzaksiya_turi`: card / bank transfer / cash / international),
  standardized amount (`miqdor_indeksi`)

Date range: **{summary.get('date_min', '—')} — {summary.get('date_max', '—')}**

*Note: for data-privacy and NDA reasons, this page shows only aggregated
statistics and charts — never raw transaction records.*
"""
)

st.divider()

# ---------------------------------------------------------------- Target
st.header("🎯 Target Distribution")
c1, c2 = st.columns([1, 1])
with c1:
    img = ASSETS_DIR / "target_distribution.png"
    if img.exists():
        st.image(str(img), use_container_width=True)
with c2:
    st.markdown(
        f"""
**Observation:** the target is notably **imbalanced** — the escalation
rate is only **{summary.get('escalation_rate', 0):.1%}**.

This directly shaped our modeling decisions:
- We used `scale_pos_weight` to compensate for the class imbalance
- We chose ROC-AUC as the evaluation metric (accuracy would be misleading
  on imbalanced data)
- We used StratifiedKFold so each fold preserves the target ratio
"""
    )

st.divider()

# ---------------------------------------------------------- Direction/Type
st.header("💳 Direction and Transaction Type")
img = ASSETS_DIR / "direction_type_distribution.png"
if img.exists():
    st.image(str(img), use_container_width=True)

st.markdown(
    """
**Observation:** the marginal (one-dimensional) distributions of direction
and transaction type are **almost identical between escalated and
dismissed alerts** (e.g. the cash-transaction share is around 6–6.5% for
both groups). This is an important, subtle finding: the signal is not
hidden in simple, single-variable statistics.

**What this told us:** our model needs to capture multi-feature
interactions and time-dependent patterns — which is why we added time
windows (1/3/7/14/30/90 days), inter-transaction time gaps (burst
detection), and cross-category combination features.
"""
)

st.divider()

# --------------------------------------------------------------- Amounts
st.header("💰 Transaction Amount Distribution")
c1, c2 = st.columns([1, 1])
with c1:
    img = ASSETS_DIR / "amount_distribution.png"
    if img.exists():
        st.image(str(img), use_container_width=True)
with c2:
    st.markdown(
        """
**Observation:** the distribution of `amount_index` (standardized
transaction size) is **very close** between escalated and dismissed
groups, with only a small difference. This shows that the simple
"large amounts are riskier" assumption doesn't fully hold — risk is
driven more by **behavioral patterns** (frequency, timing, combinations)
than by the amount itself.

This is why we added "round-number" (structuring indicator) and "recent
trend" (how recent transactions deviate from the overall average)
features.
"""
    )

st.divider()

# --------------------------------------------------------------- Windows
st.header("⏱️ Pre-signal Activity (Time Windows)")
c1, c2 = st.columns([1, 1])
with c1:
    img = ASSETS_DIR / "window_activity.png"
    if img.exists():
        st.image(str(img), use_container_width=True)
with c2:
    st.markdown(
        """
**Observation:** transaction activity computed over several time windows
before the signal date (1/3/7/14/30 days) is **consistently higher for
escalated alerts** across every window. For example, in the 30-day
window, dismissed alerts average ~78 transactions versus ~81 for
escalated alerts.

This is the **most consistent signal** we found: the gap is small, but
it points the same direction (escalated > dismissed) across every
window. This made time-windowed aggregates one of the most important
feature groups in our model.
"""
    )

st.divider()

# --------------------------------------------------------- Feature import.
img = ASSETS_DIR / "feature_importance.png"
if img.exists():
    st.header("🧠 Which Features the Model Relies on Most")
    st.image(str(img), use_container_width=True)
    st.markdown(
        """
The feature importance analysis of our LightGBM model shows that the
top-ranked features include not only basic statistics (`tx_min`,
`tx_max`) but also features motivated directly by our EDA — such as
`avg_dist_to_round` (round-number pattern) and `recent_5_trend` (recent
transaction trend). This confirms that our EDA-driven feature
engineering approach paid off.
"""
    )
    st.divider()

# --------------------------------------------------------------- Summary
st.header("📝 Conclusion — Key Findings")
st.markdown(
    f"""
1. **The target is notably imbalanced** ({summary.get('escalation_rate', 0):.1%}
   escalation rate) — this directly shaped our modeling and metric choices.
2. **Simple, one-dimensional statistics (marginal distributions) barely
   differ between escalated and dismissed groups** — showing the signal
   is complex, multi-feature, and time-dependent.
3. **Pre-signal activity (across all time windows) is our most consistent
   signal** — systematically higher for escalated alerts.
4. Based on these findings, we engineered **76 features** (base
   statistics, direction/type combinations, time windows, burst
   detection, round-number pattern, recent trend) and applied a
   **LightGBM + CatBoost + XGBoost ensemble**.

The full, reproducible pipeline code (`features.py`, `train.py`,
`predict.py`) and Jupyter notebook are available in our GitHub
repository.
"""
)

st.caption("Team Galaxy · WIUT Hackathon 2026 · Only aggregated statistics are shown, no raw data (NDA).")