"""
app.py
------
Team Galaxy — WIUT Hackathon 2026, AML Alert Prioritization — EDA website.

Bu ilova faqat `eda_assets/` papkasidagi OLDINDAN HISOBLANGAN grafik va
statistikalarni ko'rsatadi (xom train/test CSV yoki parquet fayllarni
o'qimaydi va ularga bog'liq emas) — shuning uchun uni GitHub'ga xavfsiz
push qilish va Streamlit Community Cloud'ga deploy qilish mumkin.

Local sinash:
    streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).parent / "eda_assets"

st.set_page_config(page_title="Team Galaxy — AML Alert Prioritization EDA", layout="wide")

# ---------------------------------------------------------------- Sarlavha
st.title("🔍 AML Alert Prioritization — Exploratory Data Analysis")
st.caption("Team Galaxy · 9C953F22 · WIUT Hackathon 2026 · FinTech / AI in Finance track")

st.markdown(
    """
**Yondashuvimiz qisqacha:** biz relational (bir alertga ko'plab tranzaksiya
to'g'ri keladigan) ma'lumotlar bazasini signal_id darajasidagi feature
jadvaliga aylantirdik, so'ngra LightGBM, CatBoost va XGBoost modellarini
5-fold cross-validation bilan o'qitib, ularning bashoratlarini blend
qildik. Quyida ushbu qarorlarga olib kelgan EDA jarayonimiz keltirilgan.
"""
)

# ---------------------------------------------------------------- Overview
summary = {}
summary_path = ASSETS_DIR / "summary.json"
if summary_path.exists():
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

st.header("📊 Dataset tuzilishi")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Jami signal (train)", f"{summary.get('n_signals', '—'):,}")
col2.metric("Jami tranzaksiya (train)", f"{summary.get('n_transactions', '—'):,}")
col3.metric("Escalation rate", f"{summary.get('escalation_rate', 0):.1%}")
col4.metric("O'rtacha tx/signal", f"{summary.get('avg_tx_per_signal', 0):.0f}")

st.markdown(
    f"""
**Ma'lumotlar sxemasi:**
- `signals`: `signal_id`, `signal_sanasi`, `eskalatsiya` (target: 0=dismissed, 1=escalated)
- `transactions` *(relational — bir signalga ko'plab qator)*: `signal_id`,
  `tranzaksiya_vaqti`, `kirim_chiqim` (kirim/chiqim), `tranzaksiya_turi`
  (karta/bank_otkazmasi/naqd/xalqaro), `miqdor_indeksi` (standartlashtirilgan)

Sana oralig'i: **{summary.get('date_min', '—')} — {summary.get('date_max', '—')}**

*Eslatma: barqarorlik va maxfiylik (NDA) sababli, bu sahifada xom
tranzaksiya ma'lumotlari emas, faqat jamlangan (agregatlangan)
statistikalar va grafiklar ko'rsatiladi.*
"""
)

st.divider()

# ---------------------------------------------------------------- Target
st.header("🎯 Target taqsimoti")
c1, c2 = st.columns([1, 1])
with c1:
    img = ASSETS_DIR / "target_distribution.png"
    if img.exists():
        st.image(str(img), use_container_width=True)
with c2:
    st.markdown(
        f"""
**Kuzatuv:** target sezilarli darajada **imbalanced** —
escalation rate atigi **{summary.get('escalation_rate', 0):.1%}**.

Bu bizning modellashtirish qarorlarimizga bevosita ta'sir qildi:
- `scale_pos_weight` orqali class imbalance'ni kompensatsiya qildik
- ROC-AUC metrikasini tanladik (imbalanced datada accuracy chalg'ituvchi
  bo'lishi mumkin)
- StratifiedKFold ishlatdik — har fold'da target nisbati saqlanishi uchun
"""
    )

st.divider()

# ---------------------------------------------------------- Direction/Type
st.header("💳 Kirim/Chiqim va Tranzaksiya turi")
img = ASSETS_DIR / "direction_type_distribution.png"
if img.exists():
    st.image(str(img), use_container_width=True)

st.markdown(
    """
**Kuzatuv:** kirim_chiqim va tranzaksiya_turi bo'yicha **marginal
(oddiy) taqsimotlar escalated va dismissed guruhlar orasida deyarli
farqlanmaydi** (masalan naqd tranzaksiya ulushi ikkala guruhda ham
~6-6.5% atrofida). Bu — muhim va nozik topilma: signal oddiy, bitta
o'lchovli statistikalarda YASHIRINMAGAN.

**Bu bizga nima dedi:** modelimiz ko'p-feature interaction va vaqtga
bog'liq pattern'larni ushlashi kerak — shuning uchun biz vaqt oynalari
(1/3/7/14/30/90 kun), tranzaksiyalar orasidagi vaqt farqi (burst
detection) va kategoriyalar aro kombinatsiya feature'larini qo'shdik.
"""
)

st.divider()

# --------------------------------------------------------------- Amounts
st.header("💰 Tranzaksiya miqdori taqsimoti")
c1, c2 = st.columns([1, 1])
with c1:
    img = ASSETS_DIR / "amount_distribution.png"
    if img.exists():
        st.image(str(img), use_container_width=True)
with c2:
    st.markdown(
        """
**Kuzatuv:** `miqdor_indeksi` (standartlashtirilgan tranzaksiya hajmi)
taqsimoti escalated va dismissed guruhlar orasida **juda yaqin**, ozgina
farq bilan. Bu shuni ko'rsatadiki, oddiy "yirik summalar xavfli" degan
faraz to'liq to'g'ri emas — xavf ko'proq **xatti-harakat pattern'ida**
(chastota, vaqt, kombinatsiya), summaning o'zida emas.

Shu sababdan biz "round-number" (davra summalar — structuring belgisi)
va "recent trend" (oxirgi tranzaksiyalarning umumiy o'rtachadan chetga
chiqishi) feature'larini qo'shdik.
"""
    )

st.divider()

# --------------------------------------------------------------- Windows
st.header("⏱️ Signal oldidan faollik (vaqt oynalari)")
c1, c2 = st.columns([1, 1])
with c1:
    img = ASSETS_DIR / "window_activity.png"
    if img.exists():
        st.image(str(img), use_container_width=True)
with c2:
    st.markdown(
        """
**Kuzatuv:** signal sanasidan oldingi turli vaqt oynalarida (1/3/7/14/30
kun) hisoblangan tranzaksiya faolligi, **escalated guruhda barcha
oynalarda biroz yuqoriroq**. Masalan 30 kunlik oynada dismissed guruh
o'rtacha ~78 ta, escalated guruh esa ~81 ta tranzaksiyaga ega.

Bu — biz topgan **eng izchil (consistent) signal**: farq kichik, lekin
barcha oynalarda bir tomonlama (escalated > dismissed). Shuning uchun
vaqt-oynali agregatlar bizning modelimizdagi eng muhim feature guruhlaridan
biriga aylandi.
"""
    )

st.divider()

# --------------------------------------------------------- Feature import.
img = ASSETS_DIR / "feature_importance.png"
if img.exists():
    st.header("🧠 Model qaysi feature'larga eng ko'p tayanadi")
    st.image(str(img), use_container_width=True)
    st.markdown(
        """
LightGBM modelimizning feature importance tahlili shuni ko'rsatdiki, eng
yuqori o'rinlarda nafaqat oddiy statistikalar (`tx_min`, `tx_max`), balki
EDA orqali maqsadli qo'shilgan feature'lar ham bor — masalan
`avg_dist_to_round` (round-number pattern) va `recent_5_trend` (so'nggi
tranzaksiyalar trendi). Bu bizning EDA-asoslangan feature engineering
yondashuvimizni tasdiqlaydi.
"""
    )
    st.divider()

# --------------------------------------------------------------- Xulosa
st.header("📝 Xulosa — eng muhim topilmalar")
st.markdown(
    f"""
1. **Target sezilarli imbalanced** ({summary.get('escalation_rate', 0):.1%}
   escalation rate) — modellashtirish va metrika tanlovimizga bevosita
   ta'sir qildi.
2. **Oddiy, bir o'lchovli statistikalar (marginal distributions) bo'yicha
   escalated va dismissed guruhlar deyarli farqlanmaydi** — bu signal
   murakkab, ko'p-feature va vaqtga bog'liq ekanligini ko'rsatdi.
3. **Signal oldidan faollik (barcha vaqt oynalarida) eng izchil signal** —
   escalated guruhda tizimli ravishda yuqoriroq.
4. Ushbu topilmalar asosida biz **76 ta feature** (asosiy statistikalar,
   kirim/chiqim va tur kombinatsiyalari, vaqt oynalari, burst detection,
   round-number pattern, recent trend) yaratdik va **LightGBM + CatBoost +
   XGBoost ensemble** modelini qo'lladik.

Loyihaning to'liq, qayta ishga tushirilishi mumkin bo'lgan kodi
(`features.py`, `train.py`, `predict.py`) va Jupyter notebook GitHub
repozitoriyamizda mavjud.
"""
)

st.caption("Team Galaxy · WIUT Hackathon 2026 · Faqat agregatlangan statistikalar ko'rsatilgan, xom ma'lumot emas (NDA).")
