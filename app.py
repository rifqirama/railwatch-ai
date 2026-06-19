import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="RailWatch AI",
    page_icon="🚆",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("dataset_final_indobert.csv")

# =====================================================
# METRICS PENELITIAN
# =====================================================

ACCURACY = 83.45
PRECISION = 83.26
RECALL = 83.45
F1_SCORE = 83.28

DATASET_AWAL = 3203
DATASET_BALANCED = 3444

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: white;
}

.block-container {
    padding-top: 1rem;
    max-width: 1200px;
}

.section-title{
    font-size:28px;
    font-weight:700;
    color:#2563EB;
    margin-top:20px;
    margin-bottom:10px;
}

.metric-card{
    background:white;
    border:1px solid #E5E7EB;
    border-radius:12px;
    padding:15px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div style="
background:#2563EB;
padding:30px;
border-radius:15px;
color:white;
">

<h1>🚆 RailWatch AI</h1>

<p style="font-size:18px;">
Analisis Sentimen Komentar TikTok terkait
Insiden Kecelakaan KRL Bekasi Menggunakan IndoBERT
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

st.markdown(
    "<div class='section-title'>📊 Executive Summary</div>",
    unsafe_allow_html=True
)

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric(
        "📄 Dataset Awal",
        DATASET_AWAL
    )

with col2:
    st.metric(
        "⚖️ Dataset Balanced",
        DATASET_BALANCED
    )

with col3:
    st.metric(
        "🤖 Model",
        "IndoBERT"
    )

with col4:
    st.metric(
        "🔄 Oversampling",
        "Random"
    )

st.write("")

# =====================================================
# KPI MODEL
# =====================================================

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric(
        "🎯 Accuracy",
        f"{ACCURACY}%"
    )

with col2:
    st.metric(
        "⚡ Precision",
        f"{PRECISION}%"
    )

with col3:
    st.metric(
        "📈 Recall",
        f"{RECALL}%"
    )

with col4:
    st.metric(
        "🏆 F1 Score",
        f"{F1_SCORE}%"
    )

# =====================================================
# DISTRIBUSI SENTIMEN
# =====================================================

# =====================================================
# DISTRIBUSI DATASET
# =====================================================

st.markdown(
    "<div class='section-title'>📊 Distribusi Dataset</div>",
    unsafe_allow_html=True
)

before_df = pd.DataFrame({
    "Sentimen":[
        "Negatif",
        "Netral",
        "Positif"
    ],
    "Jumlah":[
        1340,
        1052,
        811
    ]
})

after_df = pd.DataFrame({
    "Sentimen":[
        "Negatif",
        "Netral",
        "Positif"
    ],
    "Jumlah":[
        1340,
        1052,
        1052
    ]
})

left,right = st.columns(2)

with left:

    st.subheader("Before Oversampling")

    fig_before = px.bar(
        before_df,
        x="Sentimen",
        y="Jumlah",
        color="Sentimen",
        color_discrete_map={
            "Negatif":"#EF4444",
            "Netral":"#F59E0B",
            "Positif":"#10B981"
        }
    )

    st.plotly_chart(
        fig_before,
        use_container_width=True
    )

with right:

    st.subheader("After Oversampling")

    fig_after = px.bar(
        after_df,
        x="Sentimen",
        y="Jumlah",
        color="Sentimen",
        color_discrete_map={
            "Negatif":"#EF4444",
            "Netral":"#F59E0B",
            "Positif":"#10B981"
        }
    )

    st.plotly_chart(
        fig_after,
        use_container_width=True
    )

# =====================================================
# WORD CLOUD
# =====================================================

st.markdown(
    "<div class='section-title'>☁️ Word Cloud</div>",
    unsafe_allow_html=True
)

col1,col2,col3 = st.columns([1,4,1])

with col2:

    st.image(
        "assets/WordCloud.png",
        width=700
    )

# =====================================================
# EVALUASI MODEL
# =====================================================

st.markdown(
    "<div class='section-title'>🎯 Evaluasi Model IndoBERT</div>",
    unsafe_allow_html=True
)

left,right = st.columns(2)

with left:

    st.image(
        "assets/confusionmatrix.png",
        width=500
    )

with right:

    st.image(
        "assets/performa_model.png",
        width=550
    )

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

st.markdown(
    "<div class='section-title'>📑 Classification Report</div>",
    unsafe_allow_html=True
)

report_df = pd.DataFrame({
    "Sentimen":[
        "Negatif",
        "Netral",
        "Positif"
    ],
    "Precision":[
        0.82,
        0.80,
        0.88
    ],
    "Recall":[
        0.83,
        0.73,
        0.94
    ],
    "F1-Score":[
        0.83,
        0.76,
        0.91
    ]
})

st.dataframe(
    report_df,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# INSIGHT
# =====================================================

st.markdown(
    "<div class='section-title'>📌 Insight Penelitian</div>",
    unsafe_allow_html=True
)

st.markdown(f"""
<div style="
background:#EFF6FF;
padding:25px;
border-radius:15px;
border-left:6px solid #2563EB;
">

<h4>Ringkasan Temuan Penelitian</h4>

<ul>
<li>Dataset awal terdiri dari <b>{DATASET_AWAL}</b> komentar TikTok.</li>
<li>Random Oversampling meningkatkan jumlah data menjadi <b>{DATASET_BALANCED}</b> komentar.</li>
<li>Model IndoBERT memperoleh accuracy sebesar <b>{ACCURACY}%</b>.</li>
<li>Kelas positif memiliki recall tertinggi sebesar <b>94%</b>.</li>
<li>Oversampling meningkatkan accuracy dari <b>78.78%</b> menjadi <b>83.45%</b>.</li>
</ul>

</div>
""", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption("""
RailWatch AI © 2026

Projek Mandiri Program Studi Informatika

Rifqi Falih Ramadhan

Analisis Sentimen Komentar TikTok terkait
Insiden Kecelakaan KRL Bekasi Menggunakan IndoBERT
""")