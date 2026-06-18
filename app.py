import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: white;
}

.block-container {
    padding-top: 1rem;
}

.metric-box {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
}

.section-title {
    color: #2563EB;
    font-size: 28px;
    font-weight: bold;
    margin-top: 15px;
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
Analisis Sentimen dan Tingkat Engagement Komentar TikTok
terkait Kecelakaan KRL Bekasi Menggunakan IndoBERT
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

# =====================================================
# KPI
# =====================================================

accuracy = 78.78
precision = 79.05
recall = 78.78
f1 = 78.67

negatif = len(df[df["label"]=="negatif"])
netral = len(df[df["label"]=="netral"])
positif = len(df[df["label"]=="positif"])

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric("📄 Dataset", len(df))

with col2:
    st.metric("🎯 Accuracy", f"{accuracy}%")

with col3:
    st.metric("⚡ Precision", f"{precision}%")

with col4:
    st.metric("🏆 F1 Score", f"{f1}%")

st.write("")

col5,col6,col7 = st.columns(3)

with col5:
    st.metric("😡 Negatif", negatif)

with col6:
    st.metric("😐 Netral", netral)

with col7:
    st.metric("😊 Positif", positif)

# =====================================================
# DISTRIBUSI SENTIMEN
# =====================================================

st.markdown(
    "<div class='section-title'>📊 Distribusi Sentimen</div>",
    unsafe_allow_html=True
)

sentiment_counts = (
    df["label"]
    .value_counts()
    .reset_index()
)

sentiment_counts.columns = [
    "Sentimen",
    "Jumlah"
]

c1,c2 = st.columns(2)

with c1:

    fig = px.pie(
        sentiment_counts,
        names="Sentimen",
        values="Jumlah",
        hole=0.65,
        color="Sentimen",
        color_discrete_map={
            "negatif":"#EF4444",
            "netral":"#F59E0B",
            "positif":"#10B981"
        }
    )

    fig.update_layout(
        title="Donut Chart Sentimen",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with c2:

    fig2 = px.bar(
        sentiment_counts,
        x="Sentimen",
        y="Jumlah",
        color="Sentimen",
        color_discrete_map={
            "negatif":"#EF4444",
            "netral":"#F59E0B",
            "positif":"#10B981"
        }
    )

    fig2.update_layout(
        title="Jumlah Komentar per Sentimen"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =====================================================
# WORD CLOUD
# =====================================================

st.markdown(
    "<div class='section-title'>☁️ Top Keywords Discussion</div>",
    unsafe_allow_html=True
)

col1,col2,col3 = st.columns([1,4,1])

with col2:
    st.image(
        "assets/WordCloud.png",
        width=750
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

    categories = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    values = [
        accuracy,
        precision,
        recall,
        f1
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="IndoBERT"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,100]
            )
        ),
        title="Radar Chart Performa Model",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# ENGAGEMENT ANALYSIS
# =====================================================

st.markdown(
    "<div class='section-title'>🔥 Engagement Analysis</div>",
    unsafe_allow_html=True
)

engagement_data = (
    df.groupby("label")["engagement"]
    .sum()
    .reset_index()
)

fig = px.bar(
    engagement_data,
    x="engagement",
    y="label",
    orientation="h",
    color="label",
    color_discrete_map={
        "negatif":"#EF4444",
        "netral":"#F59E0B",
        "positif":"#10B981"
    },
    text="engagement"
)

fig.update_layout(
    title="Total Engagement per Sentimen",
    yaxis_title="Sentimen",
    xaxis_title="Engagement"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# TOP KOMENTAR VIRAL
# =====================================================

st.markdown(
    "<div class='section-title'>🔥 Top 10 Komentar Viral</div>",
    unsafe_allow_html=True
)

top = (
    df.sort_values(
        by="engagement",
        ascending=False
    )
    .head(10)
)

st.dataframe(
    top[
        [
            "clean_comment",
            "label",
            "engagement"
        ]
    ],
    use_container_width=True
)

# =====================================================
# DISTRIBUSI LIKES
# =====================================================

st.markdown(
    "<div class='section-title'>❤️ Distribusi Likes</div>",
    unsafe_allow_html=True
)

fig = px.histogram(
    df,
    x="likes",
    nbins=30,
    title="Distribusi Likes Komentar"
)

fig.update_layout(
    bargap=0.05
)

fig.update_layout(
    title="Distribusi Likes"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# HEATMAP
# =====================================================

st.markdown(
    "<div class='section-title'>📈 Correlation Analysis</div>",
    unsafe_allow_html=True
)

col1,col2,col3 = st.columns([1,3,1])

with col2:
    st.image(
        "assets/heatmap.png",
        width=550
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

<h4>Ringkasan Temuan</h4>

<ul>
<li>Sentimen negatif mendominasi dengan <b>{negatif}</b> komentar.</li>
<li>Total engagement tertinggi berasal dari sentimen negatif.</li>
<li>Model IndoBERT mencapai akurasi <b>{accuracy}%</b>.</li>
<li>Likes memiliki korelasi paling kuat terhadap engagement.</li>
<li>Dataset penelitian terdiri dari <b>{len(df)}</b> komentar TikTok.</li>
</ul>

</div>
""", unsafe_allow_html=True)

# =====================================================
# SAMPLE DATA
# =====================================================

st.markdown(
    "<div class='section-title'>📝 Contoh Dataset</div>",
    unsafe_allow_html=True
)

st.dataframe(
    df[
        [
            "clean_comment",
            "label",
            "likes",
            "replies",
            "engagement"
        ]
    ].head(20),
    use_container_width=True
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption("""
RailWatch AI © 2026

Projek Mandiri Informatika

Rifqi Falih Ramadhan

Analisis Sentimen dan Tingkat Engagement Komentar TikTok terkait Kecelakaan KRL Bekasi Menggunakan IndoBERT
""")