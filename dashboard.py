"""Optional browser UI, sharing the REST API model and probability cache."""

from engine import PredictRequest, get_labeler

import streamlit as st
from pydantic import ValidationError


st.set_page_config(page_title="AI Labeler", page_icon="🏷️", layout="centered")
st.title("AI Labeler")
st.caption("Labeli laporan: SAMPAH, LUBANG, KARATAN, dan RETAKAN.")

with st.form("prediction"):
    text = st.text_area(
        "Teks laporan", max_chars=20000, height=150,
        placeholder="Contoh: jalan berlubang dan banyak sampah",
    )
    threshold = st.slider("Threshold", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
    submitted = st.form_submit_button("Labeli", type="primary")

if submitted:
    try:
        payload = PredictRequest(text=text, threshold=threshold)
    except ValidationError:
        st.error("Isi teks laporan sebelum menjalankan prediksi.")
    else:
        with st.spinner("Memproses laporan..."):
            result = get_labeler().predict(payload.text, payload.threshold)
        st.subheader(
            ", ".join(result.labels) if result.labels else "Tidak ada label yang melewati threshold"
        )
        st.json(result.model_dump())

st.markdown("Integrasi aplikasi: `POST /predict`. [Dokumentasi API](/docs)")
