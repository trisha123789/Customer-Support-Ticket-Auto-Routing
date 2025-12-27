import streamlit as st
import fasttext

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Ticket Auto-Router",
    page_icon="🤖",
    layout="centered"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    color: transparent;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #94a3b8;
}
.card {
    background-color: #020617;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 0px 25px rgba(99,102,241,0.2);
}
.result {
    font-size: 22px;
    font-weight: 700;
}
.confidence {
    font-size: 18px;
    color: #38bdf8;
}
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown('<div class="title">🤖 AI Ticket Auto-Routing System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by FastText • NLP • AI</div>', unsafe_allow_html=True)

st.write("")
st.write("")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return fasttext.load_model("ticket_router.ftz")

model = load_model()

# -------------------- UI CARD --------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    ticket = st.text_area(
        "✍️ Enter Customer Support Ticket",
        placeholder="Example: OTP not received after login...",
        height=120
    )

    col1, col2 = st.columns(2)

    with col1:
        predict_btn = st.button("🚀 Route Ticket")

    with col2:
        clear_btn = st.button("🧹 Clear")

    if clear_btn:
        st.experimental_rerun()

    if predict_btn and ticket.strip() != "":
        labels, probs = model.predict(ticket)

        department = labels[0].replace("__label__", "").replace("_", " ").title()
        confidence = round(probs[0] * 100, 2)

        st.write("")
        st.markdown(f"### 📌 Routed Department")
        st.markdown(f"<div class='result'>✅ {department}</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='confidence'>🔍 Confidence: {confidence}%</div>", unsafe_allow_html=True)

        st.progress(int(confidence))

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown(
    "<div class='footer'>Built with ❤️ by Trisha | Aspiring AI Engineer</div>",
    unsafe_allow_html=True
)
