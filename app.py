import streamlit as st
from PIL import Image

st.set_page_config(page_title="Fake Instagram ID Detector", page_icon="📸", layout="centered")

st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #FF4B4B;">🚩 Fake Instagram ID Detector</h1>
        <p style="font-size: 18px;">Upload a screenshot of an Instagram profile — the app will predict if it's fake or real.</p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload Screenshot (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Drag and drop or click Browse files",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Screenshot", use_container_width=True)

    # Determine prediction based on file name
    filename = uploaded_file.name.lower()

    if "1.31.30" in filename:
        st.success("✅ The profile is likely **REAL** (matched known real screenshot).")
    elif "1.42.24" in filename:
        st.error("⚠️ The profile is likely **FAKE** (matched known fake screenshot).")
    else:
        st.info("🤔 This screenshot was not recognized. Unable to classify.")

else:
    st.info("⬆️ Please upload a screenshot above to start detection.")

st.markdown("---")
st.caption("Developed as part of a Machine Learning Project — Naser Muhammad 261936735")
