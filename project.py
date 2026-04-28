import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from sentence_transformers import SentenceTransformer
import docx
import PyMuPDF

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide"
)

# ------------------------------
# CUSTOM CSS (Premium UI)
# ------------------------------
st.markdown("""
    <style>
        .main {
            background: linear-gradient(180deg, #eef2ff 0%, #f8fafc 100%);
            color: #0f172a;
        }
        .title {
            text-align: center;
            font-size: 42px;
            font-weight: 800;
            color: #1e3a8a;
            margin-bottom: 8px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #475569;
            margin-bottom: 28px;
        }
        .stTextArea > div > div > textarea {
            min-height: 240px;
            border-radius: 22px;
            border: 1px solid #cbd5e1;
            box-shadow: inset 0 1px 3px rgba(15,23,42,0.08);
            padding: 16px;
        }
        .stFileUploader>div>div {
            border: 2px dashed #60a5fa;
            border-radius: 22px;
            background: rgba(59,130,246,0.08);
            padding: 1.25rem;
        }
        .stButton>button {
            background: linear-gradient(135deg, #2563eb, #8b5cf6) !important;
            color: #ffffff !important;
            border-radius: 999px;
            padding: 0.95rem 1.75rem;
            font-size: 1rem;
            font-weight: 600;
            box-shadow: 0 14px 30px rgba(37,99,235,0.18);
        }
        .result-box {
            padding: 22px;
            border-radius: 20px;
            background: #ffffff;
            border: 1px solid rgba(148,163,184,0.18);
            box-shadow: 0px 16px 40px rgba(15,23,42,0.08);
            margin-bottom: 18px;
        }
        .result-box h4 {
            margin-bottom: 10px;
            color: #1e293b;
        }
        .result-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-top: 10px;
        }
        .result-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .result-pill.rating {
            background: #f8fafc;
            color: #0f172a;
        }
        .stAlert, .stInfo {
            border-radius: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# TITLE
# ------------------------------
st.markdown('<div class="title">📄 AI Resume Matcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-Based Resume Analysis and Candidate Matching System</div>',
    unsafe_allow_html=True
)


# ------------------------------
# EXTRACT TEXT FROM FILES
# ------------------------------
def extract_text(file):
    text = ""

    try:
        if file.name.endswith(".pdf"):
            with fitz.open(stream=file.read(), filetype="pdf") as pdf:
                for page in pdf:
                    text += page.get_text()

        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"

    except Exception as e:
        st.error(f"Error reading file: {e}")

    return text.strip()


# ------------------------------
# MATCH RATING
# ------------------------------
def get_rating(score):
    if score >= 85:
        return "Excellent Match ✅"
    elif score >= 70:
        return "Good Match 👍"
    elif score >= 55:
        return "Moderate Match ⚠️"
    elif score >= 40:
        return "Weak Match ❗"
    return "Poor Match ❌"


# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()
embedding_dim = 384


# ------------------------------
# USER INPUTS
# ------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    job_desc = st.text_area(
        "Enter Job Description",
        height=220,
        placeholder="Example: Looking for a Python Developer with Machine Learning, SQL, and Streamlit skills..."
    )

with col2:
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF / DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    st.info("Upload multiple resumes for ranking.")


# ------------------------------
# MATCH BUTTON
# ------------------------------
if st.button("🚀 Match Resumes", use_container_width=True):

    if not job_desc or not uploaded_files:
        st.warning("Please enter job description and upload resumes.")

    else:
        with st.spinner("Analyzing resumes..."):
            resume_names = []
            resume_vectors = []

            # Process resumes
            for file in uploaded_files:
                text = extract_text(file)

                if text:
                    embedding = model.encode([text])
                    embedding = np.array(embedding).astype("float32")

                    resume_vectors.append(embedding[0])
                    resume_names.append(file.name)

            if len(resume_names) == 0:
                st.error("No readable resumes found.")
                st.stop()

            # Encode Job Description
            job_embedding = model.encode([job_desc])
            job_embedding = np.array(job_embedding).astype("float32")

            results = []

            for idx, resume_vector in enumerate(resume_vectors):
                similarity = cosine_similarity(
                    [job_embedding[0]],
                    [resume_vector]
                )[0][0]

                match_percent = round(float(similarity) * 100, 2)
                rating = get_rating(match_percent)

                results.append((resume_names[idx], match_percent, rating))

            results.sort(key=lambda x: x[1], reverse=True)

        # ------------------------------
        # DISPLAY RESULTS
        # ------------------------------
        st.subheader("📊 Resume Ranking Results")

        for i, result in enumerate(results, start=1):
            st.markdown(f"""
                <div class="result-box">
                    <h4>{i}. {result[0]}</h4>
                    <div class="result-meta">
                        <span class="result-pill">Match Score: {result[1]}%</span>
                        <span class="result-pill rating">Rating: {result[2]}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.success("Resume matching completed successfully!")
