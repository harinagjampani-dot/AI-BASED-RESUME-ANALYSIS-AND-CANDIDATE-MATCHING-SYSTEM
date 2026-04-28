import os
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import docx
import fitz


# Extract text
def extract_text(file):
    text = ""

    try:
        if file.name.endswith(".pdf"):
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()

        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"

    except Exception as e:
        st.error(f"Error reading file: {e}")

    return text.strip()


# Rating meaning
def get_rating(score):
    if score >= 85:
        return "Excellent Match"
    elif score >= 70:
        return "Good Match"
    elif score >= 55:
        return "Moderate Match"
    elif score >= 40:
        return "Weak Match"
    else:
        return "Poor Match"


# Model + FAISS
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384


# FRONTEND UI
st.title(" AI Resume Matcher")

job_desc = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF/DOCX)", 
    type=["pdf", "docx"], 
    accept_multiple_files=True
)

if st.button("Match Resumes"):

    if not job_desc or not uploaded_files:
        st.warning("Please provide job description and upload resumes.")
    else:
        index = faiss.IndexFlatL2(embedding_dim)
        resume_names = []
        resume_embeddings = []

        # Process resumes
        for file in uploaded_files:
            text = extract_text(file)

            if text:
                emb = model.encode([text])
                emb = np.array(emb).astype("float32")

                index.add(emb)
                resume_names.append(file.name)
                resume_embeddings.append(emb)

        # Encode job description
        job_emb = model.encode([job_desc])
        job_emb = np.array(job_emb).astype("float32")

        D, I = index.search(job_emb, len(resume_names))

        results = []

        for rank, idx in enumerate(I[0]):
            distance = D[0][rank]
            similarity = 1 / (1 + distance)
            percent = round(similarity * 100, 2)
            rating = get_rating(percent)

            results.append((resume_names[idx], percent, rating))

        results.sort(key=lambda x: x[1], reverse=True)

        st.subheader("📊 Resume Ranking")

        for i, res in enumerate(results):
            st.write(f"**{i+1}. {res[0]}**")
            st.write(f"Match: {res[1]}%")
            st.write(f"Rating: {res[2]}")
            st.write("---")