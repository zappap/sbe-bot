import streamlit as st
import requests
import os
import tempfile
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------
# 1) PDF'leri GitHub'dan İndir
# -------------------------------

GITHUB_REPO_URL = "https://raw.githubusercontent.com/zappap/sbe-bot/main/docs"

PDF_FILES = [
    "Akademik_Danismanlik_Yonergesi.pdf",
    "DENKLIK_YONETMELIGI.pdf",
    "DEU_SBE_Uygulama_Esaslari.pdf",
    "Yatay_Gecis_Yonergesi.pdf",
    "deu_Lisansustu_Egitim_ve_Ogretim_Yonetmeligi.pdf",
    "inthal_raporu_uygulama_esaslari.pdf",
    "yok_LISANSUSTU_EGITIM_VE_OGRETIM_YONETMELIGI.pdf"
]

def download_pdf(file_name):
    url = f"{GITHUB_REPO_URL}/{file_name}"
    response = requests.get(url)
    if response.status_code == 200:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(response.content)
        tmp.close()
        return tmp.name
    return None


# -------------------------------
# 2) PDF'i Metne Çevir
# -------------------------------

def pdf_to_text(path):
    text = ""
    with open(path, "rb") as f:
        pdf = PdfReader(f)
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


# -------------------------------
# 3) Embedding + FAISS Index
# -------------------------------

@st.cache_resource
def load_documents():
    texts = []

    for pdf_file in PDF_FILES:
        local_path = download_pdf(pdf_file)
        if local_path:
            content = pdf_to_text(local_path)
            texts.append((pdf_file, content))

    return texts


@st.cache_resource
def build_faiss_index(texts):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    corpus = [t[1] for t in texts]

    embeddings = model.encode(corpus, show_progress_bar=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, model, corpus


# -------------------------------
# 4) Belgeye Dayalı Q&A
# -------------------------------

def answer_question(query, index, model, corpus, texts):
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k=1)

    best_doc = corpus[I[0][0]]
    file_name = texts[I[0][0]][0]

    # Çok düşük benzerlik → cevap yok
    if D[0][0] > 1.2:  
        return None, None

    return best_doc[:2000], file_name  # sadece ilk 2000 karakteri göster


# -------------------------------
# 5) Streamlit Arayüzü
# -------------------------------

st.title("📘 Sosyal Bilimler Enstitüsü Mevzuat Chatbotu")
st.write("Bu chatbot yalnızca yüklenmiş PDF mevzuatına dayanarak cevap verir.")

texts = load_documents()
index, model, corpus = build_faiss_index(texts)

question = st.text_input("Sorunuzu yazın:")

if question:
    answer, source = answer_question(question, index, model, corpus, texts)

    if answer:
        st.subheader("📌 Yanıt (Belgeye Dayalı)")
        st.write(answer)

        st.info(f"📄 Kaynak dosya: **{source}**")
    else:
        st.error("❗ Bu bilgi dosyalarda bulunmuyor.")
