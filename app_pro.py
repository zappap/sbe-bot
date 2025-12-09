# app_pro.py
# SBE Chatbot Pro - Streamlit uygulaması
# Özellikler: GitHub'daki docs klasöründen dosya indirip indeksleme,
# paragraf-chunk retrieval, kaynak gösterme, audit log, opsiyonel LLM özetleme.
#
# Kullanım:
# - REPO değişkenlerini ayarlayın ya da lokal çalıştırıyorsanız docs/ klasörüne yerleştirin.
# - (Opsiyonel) OPENAI_API_KEY veya GEMINI_API_KEY environment/secrets olarak ekleyin
# - streamlit run app_pro.py

import streamlit as st
import os
import tempfile
import hashlib
import sqlite3
from typing import List, Tuple, Dict
import glob
import pathlib
import json
import time

# PDF/DOCX parsing
import pdfplumber
import docx

# embeddings & faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# LLM (opsiyonel) - OpenAI (varsa)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="SBE Chatbot Pro", layout="wide")

# If you deploy to Streamlit Cloud, set this to the raw GitHub docs URL or leave empty to use local ./docs
GITHUB_RAW_DOCS_BASE = ""  # e.g. "https://raw.githubusercontent.com/USERNAME/REPO/main/docs"

# If local, the app will read ./docs/*.pdf, .docx, .txt
LOCAL_DOCS_PATH = "./docs"

# Embedding model - multilingual / Turkish performance iyi
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # iyi Türkçe desteği
CHUNK_SIZE = 900             # karakter
CHUNK_OVERLAP = 200
EMBED_DIM = 384  # uygun model için otomatik ayarlanacak
DEFAULT_SIM_THRESHOLD = 0.63

# Audit DB
DB_PATH = "sbe_chatbot_audit.db"

# -----------------------------
# UTIL: text extraction (pdf/docx/txt)
# -----------------------------
def extract_text_from_pdf(path: str) -> str:
    texts = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    texts.append(t)
    except Exception as e:
        st.warning(f"PDF okuma hatası ({path}): {e}")
    return "\n".join(texts)

def extract_text_from_docx(path: str) -> str:
    try:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def extract_text_generic(path: str) -> str:
    p = path.lower()
    if p.endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif p.endswith(".docx") or p.endswith(".doc"):
        return extract_text_from_docx(path)
    elif p.endswith(".txt"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    else:
        return ""

# -----------------------------
# UTIL: chunking (paragraf tabanlı + sliding)
# -----------------------------
def chunk_text_paragraphwise(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    # önce paragraf olarak ayır
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = cur + "\n" + p if cur else p
        else:
            chunks.append(cur)
            # overlap için son kısmı taşı (son overlap karakter)
            carry = cur[-overlap:] if overlap < len(cur) else cur
            cur = carry + "\n" + p
    if cur:
        chunks.append(cur)
    # temizlik
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks

def mkid(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# -----------------------------
# DocStore: passages + embeddings + faiss
# -----------------------------
class DocStore:
    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # dynamic embed dim
        self.embed_dim = self.model.get_sentence_embedding_dimension()
        self.passages = []  # dicts: {id, text, source, chunk_index}
        self.embeddings = None
        self.index = None

    def clear(self):
        self.passages = []
        self.embeddings = None
        self.index = None

    def add_document(self, file_path: str, file_name: str):
        txt = extract_text_generic(file_path)
        if not txt.strip():
            return 0
        chunks = chunk_text_paragraphwise(txt)
        added = 0
        for i, ch in enumerate(chunks):
            pid = mkid(file_name + f"__{i}")
            self.passages.append({"id": pid, "text": ch, "source": file_name, "chunk_index": i})
            added += 1
        return added

    def build_index(self):
        if not self.passages:
            self.embeddings = None
            self.index = None
            return
        texts = [p["text"] for p in self.passages]
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        emb = emb.astype("float32")
        # normalize for cosine via inner product
        faiss.normalize_L2(emb)
        self.embeddings = emb
        dim = emb.shape[1]
        # IndexFlatIP for cosine similarity after normalization
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        self.index = index

    def query(self, q: str, top_k=5):
        if self.index is None or len(self.passages)==0:
            return []
        q_emb = self.model.encode([q], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        scores = D[0].tolist()
        idxs = I[0].tolist()
        results = []
        for sc, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(self.passages):
                continue
            results.append((float(sc), self.passages[idx]))
        return results

# -----------------------------
# Audit DB helpers (sqlite)
# -----------------------------
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            query TEXT,
            result TEXT, -- 'FOUND' or 'NOT_FOUND'
            best_score REAL,
            top_passages TEXT
        )
    """)
    conn.commit()
    return conn

def log_query(conn, query, result, best_score, top_passages):
    c = conn.cursor()
    c.execute("INSERT INTO queries (ts, query, result, best_score, top_passages) VALUES (?, ?, ?, ?, ?)",
              (time.time(), query, result, best_score, json.dumps(top_passages, ensure_ascii=False)))
    conn.commit()

# -----------------------------
# Load docs from local ./docs or GitHub raw (if configured)
# -----------------------------
def list_docs_from_local(path=LOCAL_DOCS_PATH):
    p = pathlib.Path(path)
    if not p.exists():
        return []
    # accept pdf/docx/txt
    files = sorted([str(x) for x in p.iterdir() if x.suffix.lower() in {".pdf", ".docx", ".doc", ".txt"}])
    return files

def download_github_docs(filenames: List[str], base_raw_url: str):
    import requests
    saved = []
    for fn in filenames:
        url = base_raw_url.rstrip("/") + "/" + fn
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(fn).suffix)
                tmp.write(r.content)
                tmp.close()
                saved.append((fn, tmp.name))
        except Exception as e:
            st.warning(f"İndirme hatası: {fn} -> {e}")
    return saved

# -----------------------------
# LLM summarization helper (OpenAI) - STRICT: only use provided passages
# -----------------------------
def llm_summarize_openai(passages: List[Dict], query: str, max_tokens=300):
    if not OPENAI_AVAILABLE:
        return None
    # Compose strict prompt
    combined = "\n\n---\n\n".join([f"[SOURCE: {p['source']} - chunk {p['chunk_index']}]\n{p['text']}" for p in passages])
    system = ("Sen bir mevzuat/hukuk asistanısın. Aşağıdaki **sadece** verilen pasajlardan özet çıkaracaksın. "
              "Yeni bilgi ekleme, genelleme yaparken bile sadece bu pasajlarda açıkça geçenlerden hareket et. "
              "Eğer pasajlarda sorunun kesin cevabı yoksa 'VERI YETERSIZ' yaz. "
              "Cevabı Türkçe ver. Kısa ve net maddeler halinde yaz. Her maddenin sonunda kaynak belirt (dosya adı ve chunk).")
    prompt = f"{system}\n\nSoru: {query}\n\nPasajlar:\n{combined}\n\nCevap:"
    # OpenAI API call
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    openai.api_key = key
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Replace with available model in your account
            messages=[{"role":"system","content":system},
                      {"role":"user","content":f"Soru: {query}\n\nPasajlar:\n{combined}\n\nCevap:"}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        text = resp['choices'][0]['message']['content'].strip()
        return text
    except Exception as e:
        st.error(f"LLM özetleme hatası: {e}")
        return None

# -----------------------------
# App state
# -----------------------------
if "store" not in st.session_state:
    st.session_state.store = DocStore()
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "sim_threshold" not in st.session_state:
    st.session_state.sim_threshold = DEFAULT_SIM_THRESHOLD

# init DB
conn = init_db(DB_PATH)

# -----------------------------
# UI: Sidebar - admin
# -----------------------------
st.sidebar.title("Admin")
st.sidebar.markdown("Doküman yükleme / indeksleme ve ayarlar")

# Docs source selection
use_github = st.sidebar.checkbox("GitHub docs klasöründen indir (raw url kullan)", value=False)
if use_github:
    st.sidebar.markdown("**GITHUB RAW BASE URL** (örnek: https://raw.githubusercontent.com/USER/REPO/main/docs)")
    gh_url = st.sidebar.text_input("Base raw URL", value=GITHUB_RAW_DOCS_BASE)
    gh_filenames = st.sidebar.text_area("Dosya adlarını virgülle ayırın (örn: dosya1.pdf,dosya2.pdf)", value="")
else:
    gh_url = ""
    gh_filenames = ""

if st.sidebar.button("Dokümanları yükle ve indeksle"):
    st.session_state.store.clear()
    loaded_files = []
    if use_github and gh_url and gh_filenames.strip():
        fns = [f.strip() for f in gh_filenames.split(",") if f.strip()]
        saved = download_github_docs(fns, gh_url)
        for fname, localpath in saved:
            n = st.session_state.store.add_document(localpath, fname)
            loaded_files.append((fname, localpath, n))
    else:
        local_files = list_docs_from_local(LOCAL_DOCS_PATH)
        for p in local_files:
            fname = os.path.basename(p)
            n = st.session_state.store.add_document(p, fname)
            loaded_files.append((fname, p, n))
    st.session_state.store.build_index()
    st.session_state.docs_loaded = True
    st.sidebar.success(f"{len(loaded_files)} dosya işlendi ve indeks oluşturuldu.")

if st.sidebar.button("Indeksi temizle"):
    st.session_state.store.clear()
    st.session_state.docs_loaded = False
    st.sidebar.success("Indeks temizlendi.")

# similarity threshold adjust
st.sidebar.markdown("Benzerlik eşik ayarı (0.0 - 1.0). Eşik altındaysa 'VERI YETERSIZ' döner.")
st.session_state.sim_threshold = st.sidebar.slider("Benzerlik eşik (cosine)", 0.4, 0.95, float(st.session_state.sim_threshold), step=0.01)

# show index stats
if st.sidebar.button("Indeks durumu"):
    store = st.session_state.store
    if store.index is None:
        st.sidebar.info("Indeks yok.")
    else:
        st.sidebar.success(f"Passage sayısı: {len(store.passages)} - Embedding dim: {store.embed_dim}")

# -----------------------------
# Main UI
# -----------------------------
st.title("📚 Sosyal Bilimler Enstitüsü — Mevzuat Chatbot Pro")
st.write("Bu chatbot yalnızca `docs/` içindeki yüklü belgeler çerçevesinde cevap üretir. Halisünasyon yok — belgede destek yoksa açıkça bildirir.")

col1, col2 = st.columns([3,1])

with col1:
    query = st.text_area("Soru (Türkçe önerilir):", height=120)
    k = st.number_input("Getirilecek en fazla pasaj sayısı (top-k):", min_value=1, max_value=10, value=3)
    btn = st.button("Sorgula")
with col2:
    st.markdown("### Bilgiler")
    st.write(f"- Yüklü passage sayısı: {len(st.session_state.store.passages)}")
    st.write(f"- Benzerlik eşiği: {st.session_state.sim_threshold:.2f}")
    st.write("- Özetleme (LLM) kullanmak istiyorsanız OpenAI API anahtarını Streamlit Secrets veya env olarak ekleyin.")

# If user pressed query
if btn and query.strip():
    store = st.session_state.store
    if store.index is None or len(store.passages)==0:
        st.error("Henüz doküman yüklenmedi veya indeks oluşturulmadı. Sidebar'dan 'Dokümanları yükle ve indeksle' ile başlatın.")
    else:
        results = store.query(query, top_k=k)
        if not results:
            st.error("Arama başarısız: indeks yok veya boş.")
        else:
            best_score = results[0][0]
            st.write(f"En yüksek benzerlik skoru: **{best_score:.3f}**")
            # Eğer skor eşikten düşükse RET
            if best_score < st.session_state.sim_threshold:
                st.warning("Yüklü belgelerde güvenilir ve doğrudan destekleyen bilgi bulunamadı. Aşağıda en ilgili pasajlar gösteriliyor, fakat cevap **VERI YETERSIZ** olarak sunulacaktır.")
                # show passages
                top_passages = []
                for score, passage in results:
                    top_passages.append({"score": score, "source": passage["source"], "chunk_index": passage["chunk_index"], "text": passage["text"][:3000]})
                # log
                log_query(conn, query, "NOT_FOUND", best_score, top_passages)
                for p in top_passages:
                    st.markdown(f"**Kaynak:** {p['source']} — chunk {p['chunk_index']} — *benzerlik: {p['score']:.3f}*")
                    st.text(p['text'])
                st.error("Cevap: **VERI YETERSIZ** — Yüklü belgelerde doğrudan destek bulunamadı.")
            else:
                # Found: göster ve opsiyonel özet
                # Filter passages with decent score
                selected = [passage for score, passage in results if score >= 0.25]
                top_passages = [{"score": float(score), "source": p["source"], "chunk_index": p["chunk_index"], "text": p["text"]} for score, p in results]
                # Log as FOUND
                log_query(conn, query, "FOUND", best_score, top_passages)
                st.success("Yeterli destek bulundu — aşağıdaki pasajlar kaynak olarak sunulmuştur.")
                for score, passage in results:
                    st.markdown(f"**Kaynak:** {passage['source']} — chunk {passage['chunk_index']} — *benzerlik: {score:.3f}*")
                    st.write(passage['text'][:4000])
                # Özetleme: sadece eğer OpenAI varsa göster düğme
                openai_key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
                if OPENAI_AVAILABLE and openai_key:
                    if st.button("LLM ile Sıkı Özetle (sadece gösterilen pasajlar kullanılır)"):
                        # prepare passages for LLM
                        summary = llm_summarize_openai([p for _,p in results], query)
                        if summary:
                            st.markdown("### LLM Özet (sadece gösterilen pasajlara dayanır)")
                            st.write(summary)
                        else:
                            st.error("LLM özetleme başarısız.")
                else:
                    st.caption("LLM özetleme için OpenAI kütüphanesi veya API anahtarı bulunamadı. Opsiyonel: OPENAI_API_KEY ekleyin.")

# -----------------------------
# Footer: deploy / instructions
# -----------------------------
st.markdown("---")
st.markdown("**Deploy notları:** 1) Local: `streamlit run app_pro.py`. 2) Streamlit Cloud: repo'ya push → Share Streamlit → set start file `app_pro.py`. 3) Eğer OpenAI kullanacaksanız, Streamlit Secrets ya da env var olarak `OPENAI_API_KEY` ekleyin.")
