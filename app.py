import streamlit as st
import os
import time
import glob
from google import genai
from google.genai import types

# --- 1. AYARLAR ---
st.set_page_config(page_title="Enstitü Mevzuat Asistanı", page_icon="🎓")
st.title("🎓 Enstitü Mevzuat Asistanı")
st.markdown("Yönetmelik, Usul ve Esaslar çerçevesinde sorularınızı yanıtlar.")

# API Key Kontrolü
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("API Anahtarı bulunamadı. Lütfen Streamlit Secrets ayarlarını yapın.")
    st.stop()

client = genai.Client(api_key=api_key)

# --- 2. DOSYA YÜKLEME FONKSİYONU (Önbellekli) ---
@st.cache_resource
def upload_files_to_gemini():
    uploaded_files = []
    # 'belgeler' klasöründeki tüm .pdf dosyalarını bul
    pdf_files = glob.glob("belgeler/*.pdf")
    
    if not pdf_files:
        st.error("HATA: 'belgeler' klasöründe hiç PDF bulunamadı! Lütfen klasörü kontrol edin.")
        return []

    status_area = st.empty()
    status_area.info(f"{len(pdf_files)} adet belge sisteme yükleniyor, lütfen bekleyin...")
    
    for pdf_path in pdf_files:
        try:
            # DÜZELTME BURADA YAPILDI: 'path' yerine 'file' yazıldı.
            file_upload = client.files.upload(file=pdf_path)
            uploaded_files.append(file_upload)
            print(f"Yüklendi: {pdf_path}")
        except Exception as e:
            st.error(f"Dosya yüklenirken hata oluştu ({pdf_path}): {e}")

    # Dosyaların işlenmesini bekle (Google tarafında 'ACTIVE' olmalı)
    while True:
        all_active = True
        for f in uploaded_files:
            remote_file = client.files.get(name=f.name)
            if remote_file.state != "ACTIVE":
                all_active = False
                break
        
        if all_active:
            break
        time.sleep(2) 
        
    status_area.success("Tüm belgeler analiz edildi ve sisteme eklendi! ✅")
    time.sleep(1)
    status_area.empty()
    
    return uploaded_files

# --- 3. UYGULAMA BAŞLATMA ---

# Dosyaları yükle ve değişkene ata
files_context = upload_files_to_gemini()

# Eğer dosya yüklenemediyse durdur
if not files_context:
    st.stop()

# Sohbet Geçmişi Başlatma
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. SOHBET DÖNGÜSÜ ---
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    
    # Kullanıcı mesajını göster
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Model Ayarları ve Sistem Talimatı
    # PDF dosyalarını burada modele 'tool' veya 'content' olarak veriyoruz.
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        system_instruction=[
            types.Part.from_text(text="""Rol: Sen Dokuz Eylül Üniversitesi Sosyal Bilimler Enstitüsü mevzuat asistanısın.
            Görevin: Soruları SADECE sana verilen PDF dosyalarına dayanarak cevapla.
            
            Kurallar:
            1. Asla belgelerin dışına çıkma. Bilgi yoksa "Yönetmeliklerde bu bilgi yok" de.
            2. MUTLAKA REFERANS GÖSTER: Her cevabın sonuna (Belge Adı, Madde No) ekle.
            3. Resmi ve yardımsever ol.
            """)
        ],
    )

    # Gemini'ye gönderilecek içerik listesi
    # İlk önce dosyaları, sonra sohbet geçmişini ekliyoruz.
    contents_to_send = []
    
    # 1. Dosyaları ekle (Sadece ilk mesajda veya her seferinde bağlam olarak verilebilir)
    # Gemini 1.5 Flash'ın hafızası geniştir, dosyaları her istekte hatırlatmak en garantisidir.
    for f in files_context:
        contents_to_send.append(types.Content(
            role="user",
            parts=[types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type)]
        ))

    # 2. Sohbet geçmişini ekle
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        contents_to_send.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # Cevabı Üret
    with st.chat_message("assistant"):
        try:
            stream = client.models.generate_content_stream(
                model="gemini-1.5-flash",
                contents=contents_to_send,
                config=generate_content_config,
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
