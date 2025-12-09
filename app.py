# --- 2. DOSYA YÜKLEME FONKSİYONU (Önbellekli) ---
@st.cache_resource
def upload_files_to_gemini():
    uploaded_files = []
    # 'belgeler' klasöründeki tüm .pdf dosyalarını bul
    pdf_files = glob.glob("docs/*.pdf")
    
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
