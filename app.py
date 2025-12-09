import streamlit as st
import os
from google import genai
from google.genai import types

# 1. Sayfa Ayarları
st.set_page_config(page_title="DEU Enstitü Asistanı", page_icon="🎓")

st.title("🎓 DEU Sosyal Bilimler Enstitüsü Asistanı")
st.markdown("Yüksek lisans ve doktora süreçlerinizle ilgili soruları sorabilirsiniz.")

# 2. API Anahtarını Al (Güvenli Yöntem)
# Streamlit Cloud'da 'Secrets' kısmından çekecek
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("API Anahtarı bulunamadı. Lütfen ayarlardan ekleyiniz.")
    st.stop()

client = genai.Client(api_key=api_key)

# 3. Sohbet Geçmişini Hatırla (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Ekrana eski mesajları yazdır
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Kullanıcıdan Girdi Al
if prompt := st.chat_input("Sorunuzu buraya yazın... (Örn: Tez savunma süresi nedir?)"):
    
    # Kullanıcı mesajını ekrana bas ve hafızaya al
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 5. Gemini'ye Gönderilecek İçeriği Hazırla
    # Sohbet geçmişini modele iletiyoruz ki bağlam kopmasın
    history_contents = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        history_contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # Model Ayarları (Sizin verdiğiniz koddan uyarlandı)
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3, # Daha tutarlı cevaplar için düşürdük
        system_instruction=[
            types.Part.from_text(text="""Rol: Sen Dokuz Eylül Üniversitesi Sosyal Bilimler Enstitüsü için geliştirilmiş, yüksek lisans ve doktora süreçlerinde uzmanlaşmış bir AI asistanısın.
            Görevin: Öğrencilerin sorularını SADECE sana yüklenen PDF dosyalarındaki bilgilere dayanarak cevaplamaktır.
            Kurallar:
            - Asla yüklenen belgelerin dışına çıkma. Bilgi yoksa uydurma.
            - Referans Zorunluluğu: Verdiğin her bilginin sonuna mutlaka kaynağını parantez içinde yaz. (Örnek: Lisansüstü Eğitim Yönetmeliği, Madde 24-b)
            - Cevapların resmi, nazik ve akademik bir dilde olsun."""),
        ],
    )

    # 6. Cevabı Üret ve Ekrana Bas
    with st.chat_message("assistant"):
        try:
            # Stream özelliği ile cevap yazılırken daktilo gibi aksın
            stream = client.models.generate_content_stream(
                model="gemini-1.5-flash", # Model adını standartlaştırdık
                contents=history_contents,
                config=generate_content_config,
            )
            
            # Streamlit'in stream yazma fonksiyonu
            response = st.write_stream(stream)
            
            # Cevabı hafızaya kaydet
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
