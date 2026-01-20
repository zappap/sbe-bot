import streamlit as st

# --- Sayfa ayarları ---
st.set_page_config(
    page_title="DEÜ Sosyal Bilimler Enstitüsü | Yapay Zeka Asistanı",
    layout="centered"
)

# --- NotebookLM linki ---
NOTEBOOKLM_URL = (
    "https://notebooklm.google.com/notebook/"
    "65aa8d8b-7e31-4897-9966-941aabf5656d"
)

# --- Başlık ---
st.markdown(
    "<h2 style='text-align:center;'>"
    "Dokuz Eylül Üniversitesi<br>"
    "Sosyal Bilimler Enstitüsü"
    "</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center;'>"
    "Yapay Zeka Destekli Bilgi Asistanı"
    "</h4>",
    unsafe_allow_html=True
)

st.divider()

# --- Açıklama ---
st.markdown("""
Bu sayfa, **DEÜ Sosyal Bilimler Enstitüsü** tarafından hazırlanmış  
**Yapay Zeka Destekli Bilgi Asistanına** erişim sağlamak amacıyla oluşturulmuştur.

Asistan; aşağıdaki mevzuat ve dokümanlar çerçevesinde,  
**ön bilgilendirme** amacıyla yanıt üretmektedir:

- YÖK Lisansüstü Eğitim ve Öğretim Yönetmeliği  
- DEÜ Lisansüstü Eğitim ve Öğretim Yönetmeliği  
- DEÜ SBE Lisansüstü Öğretim ve Sınav Uygulama Esasları  
- İlgili diğer resmî dokümanlar
""")

# --- Uyarı kutusu ---
st.warning("""
**Önemli Bilgilendirme**

Bu yapay zeka asistanı tarafından üretilen yanıtlar **resmî görüş niteliği taşımaz**.  
Bağlayıcı olan tek kaynak ilgili mevzuat ve Enstitü Yönetim Kurulu kararlarıdır.

Kesin ve bağlayıcı işlemler için ilgili Enstitü birimleri ile iletişime geçiniz.
""")

# --- Buton ---
st.markdown("<br>", unsafe_allow_html=True)

st.link_button(
    "🤖 Yapay Zeka Asistanını Aç",
    NOTEBOOKLM_URL,
    use_container_width=True
)

st.markdown("<br>", unsafe_allow_html=True)

# --- Alt bilgi ---
st.markdown(
    "<p style='text-align:center; font-size:12px; color:gray;'>"
    "© Dokuz Eylül Üniversitesi – Sosyal Bilimler Enstitüsü<br>"
    "Bu sayfa yalnızca yönlendirme ve bilgilendirme amaçlıdır."
    "</p>",
    unsafe_allow_html=True
)
