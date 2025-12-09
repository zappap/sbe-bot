import streamlit as st
from google import genai
import time

st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Gemini Chatbot")
st.write("Google Gemini API ile çalışan basit ama güçlü bir chatbot.")

# API anahtarı
api_key = st.text_input("Gemini API Key:", type="password")

# Sohbet geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

# Kullanıcı mesajı
user_msg = st.text_input("Mesajınızı yazın:")

# Gönder butonu
if st.button("Gönder"):
    if not api_key:
        st.error("API anahtarını girmelisiniz.")
    elif not user_msg.strip():
        st.error("Boş mesaj gönderemezsiniz.")
    else:
        # Gemini istemcisi
        client = genai.Client(api_key=api_key)

        # Kullanıcı mesajını ekle
        st.session_state.messages.append(("user", user_msg))

        try:
            # Gemini yanıtı
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=user_msg
            )
            bot_reply = response.text

            # Bot cevabını ekle
            st.session_state.messages.append(("bot", bot_reply))

        except Exception as e:
            st.error(f"Hata oluştu: {e}")

# Mesajları ekrana yaz
st.write("---")

for sender, msg in st.session_state.messages:
    if sender == "user":
        st.markdown(
            f"""
            <div style="text-align:right;">
                <div style="display:inline-block; background:#DCF8C6; padding:10px 14px; 
                border-radius:14px; margin:6px; max-width:80%;">{msg}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="text-align:left;">
                <div style="display:inline-block; background:#F1F0F0; padding:10px 14px; 
                border-radius:14px; margin:6px; max-width:80%;">🤖 {msg}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


