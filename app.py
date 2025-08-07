import streamlit as st
import numpy as np
import librosa
import cv2
import tempfile
import os
import random
from moviepy.editor import VideoFileClip, AudioFileClip

# ==========================
# FUNZIONI PER GLI EFFETTI
# ==========================

# ... [Le funzioni per gli effetti rimangono identiche] ...

# ==========================
# INTERFACCIA STREAMLIT CON NOME PERSONALIZZATO
# ==========================

def main():
    st.set_page_config(
        page_title="GlitchFusion Pro", 
        layout="wide",
        page_icon="🎬"
    )
    
    # Header personalizzato con nome e sottotitolo
    st.markdown(
        """
        <style>
        .app-header {
            font-size: 50px;
            font-weight: bold;
            color: #8A2BE2;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 0;
        }
        .app-subheader {
            font-size: 18px;
            color: #666;
            margin-top: 0;
            margin-bottom: 20px;
        }
        </style>
        
        <h1 class="app-header">GlitchFusion Pro</h1>
        <p class="app-subheader">by Loop507</p>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### Carica un video e un brano audio per generare effetti sincronizzati")
    
    # Sidebar per i parametri principali
    with st.sidebar:
        st.header("⚙️ Parametri Globali")
        output_fps = st.slider("FPS Output", 10, 30, 24)
        
        st.subheader("Formato Video")
        aspect_ratio = st.selectbox(
            "Rapporto d'Aspetto",
            ["Originale", "1:1 (Quadrato)", "9:16 (Verticale)", "16:9 (Orizzontale)"],
            index=0
        )
        
        # ... [resto della sidebar rimane identico] ...

    # Area principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Caricamento File")
        video_file = st.file_uploader("Carica video (MP4, AVI)", type=["mp4", "avi"])
        audio_file = st.file_uploader("Carica brano audio (MP3, WAV)", type=["mp3", "wav"])
    
    with col2:
        st.subheader("ℹ️ Informazioni")
        st.info("""
        **Istruzioni:**
        1. Carica un video e un brano audio
        2. Regola i parametri nella sidebar
        3. Clicca "Elabora Video"
        4. Scarica il risultato
        """)
        st.warning("⚠️ Per video lunghi l'elaborazione potrebbe richiedere tempo. Consigliati video di 10-30 secondi.")
    
    if st.button("🎬 Elabora Video", type="primary", use_container_width=True) and video_file and audio_file:
        with st.spinner("Analisi audio e elaborazione video..."):
            # ... [resto del codice rimane identico] ...
            
            st.balloons()
            st.success("✅ Elaborazione completata!")
            
            # Visualizzazione risultato
            st.subheader("🎬 Anteprima Video")
            st.video(final_output)
            
            # Download con stile personalizzato
            st.markdown(
                """
                <style>
                .download-btn {
                    background-color: #8A2BE2;
                    color: white;
                    padding: 10px 24px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 8px;
                    border: none;
                    width: 100%;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            with open(final_output, "rb") as f:
                st.download_button(
                    "💾 Scarica Video", 
                    f.read(), 
                    file_name="glitchfusion_video.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    key="download-btn"
                )
            
            # Pulizia
            try:
                os.unlink(video_path)
                os.unlink(audio_path)
                os.unlink(output_path)
                os.unlink(final_output)
            except Exception as e:
                st.warning(f"⚠️ Errore nella pulizia dei file temporanei: {e}")

if __name__ == "__main__":
    main()
