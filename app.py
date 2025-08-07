import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import random
import soundfile as sf
import shutil
import subprocess
from scipy import signal

# ==========================
# FUNZIONI PER GLI EFFETTI (rimangono identiche)
# ==========================

# ... [Le funzioni apply_* rimangono identiche] ...

# ==========================
# INTERFACCIA STREAMLIT MODIFICATA
# ==========================

def main():
    st.set_page_config(
        page_title="GlitchFusion Pro", 
        layout="wide",
        page_icon="🎬",
        initial_sidebar_state="expanded"
    )
    
    # Header personalizzato
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
        .stProgress .st-bo {
            background-color: #8A2BE2;
        }
        .effect-controls {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
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
        
        # Nuova opzione per abilitare/disabilitare audio
        st.subheader("🎵 Controllo Audio")
        include_audio = st.checkbox("Includi audio in output", True)
        
        st.subheader("🎚 Controlli Effetti")
        tempo_sensitivity = st.slider("Sensibilità al Tempo", 0.1, 1.0, 0.7)
        
        # Funzione per creare controlli effetti
        def create_effect_controls(name, default_bass, default_mid, default_treble):
            with st.container():
                st.markdown(f'<div class="effect-controls">', unsafe_allow_html=True)
                st.markdown(f"**{name}**")
                cols = st.columns(3)
                with cols[0]:
                    bass = st.slider("Bassi", 0.0, 1.0, default_bass, key=f"{name}_bass")
                with cols[1]:
                    mid = st.slider("Medie", 0.0, 1.0, default_mid, key=f"{name}_mid")
                with cols[2]:
                    treble = st.slider("Alte", 0.0, 1.0, default_treble, key=f"{name}_treble")
                st.markdown('</div>', unsafe_allow_html=True)
                return bass, mid, treble
        
        # Controlli individuali per ogni effetto
        shake_bass, shake_mid, shake_treble = create_effect_controls("Shake", 0.8, 0.0, 0.0)
        pixel_bass, pixel_mid, pixel_treble = create_effect_controls("Pixel Art", 0.0, 0.6, 0.0)
        tv_bass, tv_mid, tv_treble = create_effect_controls("TV Noise", 0.0, 0.0, 0.5)
        color_bass, color_mid, color_treble = create_effect_controls("Distorsione Colori", 0.7, 0.3, 0.2)
        corruption_bass, corruption_mid, corruption_treble = create_effect_controls("Corruzione Digitale", 0.8, 0.2, 0.1)
        glitch_bass, glitch_mid, glitch_treble = create_effect_controls("Glitch", 0.6, 0.4, 0.2)
        
        st.subheader("🎛 Abilita/Disabilita Effetti")
        col1, col2 = st.columns(2)
        with col1:
            enable_shake = st.checkbox("Shake", True, key="enable_shake")
            enable_pixelate = st.checkbox("Pixel Art", True, key="enable_pixel")
            enable_tv_noise = st.checkbox("TV Noise", True, key="enable_tv")
        with col2:
            enable_color = st.checkbox("Distorsione Colori", True, key="enable_color")
            enable_corruption = st.checkbox("Corruzione Digitale", True, key="enable_corrupt")
            enable_glitch = st.checkbox("Glitch", True, key="enable_glitch")
            enable_flash = st.checkbox("Flash Battiti", True, key="enable_flash")
    
    # Area principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Caricamento File")
        video_file = st.file_uploader(
            "Carica video (MP4, AVI)", 
            type=["mp4", "avi"],
            help="Dimensione massima: 200MB"
        )
        audio_file = st.file_uploader(
            "Carica brano audio (MP3, WAV) [Opzionale]", 
            type=["mp3", "wav"],
            help="Dimensione massima: 200MB. Lascia vuoto per usare l'audio del video"
        )
    
    with col2:
        st.subheader("ℹ️ Informazioni")
        st.info("""
        **Istruzioni:**
        1. Carica un video (obbligatorio)
        2. Carica un audio opzionale (o usa quello del video)
        3. Regola i parametri nella sidebar
        4. Clicca "Elabora Video"
        5. Scarica il risultato
        """)
        st.warning("⚠️ Per video lunghi l'elaborazione potrebbe richiedere tempo. Consigliati video di 10-30 secondi.")
        
        if video_file:
            st.success("✅ Video caricato correttamente!")
            if audio_file:
                st.success("✅ Audio caricato correttamente!")
            else:
                st.info("ℹ️ Verrà usato l'audio del video (se presente)")
    
    # Controlli di validazione
    if video_file and video_file.size > 200 * 1024 * 1024:  # 200MB
        st.error("❌ Il video è troppo grande. Massimo 200MB.")
        return
    
    if audio_file and audio_file.size > 200 * 1024 * 1024:  # 200MB
        st.error("❌ Il file audio è troppo grande. Massimo 200MB.")
        return

    if st.button("🎬 Elabora Video", type="primary", use_container_width=True) and video_file:
        temp_dir = None
        try:
            with st.spinner("Analisi audio e elaborazione video..."):
                # Crea directory temporanea
                temp_dir = tempfile.mkdtemp()
                
                # Salvataggio video
                video_path = os.path.join(temp_dir, f"video_{hash(video_file.name)}.mp4")
                with open(video_path, "wb") as f:
                    f.write(video_file.getvalue())
                
                # Gestione audio
                audio_path = None
                extracted_audio = False
                
                # Caso 1: audio fornito dall'utente
                if audio_file:
                    audio_path = os.path.join(temp_dir, f"audio_{hash(audio_file.name)}.wav")
                    with open(audio_path, "wb") as f:
                        f.write(audio_file.getvalue())
                
                # Caso 2: estrai audio dal video (se richiesto)
                elif include_audio:
                    audio_path = os.path.join(temp_dir, "extracted_audio.wav")
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', video_path,
                        '-vn', '-acodec', 'pcm_s16le',
                        '-ar', '44100', '-ac', '2',
                        audio_path
                    ]
                    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                    
                    if result.returncode != 0 or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                        st.warning("⚠️ Impossibile estrarre audio dal video. Procedo senza audio.")
                        audio_path = None
                    else:
                        extracted_audio = True

                # Analisi audio (solo se disponibile)
                bass_energy, mid_energy, treble_energy, t = None, None, None, None
                beat_times = []
                tempo = 120  # Default
                
                if audio_path:
                    try:
                        # Caricamento audio
                        y, sr = sf.read(audio_path)
                        
                        # Analisi audio
                        bass_energy, mid_energy, treble_energy, t = analyze_audio(y, sr)
                        
                        if bass_energy is None:
                            st.warning("⚠️ Errore nell'analisi audio. Effetti disattivati")
                        else:
                            # Calcolo BPM
                            tempo = calculate_bpm(y, sr)
                            
                            # Calcolo battiti
                            beat_interval = 60 / tempo
                            total_time = len(y) / sr
                            beat_times = np.arange(0, total_time, beat_interval)
                            
                            st.success(f"🎶 BPM Stimati: {tempo:.1f} | Battiti: {len(beat_times)}")
                    except Exception as e:
                        st.warning(f"⚠️ Errore nell'analisi audio: {str(e)}. Effetti disattivati")

                # Apertura video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("❌ Errore nell'apertura del video")
                    return
                    
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                st.info(f"📹 Video: {width}x{height}, {fps:.1f} FPS, {total_frames} frame")
                
                # Calcolo dimensioni output
                if aspect_ratio == "1:1 (Quadrato)":
                    out_width = out_height = min(width, height)
                elif aspect_ratio == "9:16 (Verticale)":
                    out_width = int(height * 9 / 16)
                    out_height = height
                elif aspect_ratio == "16:9 (Orizzontale)":
                    out_width = width
                    out_height = int(width * 9 / 16)
                else:  # Originale
                    out_width = width
                    out_height = height
                
                # Preparazione output
                temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_video_path, fourcc, output_fps, (out_width, out_height))
                
                if not out.isOpened():
                    st.error("❌ Errore nella creazione del video di output")
                    return
                    
                # Elaborazione frame-by-frame
                frame_count = 0
                beat_index = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Calcolo tempo corrente
                    current_time = frame_count / fps
                    frame_count += 1
                    
                    # Controllo battiti (solo se disponibile analisi audio)
                    beat_intensity = 0
                    if bass_energy is not None and beat_index < len(beat_times) and current_time >= beat_times[beat_index]:
                        beat_intensity = 1.0
                        beat_index += 1
                    
                    # Calcolo intensità bande di frequenza
                    bass_value = mid_value = treble_value = 0
                    if bass_energy is not None and len(t) > 0:
                        time_idx = min(int(current_time / t[-1] * len(t)), len(bass_energy)-1)
                        bass_value = bass_energy[time_idx] if len(bass_energy) > 0 else 0
                        mid_value = mid_energy[time_idx] if len(mid_energy) > 0 else 0
                        treble_value = treble_energy[time_idx] if len(treble_energy) > 0 else 0
                    
                    # Applicazione effetti
                    try:
                        # Shake (bassi)
                        if enable_shake and bass_value > shake_bass * 0.5:
                            frame = apply_shake_effect(frame, bass_value * shake_bass)
                        
                        # Pixel Art (medie)
                        if enable_pixelate and mid_value > pixel_mid * 0.5:
                            frame = apply_pixelate_effect(frame, mid_value * pixel_mid)
                        
                        # TV Noise (alte)
                        if enable_tv_noise and treble_value > tv_treble * 0.5:
                            frame = apply_tv_noise_effect(frame, treble_value * tv_treble)
                        
                        # Distorsione colori (tutte le bande)
                        if enable_color and (bass_value > color_bass * 0.5 or 
                                           mid_value > color_mid * 0.5 or 
                                           treble_value > color_treble * 0.5):
                            intensity = max(bass_value * color_bass, 
                                          mid_value * color_mid, 
                                          treble_value * color_treble)
                            frame = apply_color_distortion(frame, intensity)
                        
                        # Flash battiti
                        if enable_flash and beat_intensity > 0.8:
                            frame = apply_beat_flash(frame, beat_intensity)
                        
                        # Corruzione digitale (principalmente bassi)
                        if enable_corruption and bass_value > corruption_bass * 0.7:
                            frame = apply_digital_corruption_effect(frame, bass_value * corruption_bass)
                        
                        # Glitch (tutte le bande)
                        if enable_glitch and (bass_value > glitch_bass * 0.4 or 
                                             mid_value > glitch_mid * 0.4 or 
                                             treble_value > glitch_treble * 0.4):
                            intensity = max(bass_value * glitch_bass, 
                                          mid_value * glitch_mid, 
                                          treble_value * glitch_treble)
                            frame = apply_glitch_effect(frame, intensity)
                    except Exception as e:
                        st.warning(f"⚠️ Errore effetti al frame {frame_count}: {e}")
                    
                    # Ritaglio in base al rapporto d'aspetto
                    if aspect_ratio != "Originale":
                        h, w = frame.shape[:2]
                        
                        if aspect_ratio == "1:1 (Quadrato)":
                            size = min(w, h)
                            x = (w - size) // 2
                            y = (h - size) // 2
                            frame = frame[y:y+size, x:x+size]
                        
                        elif aspect_ratio == "9:16 (Verticale)":
                            target_width = int(h * 9 / 16)
                            if target_width > w:
                                target_height = int(w * 16 / 9)
                                y = (h - target_height) // 2
                                frame = frame[y:y+target_height, :]
                            else:
                                x = (w - target_width) // 2
                                frame = frame[:, x:x+target_width]
                        
                        elif aspect_ratio == "16:9 (Orizzontale)":
                            target_height = int(w * 9 / 16)
                            if target_height > h:
                                target_width = int(h * 16 / 9)
                                x = (w - target_width) // 2
                                frame = frame[:, x:x+target_width]
                            else:
                                y = (h - target_height) // 2
                                frame = frame[y:y+target_height, :]
                    
                    # Ridimensionamento finale
                    frame = cv2.resize(frame, (out_width, out_height))
                    out.write(frame)
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                cap.release()
                out.release()
                
                # Unione audio al video (se richiesto e disponibile)
                final_output_path = os.path.join(temp_dir, "final_output.mp4")
                
                if include_audio and audio_path:
                    status_text.text("🎵 Unione audio e video...")
                    final_video = merge_audio_video(temp_video_path, audio_path, final_output_path, output_fps)
                    st.success("✅ Audio unito correttamente!")
                else:
                    final_video = temp_video_path
                    if include_audio:
                        st.warning("⚠️ Audio non disponibile per l'unione")
                    else:
                        st.info("ℹ️ Generazione video senza audio")
                
                st.balloons()
                st.success("✅ Elaborazione completata!")
                
                # Leggi il file finale
                with open(final_video, "rb") as video_file:
                    video_bytes = video_file.read()
                
                # Visualizzazione risultato
                st.subheader("🎬 Anteprima Video")
                st.video(video_bytes)
                
                # Download
                filename = "glitchfusion_output.mp4"
                st.download_button(
                    "💾 Scarica Video", 
                    video_bytes, 
                    file_name=filename,
                    mime="video/mp4",
                    use_container_width=True,
                    key="download-btn"
                )
                
        except Exception as e:
            st.error(f"❌ Errore generale: {str(e)}")
            st.exception(e)
            
        finally:
            # Pulizia garantita
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    st.warning(f"⚠️ Errore nella pulizia: {e}")

    # Footer
    st.markdown("---")
    st.markdown("*GlitchFusion Pro v2.0 - Trasforma i tuoi video in esperienze audiovisive uniche*")

if __name__ == "__main__":
    main()
