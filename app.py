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
# FUNZIONI PER GLI EFFETTI
# ==========================

def apply_shake_effect(frame, intensity):
    h, w = frame.shape[:2]
    max_offset = int(15 * intensity)
    
    dx = random.randint(-max_offset, max_offset)
    dy = random.randint(-max_offset, max_offset)
    
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    frame = cv2.warpAffine(frame, M, (w, h))
    
    frame = cv2.copyMakeBorder(frame, 
                              abs(dy), abs(dy), 
                              abs(dx), abs(dx), 
                              cv2.BORDER_CONSTANT, 
                              value=[0, 0, 0])
    
    frame = frame[abs(dy):h+abs(dy), abs(dx):w+abs(dx)]
    return frame

def apply_pixelate_effect(frame, intensity):
    h, w = frame.shape[:2]
    pixel_size = max(1, int(1 + 15 * intensity))
    small = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_NEAREST)
    frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame

def apply_color_distortion(frame, intensity):
    b, g, r = cv2.split(frame)
    
    shift_b = int(10 * intensity * random.choice([-1, 1]))
    shift_g = int(8 * intensity * random.choice([-1, 1]))
    shift_r = int(6 * intensity * random.choice([-1, 1]))
    
    b = np.roll(b, shift_b, axis=0)
    g = np.roll(g, shift_g, axis=1)
    r = np.roll(r, shift_r, axis=(0, 1))
    
    frame = cv2.merge((b, g, r))
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + intensity), 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return frame

def apply_tv_noise_effect(frame, intensity):
    h, w = frame.shape[:2]
    
    num_lines = int(20 * intensity)
    for _ in range(num_lines):
        y = random.randint(0, h-1)
        thickness = random.randint(1, 3)
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        cv2.line(frame, (0, y), (w, y), color, thickness)
    
    noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    frame = cv2.addWeighted(frame, 1 - intensity*0.5, noise, intensity*0.5, 0)
    
    if random.random() < intensity:
        frame = frame // 2
    
    return frame

def apply_digital_corruption_effect(frame, intensity):
    h, w = frame.shape[:2]
    
    block_size = max(4, int(32 * (1 - intensity)))
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if random.random() < intensity * 0.8:
                block_h = min(block_size + random.randint(-10, 20), h - y)
                block_w = min(block_size + random.randint(-10, 20), w - x)
                
                if block_h <= 0 or block_w <= 0:
                    continue
                
                offset_x = random.randint(-int(w * 0.1), int(w * 0.1))
                offset_y = random.randint(-int(h * 0.1), int(h * 0.1))
                
                block = frame[y:y+block_h, x:x+block_w].copy()
                new_x = max(0, min(w - block_w, x + offset_x))
                new_y = max(0, min(h - block_h, y + offset_y))
                
                if random.random() < 0.7 and block.size > 0:
                    block = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
                    block[:, :, 0] = (block[:, :, 0] + random.randint(0, 180)) % 180
                    block = cv2.cvtColor(block, cv2.COLOR_HSV2BGR)
                
                if new_y + block_h <= h and new_x + block_w <= w:
                    frame[new_y:new_y+block_h, new_x:new_x+block_w] = block
    
    return frame

def apply_glitch_effect(frame, intensity):
    h, w = frame.shape[:2]
    
    if intensity > 0.3:
        shift = int(intensity * 50)
        for i in range(0, h, 5):
            if i < h:
                frame[i] = np.roll(frame[i], shift * (i % 3 - 1), axis=0)
    
    if intensity > 0.4:
        b, g, r = cv2.split(frame)
        frame = cv2.merge((
            np.roll(b, int(intensity * 10)),
            np.roll(g, int(intensity * -5)),
            r
        ))
    
    return frame

def apply_beat_flash(frame, intensity):
    if intensity > 0.8:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame

def merge_audio_video(video_path, audio_path, output_path, fps=24):
    """Unisce video e audio usando ffmpeg"""
    try:
        # Controlla se ffmpeg è disponibile
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            st.warning("FFmpeg non disponibile. Il video sarà generato senza audio.")
            return video_path
        
        # Comando ffmpeg per unire video e audio
        cmd = [
            'ffmpeg', '-y',  # -y per sovrascrivere file esistenti
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-b:a', '192k',
            '-r', str(fps),
            '-shortest',  # Usa la durata del file più corto
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return output_path
        else:
            st.warning(f"Errore FFmpeg: {result.stderr}")
            return video_path
            
    except Exception as e:
        st.warning(f"Errore nell'unione audio/video: {e}")
        return video_path

@st.cache_data
def calculate_bpm(y, sr):
    """Calcola il BPM usando autocorrelazione"""
    try:
        # Calcola l'autocorrelazione
        corr = np.correlate(y, y, mode='full')
        corr = corr[len(corr)//2:]
        
        # Trova i picchi
        min_peak_distance = sr // 4  # Almeno 0.25 secondi tra i picchi
        peaks = []
        for i in range(1, len(corr)-1):
            if corr[i] > corr[i-1] and corr[i] > corr[i+1] and corr[i] > np.max(corr)*0.1:
                if not peaks or i - peaks[-1] > min_peak_distance:
                    peaks.append(i)
        
        # Calcola il BPM
        if len(peaks) > 1:
            peak_diffs = np.diff(peaks)
            avg_peak_diff = np.mean(peak_diffs)
            bpm = 60 / (avg_peak_diff / sr)
            return min(200, max(60, bpm))  # Limita tra 60 e 200 BPM
        return 120  # Valore di default
    except Exception as e:
        st.warning(f"Errore nel calcolo BPM: {e}")
        return 120

@st.cache_data
def analyze_audio(audio_data, sr):
    """Analizza l'audio e restituisce le energie delle bande"""
    try:
        # Converti in mono se necessario
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Calcolo spettrogramma
        f, t, Sxx = signal.spectrogram(audio_data, fs=sr, nperseg=1024, noverlap=512)
        
        # Definizione bande di frequenza
        bass_band = (20, 200)
        mid_band = (200, 2000)
        treble_band = (2000, 10000)
        
        # Calcolo indici
        bass_idx = np.where((f >= bass_band[0]) & (f <= bass_band[1]))[0]
        mid_idx = np.where((f >= mid_band[0]) & (f <= mid_band[1]))[0]
        treble_idx = np.where((f >= treble_band[0]) & (f <= treble_band[1]))[0]
        
        # Calcolo energie
        bass_energy = np.mean(Sxx[bass_idx, :], axis=0) if len(bass_idx) > 0 else np.zeros(Sxx.shape[1])
        mid_energy = np.mean(Sxx[mid_idx, :], axis=0) if len(mid_idx) > 0 else np.zeros(Sxx.shape[1])
        treble_energy = np.mean(Sxx[treble_idx, :], axis=0) if len(treble_idx) > 0 else np.zeros(Sxx.shape[1])
        
        # Normalizzazione
        def normalize(arr):
            if np.max(arr) - np.min(arr) > 0:
                return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
            return np.zeros_like(arr)
        
        bass_energy = normalize(bass_energy)
        mid_energy = normalize(mid_energy)
        treble_energy = normalize(treble_energy)
        
        return bass_energy, mid_energy, treble_energy, t
    except Exception as e:
        st.error(f"Errore nell'analisi audio: {e}")
        return None, None, None, None

# ==========================
# INTERFACCIA STREAMLIT
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
        
        st.subheader("🎚 Controlli Effetti")
        tempo_sensitivity = st.slider("Sensibilità al Tempo", 0.1, 1.0, 0.7)
        bass_intensity = st.slider("Intensità Bassi (Shake)", 0.0, 1.0, 0.8)
        mid_intensity = st.slider("Intensità Medie (Pixel Art)", 0.0, 1.0, 0.6)
        treble_intensity = st.slider("Intensità Alte (TV Noise)", 0.0, 1.0, 0.5)
        color_dist_intensity = st.slider("Intensità Distorsione Colori", 0.0, 1.0, 0.7)
        corruption_intensity = st.slider("Intensità Corruzione Digitale", 0.0, 1.0, 0.8)
        
        st.subheader("🎛 Abilita/Disabilita Effetti")
        col1, col2 = st.columns(2)
        with col1:
            enable_shake = st.checkbox("Shake", True)
            enable_pixelate = st.checkbox("Pixel Art", True)
            enable_tv_noise = st.checkbox("TV Noise", True)
        with col2:
            enable_color = st.checkbox("Distorsione Colori", True)
            enable_corruption = st.checkbox("Corruzione Digitale", True)
            enable_glitch = st.checkbox("Glitch", True)
            enable_flash = st.checkbox("Flash Battiti", True)
    
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
            "Carica brano audio (MP3, WAV)", 
            type=["mp3", "wav"],
            help="Dimensione massima: 200MB"
        )
    
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
        
        if video_file and audio_file:
            st.success("✅ File caricati correttamente!")
    
    # Controlli di validazione
    if video_file:
        if video_file.size > 200 * 1024 * 1024:  # 200MB
            st.error("❌ Il video è troppo grande. Massimo 200MB.")
            return
    
    if audio_file:
        if audio_file.size > 200 * 1024 * 1024:  # 200MB
            st.error("❌ Il file audio è troppo grande. Massimo 200MB.")
            return

    if st.button("🎬 Elabora Video", type="primary", use_container_width=True) and video_file and audio_file:
        temp_dir = None
        try:
            with st.spinner("Analisi audio e elaborazione video..."):
                # Crea directory temporanea
                temp_dir = tempfile.mkdtemp()
                
                # Salvataggio file con nomi sicuri
                video_path = os.path.join(temp_dir, f"video_{hash(video_file.name)}.mp4")
                audio_path = os.path.join(temp_dir, f"audio_{hash(audio_file.name)}.wav")
                
                # Scrivi i file
                with open(video_path, "wb") as f:
                    f.write(video_file.getvalue())
                
                with open(audio_path, "wb") as f:
                    f.write(audio_file.getvalue())

                # Analisi audio
                try:
                    # Caricamento audio
                    y, sr = sf.read(audio_path)
                    
                    # Analisi audio con cache
                    bass_energy, mid_energy, treble_energy, t = analyze_audio(y, sr)
                    
                    if bass_energy is None:
                        st.error("Errore nell'analisi audio")
                        return
                    
                    # Calcolo BPM
                    tempo = calculate_bpm(y, sr)
                    
                    # Calcolo battiti
                    beat_interval = 60 / tempo
                    total_time = len(y) / sr
                    beat_times = np.arange(0, total_time, beat_interval)
                    
                    st.success(f"🎶 BPM Stimati: {tempo:.1f} | Battiti: {len(beat_times)}")
                except Exception as e:
                    st.error(f"Errore nell'analisi audio: {str(e)}")
                    return

                # Apertura video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Errore nell'apertura del video")
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
                    st.error("Errore nella creazione del video di output")
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
                    
                    # Controllo battiti
                    beat_intensity = 0
                    if beat_index < len(beat_times) and current_time >= beat_times[beat_index]:
                        beat_intensity = 1.0
                        beat_index += 1
                    
                    # Calcolo intensità bande di frequenza
                    if len(t) > 0:
                        time_idx = min(int(current_time / t[-1] * len(t)), len(bass_energy)-1)
                        bass_value = bass_energy[time_idx] if len(bass_energy) > 0 else 0
                        mid_value = mid_energy[time_idx] if len(mid_energy) > 0 else 0
                        treble_value = treble_energy[time_idx] if len(treble_energy) > 0 else 0
                    else:
                        bass_value = mid_value = treble_value = 0
                    
                    # Aggiornamento stato
                    if frame_count % 10 == 0:
                        status_text.text(f"📊 Frame: {frame_count}/{total_frames} | Progresso: {frame_count/total_frames*100:.1f}%")
                    
                    # Applicazione effetti
                    try:
                        if enable_shake and bass_value > bass_intensity * 0.8:
                            frame = apply_shake_effect(frame, bass_value * bass_intensity)
                        
                        if enable_pixelate and mid_value > mid_intensity * 0.7:
                            frame = apply_pixelate_effect(frame, mid_value * mid_intensity)
                        
                        if enable_tv_noise and treble_value > treble_intensity * 0.6:
                            frame = apply_tv_noise_effect(frame, treble_value * treble_intensity)
                        
                        if enable_color and (beat_intensity > 0.5 or bass_value > color_dist_intensity * 0.7):
                            frame = apply_color_distortion(frame, color_dist_intensity)
                        
                        if enable_flash and beat_intensity > 0.8:
                            frame = apply_beat_flash(frame, beat_intensity)
                        
                        if enable_corruption and bass_value > corruption_intensity * 0.9:
                            frame = apply_digital_corruption_effect(frame, bass_value * corruption_intensity)
                        
                        if enable_glitch and bass_value > 0.5:
                            frame = apply_glitch_effect(frame, bass_value)
                    except Exception as e:
                        st.warning(f"Errore nell'applicazione effetti al frame {frame_count}: {e}")
                    
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
                
                # Unione audio al video usando FFmpeg
                status_text.text("🎵 Unione audio e video...")
                final_output_path = os.path.join(temp_dir, "final_output.mp4")
                
                # Prova a unire con FFmpeg
                final_video = merge_audio_video(temp_video_path, audio_path, final_output_path, output_fps)
                
                if final_video == temp_video_path:
                    st.warning("⚠️ Video generato senza audio (FFmpeg non disponibile)")
                    final_video = temp_video_path
                else:
                    st.success("✅ Audio unito correttamente!")
                
                st.balloons()
                st.success("✅ Elaborazione completata!")
                
                # Leggi il file finale
                with open(final_video, "rb") as video_file:
                    video_bytes = video_file.read()
                
                # Visualizzazione risultato
                st.subheader("🎬 Anteprima Video")
                st.video(video_bytes)
                
                # Download
                filename = f"glitchfusion_{tempo:.0f}bpm.mp4" if final_video != temp_video_path else "glitchfusion_no_audio.mp4"
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
                    st.success("🧹 Pulizia completata")
                except Exception as e:
                    st.warning(f"⚠️ Errore nella pulizia: {e}")

    # Footer
    st.markdown("---")
    st.markdown("*GlitchFusion Pro v2.0 - Trasforma i tuoi video in esperienze audiovisive uniche*")

if __name__ == "__main__":
    main()
