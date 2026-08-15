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
from collections import deque

# ============================================================
# EFFETTI COSMETICI (sovrapposti)
# ============================================================

def apply_shake_effect(frame, intensity):
    h, w = frame.shape[:2]
    max_offset = int(15 * intensity)
    dx = random.randint(-max_offset, max_offset)
    dy = random.randint(-max_offset, max_offset)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return frame

def apply_pixelate_effect(frame, intensity):
    h, w = frame.shape[:2]
    pixel_size = max(2, int(2 + 20 * intensity))
    small = cv2.resize(frame, (max(1, w // pixel_size), max(1, h // pixel_size)),
                       interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_color_distortion(frame, intensity):
    b, g, r = cv2.split(frame)
    shift_b = int(12 * intensity * random.choice([-1, 1]))
    shift_g = int(9 * intensity * random.choice([-1, 1]))
    shift_r = int(6 * intensity * random.choice([-1, 1]))
    b = np.roll(b, shift_b, axis=0)
    g = np.roll(g, shift_g, axis=1)
    r = np.roll(r, shift_r, axis=(0, 1))
    frame = cv2.merge((b, g, r))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + intensity * 0.8), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_tv_noise_effect(frame, intensity):
    h, w = frame.shape[:2]
    num_lines = int(15 * intensity)
    for _ in range(num_lines):
        y = random.randint(0, h - 1)
        thickness = random.randint(1, 2)
        color = [random.randint(100, 255)] * 3
        cv2.line(frame, (0, y), (w, y), color, thickness)
    noise = np.random.randint(0, int(60 * intensity) + 1, (h, w, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)
    if random.random() < intensity * 0.3:
        frame = (frame * 0.5).astype(np.uint8)
    return frame

def apply_beat_flash(frame, intensity):
    if intensity > 0.8:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame

# ============================================================
# EFFETTI DISTRUTTIVI (dentro al video)
# ============================================================

def apply_displacement_map(frame, intensity, seed_frame=None):
    """
    Displacement map: ogni pixel viene spostato in base ai valori
    di un'altra immagine (o del frame stesso blurred).
    """
    h, w = frame.shape[:2]
    if seed_frame is not None:
        guide = cv2.cvtColor(seed_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    guide = cv2.GaussianBlur(guide, (21, 21), 0)
    guide = (guide - guide.min()) / (guide.max() - guide.min() + 1e-6)

    amplitude = intensity * 40
    map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)

    map_x += (guide - 0.5) * amplitude * 2
    map_y += (guide - 0.5) * amplitude

    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)

def apply_pixel_sort(frame, intensity, mode="luminosity"):
    """
    Pixel sorting: riordina i pixel lungo le righe in base
    alla luminosità (o canale). Crea scie e lacerazioni.
    """
    h, w = frame.shape[:2]
    result = frame.copy()
    threshold_low = int((1 - intensity) * 200)
    threshold_high = 255

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_rows = max(1, int(h * intensity))
    row_indices = random.sample(range(h), min(num_rows, h))

    for y in row_indices:
        row = gray[y, :]
        mask = (row >= threshold_low) & (row <= threshold_high)
        segs = []
        in_seg = False
        start = 0
        for x in range(w):
            if mask[x] and not in_seg:
                start = x
                in_seg = True
            elif not mask[x] and in_seg:
                segs.append((start, x))
                in_seg = False
        if in_seg:
            segs.append((start, w))

        for s, e in segs:
            if e - s > 2:
                seg_pixels = frame[y, s:e]
                lum = gray[y, s:e]
                order = np.argsort(lum)
                result[y, s:e] = seg_pixels[order]

    return result

def apply_datamosh(frame, prev_frame, intensity):
    """
    Simula il datamoshing: il frame precedente "invade" quello attuale
    tramite differenza di movimento, creando ghost e smear.
    """
    if prev_frame is None or prev_frame.shape != frame.shape:
        return frame

    h, w = frame.shape[:2]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=2, winsize=15,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
    )

    map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)

    warp_strength = intensity * 3.0
    map_x_warp = np.clip(map_x + flow[:, :, 0] * warp_strength, 0, w - 1)
    map_y_warp = np.clip(map_y + flow[:, :, 1] * warp_strength, 0, h - 1)

    warped_prev = cv2.remap(prev_frame, map_x_warp, map_y_warp,
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    alpha = np.clip(intensity * 0.85, 0, 0.9)
    return cv2.addWeighted(frame, 1 - alpha, warped_prev, alpha, 0)

def apply_frame_echo(frame, echo_buffer, intensity):
    """
    Frame echo / feedback loop: mescola il frame attuale con
    versioni precedenti deformate, creando eco visivo accumulato.
    """
    if not echo_buffer:
        return frame

    result = frame.astype(np.float32)
    decay = 0.7
    weight_total = 1.0

    for i, old_frame in enumerate(reversed(echo_buffer)):
        if old_frame.shape != frame.shape:
            continue
        w_echo = intensity * (decay ** (i + 1))
        result += old_frame.astype(np.float32) * w_echo
        weight_total += w_echo

    result = np.clip(result / weight_total * (1 + intensity * 0.3), 0, 255)
    return result.astype(np.uint8)

def apply_digital_corruption_effect(frame, intensity):
    """
    Corruzione digitale a blocchi: blocchi di pixel vengono
    riposizionati e colorati in modo casuale.
    """
    h, w = frame.shape[:2]
    block_size = max(4, int(24 * (1 - intensity * 0.7)))
    num_blocks = int(intensity * 30)

    for _ in range(num_blocks):
        y = random.randint(0, h - block_size - 1)
        x = random.randint(0, w - block_size - 1)
        bh = min(block_size + random.randint(0, 16), h - y)
        bw = min(block_size + random.randint(0, 16), w - x)
        if bh <= 0 or bw <= 0:
            continue

        offset_x = random.randint(-int(w * 0.08), int(w * 0.08))
        offset_y = random.randint(-int(h * 0.08), int(h * 0.08))
        new_x = max(0, min(w - bw, x + offset_x))
        new_y = max(0, min(h - bh, y + offset_y))

        block = frame[y:y + bh, x:x + bw].copy()

        if random.random() < 0.6 and block.size > 0:
            hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + random.randint(60, 180)) % 180
            block = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if new_y + bh <= h and new_x + bw <= w:
            frame[new_y:new_y + bh, new_x:new_x + bw] = block

    return frame

def apply_glitch_lines(frame, intensity):
    """
    Glitch a righe orizzontali: fasce del frame vengono shiftate
    lateralmente di quantità variabili, lacerando l'immagine.
    """
    h, w = frame.shape[:2]
    result = frame.copy()
    num_bands = int(intensity * 20)

    for _ in range(num_bands):
        y_start = random.randint(0, h - 1)
        band_h = random.randint(1, max(2, int(h * 0.06)))
        y_end = min(y_start + band_h, h)
        shift = int(random.uniform(-intensity * 60, intensity * 60))
        result[y_start:y_end] = np.roll(frame[y_start:y_end], shift, axis=1)

    # Chromatic aberration sulle righe shiftate
    if intensity > 0.4:
        b, g, r = cv2.split(result)
        shift_chr = int(intensity * 8)
        b = np.roll(b, shift_chr)
        r = np.roll(r, -shift_chr)
        result = cv2.merge((b, g, r))

    return result

# ============================================================
# ANALISI AUDIO
# ============================================================

def analyze_audio_cached(audio_path: str):
    """Analizza l'audio dal path (hashabile per la cache di Streamlit)."""
    try:
        y, sr = sf.read(audio_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        f, t, Sxx = signal.spectrogram(y, fs=sr, nperseg=2048, noverlap=1024)

        def band_energy(f_low, f_high):
            idx = np.where((f >= f_low) & (f <= f_high))[0]
            if len(idx) == 0:
                return np.zeros(Sxx.shape[1])
            e = np.mean(Sxx[idx, :], axis=0)
            rng = e.max() - e.min()
            return (e - e.min()) / rng if rng > 0 else np.zeros_like(e)

        bass_e   = band_energy(20, 200)
        mid_e    = band_energy(200, 2000)
        treble_e = band_energy(2000, 10000)

        # BPM via onset strength su finestra breve
        window = min(len(y), sr * 30)  # max 30s per il calcolo
        y_short = y[:window]
        # onset envelope semplice: derivata dell'energia
        hop = 512
        energy = np.array([
            np.sum(y_short[i:i+hop]**2)
            for i in range(0, len(y_short) - hop, hop)
        ])
        diff = np.diff(energy)
        diff = np.maximum(diff, 0)
        # autocorrelazione sull'onset
        corr = np.correlate(diff, diff, mode='full')
        corr = corr[len(corr)//2:]
        sr_hop = sr / hop
        min_lag = int(sr_hop * 60 / 200)
        max_lag = int(sr_hop * 60 / 60)
        search = corr[min_lag:max_lag]
        if len(search) > 0:
            best_lag = np.argmax(search) + min_lag
            bpm = sr_hop * 60 / best_lag
        else:
            bpm = 120.0

        total_time = len(y) / sr
        beat_interval = 60 / bpm
        beat_times = np.arange(0, total_time, beat_interval)

        return bass_e, mid_e, treble_e, t, float(bpm), beat_times, sr

    except Exception as e:
        st.error(f"Errore nell'analisi audio: {e}")
        return None, None, None, None, 120.0, np.array([]), 44100

# Cache Streamlit basata sul path (stringa, hashabile)
analyze_audio_cached = st.cache_data(analyze_audio_cached)

# ============================================================
# MERGE AUDIO/VIDEO
# ============================================================

def merge_audio_video(video_path, audio_path, output_path, fps=24):
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        if result.returncode != 0:
            st.warning("FFmpeg non disponibile. Output senza audio.")
            return video_path

        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '192k',
            '-r', str(fps),
            '-shortest',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return output_path
        else:
            st.warning(f"Errore FFmpeg: {result.stderr[-300:]}")
            return video_path
    except Exception as e:
        st.warning(f"Errore merge audio/video: {e}")
        return video_path

def make_preview_480p(video_path, temp_dir):
    """
    Genera una versione leggera (altezza 480px) del video solo per l'anteprima
    a schermo: velocizza il caricamento nel browser senza toccare il file
    finale che viene offerto in download (che resta a piena qualità).
    """
    preview_path = os.path.join(temp_dir, "preview_480p.mp4")
    try:
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', "scale=-2:480:flags=fast_bilinear",
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '30',
            '-c:a', 'aac', '-b:a', '96k',
            preview_path
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and os.path.exists(preview_path) and os.path.getsize(preview_path) > 0:
            return preview_path
    except Exception:
        pass
    return video_path  # fallback: mostra il video originale se la conversione fallisce

# ============================================================
# UI HELPER
# ============================================================

def effect_controls(label, key, default_bass=0.0, default_mid=0.0, default_treble=0.0):
    with st.expander(label, expanded=False):
        enabled = st.checkbox("Abilitato", True, key=f"{key}_on")
        max_i = st.slider("Intensità Max", 0.0, 2.0, 1.0, key=f"{key}_max")
        c1, c2, c3 = st.columns(3)
        bass   = c1.slider("Bassi",  0.0, 1.0, default_bass,   key=f"{key}_bass")
        mid    = c2.slider("Medie",  0.0, 1.0, default_mid,    key=f"{key}_mid")
        treble = c3.slider("Alte",   0.0, 1.0, default_treble, key=f"{key}_treble")
    return enabled, bass, mid, treble, max_i

# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(
        page_title="GlitchFusion Pro",
        layout="wide",
        page_icon="🎬",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .app-header { font-size:48px; font-weight:bold; color:#8A2BE2;
                  text-shadow:2px 2px 8px rgba(0,0,0,0.3); margin-bottom:0; }
    .app-sub    { font-size:16px; color:#888; margin-top:0; margin-bottom:16px; }
    </style>
    <h1 class="app-header">GlitchFusion Pro</h1>
    <p class="app-sub">by Loop507 — v3.0</p>
    """, unsafe_allow_html=True)

    st.markdown("### Carica un video e un brano audio per generare effetti sincronizzati")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ Parametri Globali")
        output_fps = st.slider("FPS Output", 10, 30, 24)

        st.subheader("Formato Video")
        aspect_ratio = st.selectbox(
            "Rapporto d'Aspetto",
            ["Originale", "1:1 (Quadrato)", "9:16 (Verticale)", "16:9 (Orizzontale)"]
        )

        st.subheader("🎵 Audio")
        include_audio = st.checkbox("Includi audio in output", True)

        st.markdown("---")
        st.subheader("🎨 Effetti Cosmetici")
        en_shake,   shake_b,   shake_m,   shake_t,   shake_max   = effect_controls("🫨 Shake",              "shake",   0.8, 0.0, 0.0)
        en_pixel,   pixel_b,   pixel_m,   pixel_t,   pixel_max   = effect_controls("🟦 Pixel Art",          "pixel",   0.0, 0.6, 0.0)
        en_tv,      tv_b,      tv_m,      tv_t,      tv_max      = effect_controls("📺 TV Noise",           "tv",      0.0, 0.0, 0.5)
        en_color,   color_b,   color_m,   color_t,   color_max   = effect_controls("🌈 Distorsione Colori", "color",   0.6, 0.3, 0.2)
        en_flash,   *_                                            = effect_controls("⚡ Flash Battiti",      "flash",   0.0, 0.0, 0.0)

        st.markdown("---")
        st.subheader("💥 Effetti Distruttivi")
        en_disp,    disp_b,    disp_m,    disp_t,    disp_max    = effect_controls("🌊 Displacement Map",   "disp",    0.5, 0.3, 0.1)
        en_sort,    sort_b,    sort_m,    sort_t,    sort_max    = effect_controls("🧬 Pixel Sorting",      "sort",    0.0, 0.5, 0.7)
        en_mosh,    mosh_b,    mosh_m,    mosh_t,    mosh_max    = effect_controls("👻 Datamosh",           "mosh",    0.7, 0.2, 0.0)
        en_echo,    echo_b,    echo_m,    echo_t,    echo_max    = effect_controls("🔁 Frame Echo",         "echo",    0.4, 0.4, 0.2)
        en_corrupt, corr_b,    corr_m,    corr_t,    corr_max    = effect_controls("🧱 Corruzione Digitale","corrupt", 0.8, 0.1, 0.0)
        en_glitch,  glit_b,    glit_m,    glit_t,    glit_max    = effect_controls("⚡ Glitch Lines",       "glitch",  0.6, 0.3, 0.2)

    # ---- Upload ----
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📤 Caricamento File")
        video_file = st.file_uploader("Video (MP4, AVI)", type=["mp4", "avi"])
        audio_file = st.file_uploader("Audio (MP3, WAV) [Opzionale]", type=["mp3", "wav"])

    with col2:
        st.subheader("ℹ️ Info")
        st.info("""
        **Istruzioni:**
        1. Carica un video (obbligatorio)
        2. Carica un audio opzionale
        3. Regola gli effetti nella sidebar
        4. Clicca **Elabora Video**
        5. Scarica il risultato
        """)
        st.warning("⚠️ Consigliati video di 10–30 secondi per tempi ragionevoli.")
        if video_file:
            st.success("✅ Video caricato!")
        if audio_file:
            st.success("✅ Audio caricato!")

    # Validazione dimensioni
    for f, label in [(video_file, "video"), (audio_file, "audio")]:
        if f and f.size > 200 * 1024 * 1024:
            st.error(f"❌ Il file {label} supera i 200MB.")
            return

    if not st.button("🎬 Elabora Video", type="primary", use_container_width=True) or not video_file:
        return

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()

        # Salva video
        video_path = os.path.join(temp_dir, "input_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.getvalue())

        # Gestione audio
        audio_path = None
        if audio_file:
            ext = os.path.splitext(audio_file.name)[1].lower()
            raw_audio = os.path.join(temp_dir, f"input_audio{ext}")
            with open(raw_audio, "wb") as f:
                f.write(audio_file.getvalue())

            # Converti sempre in WAV per soundfile
            audio_path = os.path.join(temp_dir, "audio.wav")
            cmd = ['ffmpeg', '-y', '-i', raw_audio,
                   '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0 or not os.path.exists(audio_path):
                st.warning("⚠️ Conversione audio fallita, procedo senza.")
                audio_path = None

        elif include_audio:
            audio_path = os.path.join(temp_dir, "extracted_audio.wav")
            cmd = ['ffmpeg', '-y', '-i', video_path, '-vn',
                   '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0 or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                st.warning("⚠️ Nessun audio nel video.")
                audio_path = None

        # Analisi audio
        bass_e = mid_e = treble_e = t_arr = None
        beat_times = np.array([])
        tempo = 120.0

        if audio_path:
            with st.spinner("🎵 Analisi audio in corso..."):
                bass_e, mid_e, treble_e, t_arr, tempo, beat_times, _ = analyze_audio_cached(audio_path)
            if bass_e is not None:
                st.success(f"🎶 BPM stimati: {tempo:.1f} | Battiti: {len(beat_times)}")
            else:
                st.warning("⚠️ Analisi audio fallita. Effetti applicati in modalità statica (senza sincronizzazione al beat).")

        # Apertura video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Impossibile aprire il video.")
            return

        fps        = cap.get(cv2.CAP_PROP_FPS) or 24
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.info(f"📹 {width}×{height} @ {fps:.1f} FPS — {total_frames} frame")

        # Dimensioni output
        if aspect_ratio == "1:1 (Quadrato)":
            out_w = out_h = min(width, height)
        elif aspect_ratio == "9:16 (Verticale)":
            out_w = int(height * 9 / 16)
            out_h = height
        elif aspect_ratio == "16:9 (Orizzontale)":
            out_w = width
            out_h = int(width * 9 / 16)
        else:
            out_w, out_h = width, height

        temp_video = os.path.join(temp_dir, "temp_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, output_fps, (out_w, out_h))
        if not out.isOpened():
            st.error("❌ Impossibile creare il video di output.")
            return

        progress_bar = st.progress(0)
        status_text  = st.empty()

        frame_count = 0
        beat_index  = 0
        prev_frame  = None
        echo_buffer = deque(maxlen=6)  # buffer per frame echo

        # Conversione framerate: se output_fps != fps originale, scrivere
        # i frame 1:1 comprime/allunga la durata (bug: video accorciato/allungato).
        # Con questo accumulatore i frame vengono duplicati o saltati in modo
        # da preservare la durata reale del video sorgente.
        frame_ratio = (output_fps / fps) if fps > 0 else 1.0
        write_carry = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / fps
            frame_count += 1

            # Beat detection
            beat_intensity = 0.0
            while beat_index < len(beat_times) and current_time >= beat_times[beat_index]:
                beat_intensity = 1.0
                beat_index += 1

            # Energia bande per il frame corrente
            def band_val(energy_arr):
                if energy_arr is None or t_arr is None or len(t_arr) == 0 or t_arr[-1] <= 0:
                    return 0.0
                idx = min(int(current_time / t_arr[-1] * len(t_arr)), len(energy_arr) - 1)
                return float(energy_arr[idx])

            if bass_e is None and mid_e is None and treble_e is None:
                # Nessun audio disponibile (o analisi fallita): non azzerare
                # bv/mv/tv, altrimenti la condizione "energia > peso*0.5" e'
                # sempre falsa e nessun effetto scatta mai. Uso un valore
                # costante cosi' gli effetti restano attivi (modalita' statica,
                # senza reattivita' al beat).
                bv = mv = tv = 1.0
            else:
                bv = band_val(bass_e)
                mv = band_val(mid_e)
                tv = band_val(treble_e)

            def intensity(e_b, w_b, e_m, w_m, e_t, w_t, max_i):
                raw = max(e_b * w_b, e_m * w_m, e_t * w_t)
                return min(max_i, raw)

            try:
                # --- EFFETTI COSMETICI ---
                if en_shake and (bv > shake_b * 0.5 or mv > shake_m * 0.5 or tv > shake_t * 0.5):
                    frame = apply_shake_effect(frame, intensity(bv, shake_b, mv, shake_m, tv, shake_t, shake_max))

                if en_pixel and (bv > pixel_b * 0.5 or mv > pixel_m * 0.5 or tv > pixel_t * 0.5):
                    frame = apply_pixelate_effect(frame, intensity(bv, pixel_b, mv, pixel_m, tv, pixel_t, pixel_max))

                if en_tv and (bv > tv_b * 0.5 or mv > tv_m * 0.5 or tv > tv_t * 0.5):
                    frame = apply_tv_noise_effect(frame, intensity(bv, tv_b, mv, tv_m, tv, tv_t, tv_max))

                if en_color and (bv > color_b * 0.5 or mv > color_m * 0.5 or tv > color_t * 0.5):
                    frame = apply_color_distortion(frame, intensity(bv, color_b, mv, color_m, tv, color_t, color_max))

                if en_flash and beat_intensity > 0.8:
                    frame = apply_beat_flash(frame, beat_intensity)

                # --- EFFETTI DISTRUTTIVI ---
                if en_disp and (bv > disp_b * 0.5 or mv > disp_m * 0.5 or tv > disp_t * 0.5):
                    i_val = intensity(bv, disp_b, mv, disp_m, tv, disp_t, disp_max)
                    frame = apply_displacement_map(frame, i_val, prev_frame)

                if en_sort and (bv > sort_b * 0.5 or mv > sort_m * 0.5 or tv > sort_t * 0.5):
                    i_val = intensity(bv, sort_b, mv, sort_m, tv, sort_t, sort_max)
                    frame = apply_pixel_sort(frame, i_val)

                if en_mosh and (bv > mosh_b * 0.5 or mv > mosh_m * 0.5 or tv > mosh_t * 0.5):
                    i_val = intensity(bv, mosh_b, mv, mosh_m, tv, mosh_t, mosh_max)
                    frame = apply_datamosh(frame, prev_frame, i_val)

                if en_echo and (bv > echo_b * 0.5 or mv > echo_m * 0.5 or tv > echo_t * 0.5):
                    i_val = intensity(bv, echo_b, mv, echo_m, tv, echo_t, echo_max)
                    frame = apply_frame_echo(frame, echo_buffer, i_val)

                if en_corrupt and (bv > corr_b * 0.5 or mv > corr_m * 0.5 or tv > corr_t * 0.5):
                    i_val = intensity(bv, corr_b, mv, corr_m, tv, corr_t, corr_max)
                    frame = apply_digital_corruption_effect(frame, i_val)

                if en_glitch and (bv > glit_b * 0.5 or mv > glit_m * 0.5 or tv > glit_t * 0.5):
                    i_val = intensity(bv, glit_b, mv, glit_m, tv, glit_t, glit_max)
                    frame = apply_glitch_lines(frame, i_val)

            except Exception as e:
                st.warning(f"⚠️ Errore effetti al frame {frame_count}: {e}")

            # Aggiorna buffer
            prev_frame = frame.copy()
            echo_buffer.append(frame.copy())

            # Crop aspect ratio
            if aspect_ratio != "Originale":
                fh, fw = frame.shape[:2]
                if aspect_ratio == "1:1 (Quadrato)":
                    s = min(fw, fh)
                    frame = frame[(fh-s)//2:(fh-s)//2+s, (fw-s)//2:(fw-s)//2+s]
                elif aspect_ratio == "9:16 (Verticale)":
                    tw = int(fh * 9 / 16)
                    if tw > fw:
                        th = int(fw * 16 / 9)
                        frame = frame[(fh-th)//2:(fh-th)//2+th, :]
                    else:
                        frame = frame[:, (fw-tw)//2:(fw-tw)//2+tw]
                elif aspect_ratio == "16:9 (Orizzontale)":
                    th = int(fw * 9 / 16)
                    if th > fh:
                        tw = int(fh * 16 / 9)
                        frame = frame[:, (fw-tw)//2:(fw-tw)//2+tw]
                    else:
                        frame = frame[(fh-th)//2:(fh-th)//2+th, :]

            frame = cv2.resize(frame, (out_w, out_h))

            # Scrive il frame N volte (0, 1 o più) in base al rapporto tra
            # fps originale e fps di output, così la durata resta corretta.
            write_carry += frame_ratio
            while write_carry >= 1.0:
                out.write(frame)
                write_carry -= 1.0

            progress_bar.progress(min(frame_count / max(total_frames, 1), 1.0))
            if frame_count % 30 == 0:
                status_text.text(f"⚙️ Frame {frame_count}/{total_frames}")

        cap.release()
        out.release()

        # Merge audio
        final_path = os.path.join(temp_dir, "final_output.mp4")
        if include_audio and audio_path:
            status_text.text("🎵 Unione audio e video...")
            final_video = merge_audio_video(temp_video, audio_path, final_path, output_fps)
            if final_video == final_path and os.path.exists(final_path):
                st.success("✅ Audio unito!")
            else:
                st.warning("⚠️ Audio non unito (FFmpeg non disponibile o errore). Output solo video.")
        else:
            final_video = temp_video

        status_text.empty()
        st.balloons()
        st.success("✅ Elaborazione completata!")

        with open(final_video, "rb") as vf:
            video_bytes = vf.read()

        status_text.text("🖼️ Genero anteprima leggera (480p)...")
        preview_path = make_preview_480p(final_video, temp_dir)
        status_text.empty()

        st.subheader("🎬 Anteprima (480p)")
        if preview_path != final_video:
            with open(preview_path, "rb") as pf:
                st.video(pf.read())
            st.caption("L'anteprima è compressa a 480p per un caricamento più veloce. Il file scaricato è alla qualità originale.")
        else:
            st.video(video_bytes)

        st.download_button(
            "💾 Scarica Video",
            video_bytes,
            file_name="glitchfusion_output.mp4",
            mime="video/mp4",
            use_container_width=True,
            key="download_btn"
        )

    except Exception as e:
        st.error(f"❌ Errore generale: {e}")
        st.exception(e)
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    st.markdown("---")
    st.markdown("*GlitchFusion Pro v3.0 — Trasforma i tuoi video in esperienze audiovisive uniche*")

if __name__ == "__main__":
    main()
