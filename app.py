# ==========================
# MODIFICHE DA APPLICARE
# ==========================

# All'interno della funzione create_effect_controls:
def create_effect_controls(name, default_bass, default_mid, default_treble):
    with st.container():
        st.markdown(f'<div class="effect-controls">', unsafe_allow_html=True)
        st.markdown(f"**{name}**")
        
        # Aggiunta di slider per intensità massima
        max_intensity = st.slider(
            "Intensità Max", 
            0.0, 2.0, 1.0,  # Min, Max, Default
            key=f"{name}_max",
            help="Regola l'intensità massima dell'effetto"
        )
        
        cols = st.columns(3)
        with cols[0]:
            bass = st.slider("Bassi", 0.0, 1.0, default_bass, key=f"{name}_bass")
        with cols[1]:
            mid = st.slider("Medie", 0.0, 1.0, default_mid, key=f"{name}_mid")
        with cols[2]:
            treble = st.slider("Alte", 0.0, 1.0, default_treble, key=f"{name}_treble")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return bass, mid, treble, max_intensity  # Aggiunto max_intensity nel return

# Nella sezione parametri effetti (sidebar):
shake_bass, shake_mid, shake_treble, shake_max = create_effect_controls("Shake", 0.8, 0.0, 0.0)
pixel_bass, pixel_mid, pixel_treble, pixel_max = create_effect_controls("Pixel Art", 0.0, 0.6, 0.0)
tv_bass, tv_mid, tv_treble, tv_max = create_effect_controls("TV Noise", 0.0, 0.0, 0.5)
color_bass, color_mid, color_treble, color_max = create_effect_controls("Distorsione Colori", 0.7, 0.3, 0.2)
corruption_bass, corruption_mid, corruption_treble, corruption_max = create_effect_controls("Corruzione Digitale", 0.8, 0.2, 0.1)
glitch_bass, glitch_mid, glitch_treble, glitch_max = create_effect_controls("Glitch", 0.6, 0.4, 0.2)

# Nel loop di elaborazione frame (dove si applicano gli effetti):
# Modifica per ogni effetto (esempio per shake):
if enable_shake and bass_value > shake_bass * 0.5:
    intensity = min(shake_max, bass_value * shake_bass)  # Limita all'intensità massima
    frame = apply_shake_effect(frame, intensity)

# Applica lo stesso pattern a tutti gli altri effetti:
# Pixel Art
if enable_pixelate and mid_value > pixel_mid * 0.5:
    intensity = min(pixel_max, mid_value * pixel_mid)
    frame = apply_pixelate_effect(frame, intensity)

# TV Noise
if enable_tv_noise and treble_value > tv_treble * 0.5:
    intensity = min(tv_max, treble_value * tv_treble)
    frame = apply_tv_noise_effect(frame, intensity)

# Distorsione Colori
if enable_color and (bass_value > color_bass * 0.5 or 
                   mid_value > color_mid * 0.5 or 
                   treble_value > color_treble * 0.5):
    intensity = min(color_max, max(
        bass_value * color_bass, 
        mid_value * color_mid, 
        treble_value * color_treble
    ))
    frame = apply_color_distortion(frame, intensity)

# Corruzione Digitale
if enable_corruption and bass_value > corruption_bass * 0.7:
    intensity = min(corruption_max, bass_value * corruption_bass)
    frame = apply_digital_corruption_effect(frame, intensity)

# Glitch
if enable_glitch and (bass_value > glitch_bass * 0.4 or 
                     mid_value > glitch_mid * 0.4 or 
                     treble_value > glitch_treble * 0.4):
    intensity = min(glitch_max, max(
        bass_value * glitch_bass, 
        mid_value * glitch_mid, 
        treble_value * glitch_treble
    ))
    frame = apply_glitch_effect(frame, intensity)
