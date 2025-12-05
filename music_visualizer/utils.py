import librosa
import numpy as np

W_DIM = 512

def generate_power(y, sr, frame_length=512):
    spec = np.abs(librosa.cqt(y=y, sr=sr, hop_length=frame_length))

    spec_mean = np.mean(spec,axis=0)
    grad_mean = np.gradient(spec_mean)
    grad_mean = np.clip(grad_mean/np.max(grad_mean), 0, None)
    spec_mean = (spec_mean-np.min(spec_mean))/np.ptp(spec_mean)

    return spec, spec_mean, grad_mean

def generate_chroma(y, sr, frame_length=512):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=frame_length)
    return chroma

def generate_chroma_delta(chroma):
    chroma_delta = np.abs(np.diff(chroma, axis=1))
    chroma_delta = np.pad(chroma_delta, ((0, 0), (1, 0)), mode='constant')
    return chroma_delta

def get_sensitivity(jitter=0.5):
    return np.random.choice([1, 1-jitter], size=W_DIM)

def get_frame_lim(seconds, frame_length, batch_size):
    return int(np.floor(seconds*22050/frame_length/batch_size))

def new_update_dir(w, update_dir, tempo_sensitivity, truncation):
    update_dir[w >= truncation - tempo_sensitivity] = -1
    update_dir[w < -truncation + tempo_sensitivity] = 1
    return update_dir

def smooth(class_vectors,smooth_factor):
    class_vectors_sm = []
    for c in range(int(np.floor(len(class_vectors)/smooth_factor)-1)):  
        ci = int(c*smooth_factor)          
        cva = np.mean(class_vectors[ci:ci+smooth_factor],axis=0)
        cvb = np.mean(class_vectors[ci+smooth_factor:ci+smooth_factor*2],axis=0)
        for j in range(smooth_factor):
            terp_frac = j/(smooth_factor-1)
            cvc = cva * (1-terp_frac) + cvb * terp_frac                                          
            class_vectors_sm.append(cvc)
    return np.array(class_vectors_sm)


def generate_w_vectors(y, sr, 
                       tempo_sensitivity, 
                       pitch_sensitivity, 
                       melodic_sensitivity, 
                       sax_sensitivity, 
                       chroma_content_sensitivity, 
                       truncation, 
                       frame_length, 
                       drift_magnitude, 
                       num_ws, w_dim, jitter, preload=False):
    if preload:
        return np.load('saved_vectors/w_vectors.npy')

    # --- Feature Extraction (Audio Features) ---
    spec, spec_mean, grad_mean = generate_power(y, sr, frame_length)
    chroma = generate_chroma(y, sr, frame_length)
    chroma_delta = generate_chroma_delta(chroma)
    
    # --- Mid-Frequency and High-Frequency Energy ---
    cqt_freqs = librosa.cqt_frequencies(n_bins=spec.shape[0], fmin=librosa.note_to_hz('C1'))

    MF_START_FREQ = 300     
    MF_END_FREQ = 3500      
    mf_indices = np.where((cqt_freqs >= MF_START_FREQ) & (cqt_freqs <= MF_END_FREQ))[0]
    raw_mf_energy = np.sum(spec[mf_indices, :], axis=0)
    log_mf_energy = np.log1p(raw_mf_energy)
    if log_mf_energy.max() > log_mf_energy.min():
        mid_frequency_magnitude = (log_mf_energy - log_mf_energy.min()) / (log_mf_energy.max() - log_mf_energy.min())
    else:
        mid_frequency_magnitude = np.zeros_like(log_mf_energy)

    HF_START_FREQ = 5000 
    HF_END_FREQ = 10000
    hf_indices = np.where((cqt_freqs >= HF_START_FREQ) & (cqt_freqs <= HF_END_FREQ))[0]
    raw_hf_energy = np.sum(spec[hf_indices, :], axis=0)
    log_hf_energy = np.log1p(raw_hf_energy)
    if log_hf_energy.max() > log_hf_energy.min():
        hf_energy_magnitude = (log_hf_energy - log_hf_energy.min()) / (log_hf_energy.max() - log_hf_energy.min())
    else:
        hf_energy_magnitude = np.zeros_like(log_hf_energy)

    # 2. Initialize W vector
    w_base = np.clip(np.random.randn(w_dim) * truncation, -truncation, truncation)

    # CRITICAL FIX: Reset W_base much closer to center (0,0,0) for a stable start
    w_base *= 0.1
    
    ws = []
    update_dir = np.where(w_base < 0, 1, -1)
    
    # --- Jitter Vetting Logic ---
    JITTER_EXPECTED_MEAN = 1 - (jitter / 2) 
    TOLERANCE = 0.05 
    JITTER_VECTOR_COARSE = None # New separate jitter vector
    JITTER_VECTOR_MIDDLE = None # New separate jitter vector
    
    vetting_attempts = 0
    # Vet only one vector to save computation, then copy it.
    while True:
        JITTER_VECTOR_COARSE = get_sensitivity(jitter=jitter)
        vetting_attempts += 1
        if np.abs(np.mean(JITTER_VECTOR_COARSE) - JITTER_EXPECTED_MEAN) <= TOLERANCE:
            JITTER_VECTOR_MIDDLE = get_sensitivity(jitter=jitter) # Generate second one only after first succeeds
            break
        if vetting_attempts > 100:
            print(f"Warning: Failed to meet Jitter threshold after 100 attempts.")
            JITTER_VECTOR_MIDDLE = get_sensitivity(jitter=jitter) # Fallback
            break
    print(f"Jitter Vetting complete in {vetting_attempts} attempts.")
    
    w_coarse = w_base.copy() 
    w_middle = w_base.copy() 
    
    # --- DAMPING CONSTANTS ---
    DAMPING_FACTOR_ACCUMULATION = 1000 
    DAMPING_FACTOR_PUNCH = 5            
    SUPER_PUNCH_THRESHOLD = 0.5 
    
    # --- Damping Factors ---
    DAMPING_FACTOR_MELODY = DAMPING_FACTOR_ACCUMULATION * 1.5 
    
    # CRITICAL FIX: Middle layer movement must be much stronger than Coarse.
    # Division by 100 instead of 1000 allows for visible color shifts.
    DAMPING_FACTOR_SAX = 100.0
    DAMPING_FACTOR_CHROMA_CONTENT = 100.0
    # -----------------------------

    for f in range(len(spec_mean)):
        # --- 1. Calculate Magnitudes (Use externalized sensitivities) ---
        rhythm_magnitude = tempo_sensitivity * grad_mean[f]
        melodic_magnitude = spec_mean[f] * melodic_sensitivity 
        sax_magnitude = mid_frequency_magnitude[f] * sax_sensitivity 
        chroma_content_magnitude = np.sum(chroma[:, f]) * chroma_content_sensitivity 
        flicker_magnitude = hf_energy_magnitude[f] * 0.005 

        # Allocate drift: 90% for pose/color, 10% for detail
        base_drift_coarse = np.random.randn(w_dim) * drift_magnitude # Decouple drift
        base_drift_middle = np.random.randn(w_dim) * drift_magnitude
        
        # --- Adaptive Damping Logic for RHYTHM ---
        beat_strength = grad_mean[f] 
        if beat_strength > SUPER_PUNCH_THRESHOLD:
            current_damping_rhythm = DAMPING_FACTOR_PUNCH
        else:
            current_damping_rhythm = DAMPING_FACTOR_ACCUMULATION
        
        # ------------------------------------------------------------------
        # --- W_COARSE (Layers 0-3): STRUCTURE/POSE DRIVERS ---
        # ------------------------------------------------------------------
        
        # 1. CORE RHYTHM MOVEMENT: (Piano attacks)
        rhythmic_movement = np.full(w_dim, rhythm_magnitude / current_damping_rhythm) * update_dir * JITTER_VECTOR_COARSE
        w_coarse += rhythmic_movement
        
        # 2. MELODIC/SUSTAINED MOVEMENT: (General mood/brightness)
        melodic_movement = np.full(w_dim, melodic_magnitude / DAMPING_FACTOR_MELODY) * update_dir * JITTER_VECTOR_COARSE
        w_coarse += melodic_movement
        
        # 3. TARGETED CHROMA NUDGE: (Small, direct color bias for key changes)
        chroma_delta_power = np.sum(chroma_delta[:, f]) * pitch_sensitivity / 50 
        w_coarse[:12] += chroma_delta[:12, f] * chroma_delta_power * 0.2
        
        # ------------------------------------------------------------------
        # --- W_MIDDLE (Layers 4-7): STYLE/COLOR DRIVERS ---
        # ------------------------------------------------------------------
        
        # 4. SAXOPHONE DRIVER: (Sustained melody color shift)
        sax_movement = np.full(w_dim, sax_magnitude / DAMPING_FACTOR_SAX) * update_dir * JITTER_VECTOR_MIDDLE # Use middle jitter
        w_middle += sax_movement 
        
        # 5. CHROMA CONTENT DRIVER: (Absolute Pitch Power for rapid color swap)
        chroma_content_movement = np.full(w_dim, chroma_content_magnitude / DAMPING_FACTOR_CHROMA_CONTENT) * update_dir * JITTER_VECTOR_MIDDLE # Use middle jitter
        w_middle += chroma_content_movement
        
        # ------------------------------------------------------------------
        # --- FINAL CLIPPING AND W+ CONSTRUCTION ---
        # ------------------------------------------------------------------
        
        # Apply Non-Musical Drift and clip accumulation vectors
        w_coarse = np.clip(w_coarse + base_drift_coarse, -truncation, truncation) # Use coarse drift
        w_middle = np.clip(w_middle + base_drift_middle, -truncation, truncation) # Use middle drift
        
        # 6. FINE (Layers 8+): Detail/Noise 
        # CRITICAL FIX: Base w_fine on w_coarse (stable pose) to reduce noise.
        w_fine_base = w_coarse.copy() 
        w_fine_movement = np.random.randn(w_dim) * flicker_magnitude * 0.5 
        w_fine = np.clip(w_fine_base + w_fine_movement, -truncation, truncation)
        
        # --- W+ Vector Concatenation ---
        coarse_ws = np.repeat(w_coarse[np.newaxis, :], 4, axis=0) 
        middle_ws = np.repeat(w_middle[np.newaxis, :], 4, axis=0) 
        fine_ws = np.repeat(w_fine[np.newaxis, :], num_ws - 8, axis=0)
        
        w_plus = np.concatenate([coarse_ws, middle_ws, fine_ws], axis=0)
        
        if w_plus.shape[0] != num_ws:
            w_plus = np.repeat(w_middle[np.newaxis, :], num_ws, axis=0)
            
        ws.append(w_plus)
        
        # Use coarse W for direction reversal check
        update_dir = new_update_dir(w_coarse, update_dir, tempo_sensitivity, truncation)
    
    np.save('saved_vectors/w_vectors.npy', np.array(ws))
    return np.array(ws)