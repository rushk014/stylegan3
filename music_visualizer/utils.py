import librosa
import numpy as np
import os
from typing import List, Union 

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
                       walk_rate_sensitivity, 
                       pitch_sensitivity, 
                       melodic_sensitivity, 
                       sax_sensitivity, 
                       chroma_content_sensitivity, 
                       truncation, 
                       frame_length, 
                       noise_floor_magnitude, 
                       jump_threshold, 
                       num_ws, w_dim, jitter, preload=False, 
                       w_avg_vec: Union[np.ndarray, None] = None):
    if preload:
        return np.load('saved_vectors/w_vectors.npy')

    # --- Feature Extraction ---
    spec, spec_mean, grad_mean = generate_power(y, sr, frame_length)
    chroma = generate_chroma(y, sr, frame_length)
    
    # Mid-Frequency and High-Frequency Energy (CRITICAL PLACEMENT)
    cqt_freqs = librosa.cqt_frequencies(n_bins=spec.shape[0], fmin=librosa.note_to_hz('C1'))
    
    MF_START_FREQ = 300; MF_END_FREQ = 3500      
    mf_indices = np.where((cqt_freqs >= MF_START_FREQ) & (cqt_freqs <= MF_END_FREQ))[0]
    raw_mf_energy = np.sum(spec[mf_indices, :], axis=0)
    mid_frequency_magnitude = (np.log1p(raw_mf_energy) - np.min(np.log1p(raw_mf_energy))) / (np.ptp(np.log1p(raw_mf_energy)) if np.ptp(np.log1p(raw_mf_energy)) > 0 else 1)

    HF_START_FREQ = 5000; HF_END_FREQ = 10000
    hf_indices = np.where((cqt_freqs >= HF_START_FREQ) & (cqt_freqs <= HF_END_FREQ))[0]
    raw_hf_energy = np.sum(spec[hf_indices, :], axis=0)
    hf_energy_magnitude = (np.log1p(raw_hf_energy) - np.min(np.log1p(raw_hf_energy))) / (np.ptp(np.log1p(raw_hf_energy)) if np.ptp(np.log1p(raw_hf_energy)) > 0 else 1)
    
    # 2. Define Stable Anchor Points (W_ANCHORS)
    if w_avg_vec is None:
        w_avg_vec = np.zeros(w_dim, dtype=np.float32)
        print("Warning: W_AVG not supplied. Defaulting to zero vector, which may cause artifacts.")

    PERTURBATION_MAGNITUDE = 1.0
    NUM_ANCHORS = 6

    W_ANCHORS: List[np.ndarray] = []
    for _ in range(NUM_ANCHORS):
        random_noise = np.random.randn(w_dim).astype(np.float32)
        w_anchor = w_avg_vec + (random_noise * PERTURBATION_MAGNITUDE)
        W_ANCHORS.append(w_anchor)
    
    # --- ANCHOR SAVING BLOCK REMAINS ---
    anchor_ws_plus: List[np.ndarray] = []
    for anchor in W_ANCHORS:
        anchor_ws_plus.append(np.repeat(anchor[np.newaxis, :], num_ws, axis=0))

    os.makedirs('saved_vectors', exist_ok=True)
    np.save('saved_vectors/anchor_ws_check.npy', np.array(anchor_ws_plus, dtype=np.float32))
    # ----------------------------------------------------

    # --- MOVEMENT SCALING COEFFICIENTS ---
    BASE_WALK_RATE = 0.005 # Base walk speed
    RHYTHM_ACCELERATION_SCALE = walk_rate_sensitivity 
    CHROMA_ACCELERATION_SCALE = 0.05 
    # JUMP_THRESHOLD is the external parameter (jump_threshold)
    # -------------------------------------------------------------
    
    lerp_progress = 0.0 
    
    ws = []
    
    for f in range(len(spec_mean)):
        # --- 1. Feature Magnitudes ---
        rhythm_magnitude = grad_mean[f] * RHYTHM_ACCELERATION_SCALE # Used for both fractional shift and jump check
        sax_magnitude = mid_frequency_magnitude[f] * sax_sensitivity 
        chroma_content_magnitude = np.sum(chroma[:, f]) * chroma_content_sensitivity 
        
        # --- 2. JUMP LOGIC (Non-Adjacent Anchor Hops) ---
        if rhythm_magnitude > jump_threshold:
            # Jump 1 (adjacent) to 3 (non-adjacent) anchors ahead
            jump_steps = np.random.randint(1, 4) 
            lerp_progress += jump_steps
        
        # --- 3. Circular Path Update (Base Walk and Fractional Acceleration) ---
        progress_shift = BASE_WALK_RATE + (rhythm_magnitude * RHYTHM_ACCELERATION_SCALE) 
        progress_shift += (sax_magnitude * CHROMA_ACCELERATION_SCALE)
        
        lerp_progress += progress_shift
        
        # --- 4. Determine Current Anchor Points and Fraction ---
        start_index = int(np.floor(lerp_progress)) % NUM_ANCHORS
        target_index = (start_index + 1) % NUM_ANCHORS
        lerp_fraction = lerp_progress - np.floor(lerp_progress)

        W_START = W_ANCHORS[start_index]
        W_TARGET = W_ANCHORS[target_index]
        
        # --- 5. LERP Interpolation ---
        w_coarse_current = W_START * (1 - lerp_fraction) + W_TARGET * lerp_fraction
        
        color_shift = (chroma_content_magnitude * 0.1) 
        w_middle_fraction = np.clip(lerp_fraction + color_shift, 0, 1)
        w_middle_current = W_START * (1 - w_middle_fraction) + W_TARGET * w_middle_fraction
        
        # --- 6. PSYCHEDELIC NOISE INJECTION (BIGGAN STYLE) ---
        
        noise_injection_magnitude = noise_floor_magnitude + (spec_mean[f] * 0.5) 
        
        w_fine_base = w_coarse_current.copy() 
        
        w_fine_movement = np.random.randn(w_dim) * noise_injection_magnitude 
        w_fine = np.clip(w_fine_base + w_fine_movement, -truncation, truncation)
        
        # --- W+ Vector Concatenation ---
        coarse_ws = np.repeat(w_coarse_current[np.newaxis, :], 4, axis=0) 
        middle_ws = np.repeat(w_middle_current[np.newaxis, :], 4, axis=0) 
        fine_ws = np.repeat(w_fine[np.newaxis, :], num_ws - 8, axis=0)
        
        w_plus = np.concatenate([coarse_ws, middle_ws, fine_ws], axis=0)
        
        if w_plus.shape[0] != num_ws:
            w_plus = np.repeat(w_middle_current[np.newaxis, :], num_ws, axis=0)
            
        ws.append(w_plus)
    
    np.save('saved_vectors/w_vectors.npy', np.array(ws, dtype=np.float32)) 
    return np.array(ws, dtype=np.float32)