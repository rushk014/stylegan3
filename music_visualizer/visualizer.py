import os
import argparse
import torch
from tqdm import tqdm
import librosa
import dnnlib
import legacy
import json
from moviepy import AudioFileClip, ImageSequenceClip
import time
import numpy as np 

# Update the utility function import
from .utils import generate_w_vectors, get_frame_lim, W_DIM

def setup_parser():
    script_dir = os.path.dirname(__file__)
    defaults_path = os.path.join(script_dir, 'defaults.json')
    try:
        with open(defaults_path, 'r') as f:
            defaults = json.load(f)
    except FileNotFoundError:
        print("Warning: defaults.json not found. Using hardcoded values.")
        defaults = {}
    
    parser = argparse.ArgumentParser(description="Audio visualizer using StyleGAN3 and audio analysis", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--song", required=True, default=defaults.get("song", "input/romantic.mp3"), help="path to input audio file")
    parser.add_argument("-d", "--duration", type=int, help="output video duration, defaults to entire song length")
    
    # --- EXTERNALIZED SENSITIVITY ARGUMENTS ---
    parser.add_argument("-ps", "--pitch_sensitivity", type=int, default=defaults.get("pitch_sensitivity", 150), metavar="[50-400]", help="controls the sensitivity of the CHROMA NUDGE to changes in pitch")
    
    # Renamed/Re-purposed Argument: Now controls rhythmic acceleration
    parser.add_argument("-wrs", "--walk_rate_sensitivity", type=float, default=defaults.get("walk_rate_sensitivity", 0.15), metavar="[0.05-0.8]", help="controls how strongly RHYTHMIC HITS accelerate the morph speed (BASE_WALK_RATE in utils.py)")
    
    parser.add_argument("-ms", "--melodic_sensitivity", type=float, default=defaults.get("melodic_sensitivity", 0.5), metavar="[0.05-1.0]", help="controls the sensitivity of the general MELODIC/LOUDNESS driver")
    parser.add_argument("-sas", "--sax_sensitivity", type=float, default=defaults.get("sax_sensitivity", 0.2), metavar="[0.05-1.0]", help="controls the sensitivity of the MID-FREQUENCY (Saxophone) driver")
    parser.add_argument("-ccs", "--chroma_content_sensitivity", type=float, default=defaults.get("chroma_content_sensitivity", 0.1), metavar="[0.01-0.5]", help="controls the sensitivity of the CHROMA CONTENT (note-swap) driver")
    # ------------------------------------------

    parser.add_argument("-j", "--jitter", type=float, default=defaults.get("jitter", 0.15), metavar="[0-1]", help="controls jitter of the latent vector to reduce repitition")
    parser.add_argument("-fl", "--frame_length", type=int, default=defaults.get("frame_length", 512), metavar="i*2^6", help="number of audio frames to video frames in the output")
    parser.add_argument("-t", "--truncation", type=float, default=defaults.get("truncation", 0.75), metavar="[0.1-1]", help="StyleGAN truncation parameter controls complexity of structure within frames")
    parser.add_argument("-bs", "--batch_size", type=int, default=defaults.get("batch_size", 8), help="Batch size for GPU generation")
    parser.add_argument("-o", "--output_file", default="", help="name of output file stored in output/, defaults to [--song] path base_name")
    parser.add_argument("-sr", "--sample_rate", type=int, default=defaults.get("sample_rate", 22050), metavar="[11025, 22050, 44100]", help="Sample rate for audio analysis (affects FPS and fidelity)")
    
    # --- JUMP THRESHOLD (Vestigial but Retained) ---
    parser.add_argument("-jt", "--jump_threshold", type=float, default=defaults.get("jump_threshold", 0.4), metavar="[0.1-1.0]", help="normalized rhythm intensity needed to trigger a non-adjacent jump")
    # ---------------------------------------------

    # --- NOISE FLOOR MAGNITUDE ---
    parser.add_argument("-nfm", "--noise_floor_magnitude", type=float, default=defaults.get("noise_floor_magnitude", 0.1), metavar="[0.05-0.5]", help="base magnitude of random noise injected into fine layers for psychedelic texture")
    # -----------------------------

    parser.add_argument("--use_last_vectors", action="store_true", default=False, help="set flag to use previous saved class/noise vectors")
    return parser

def visualize(w_vectors, model, batch_size, frame_lim):
    """Generates frames from StyleGAN3 model using W+ vectors."""
    frames = []
    synthesis_net = model.synthesis
    
    for i in tqdm(range(frame_lim)):
        if (i+1) * batch_size > len(w_vectors):
            torch.cuda.empty_cache()
            break
        
        # W+ vector batch shape: [batch_size, num_ws, W_DIM]
        # Ensure w_batch is on the same device and is float32
        w_batch = w_vectors[i*batch_size:(i+1)*batch_size].float()
        
        with torch.no_grad():
            output = synthesis_net(w_batch, noise_mode='const')
        
        # StyleGAN output is float32 in range [-1, 1]. Convert to uint8 [0, 255].
        output_images = (output * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        
        frames.extend(list(output_images))
        torch.cuda.empty_cache()
    return frames


if __name__ == '__main__':
    start_time = time.time()
    parser = setup_parser()
    args = parser.parse_args()
    
    # --- PARAMS FROM ARGS ---
    song = args.song
    frame_length = args.frame_length
    pitch_sensitivity = args.pitch_sensitivity
    
    # Renamed/Re-purposed Argument
    walk_rate_sensitivity = args.walk_rate_sensitivity
    
    melodic_sensitivity = args.melodic_sensitivity
    sax_sensitivity = args.sax_sensitivity
    chroma_content_sensitivity = args.chroma_content_sensitivity
    jitter = args.jitter
    batch_size = args.batch_size
    use_last_vectors = args.use_last_vectors
    truncation = args.truncation
    sr_to_use = args.sample_rate
    
    # New Arguments
    noise_floor_magnitude = args.noise_floor_magnitude
    jump_threshold = args.jump_threshold
    # -------------------------

    # ensure necessary directories exist
    os.makedirs('saved_vectors', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # --- UNIQUE FILENAME LOGIC ---
    if args.output_file:
        outname = args.output_file
    else:
        timestamp = time.strftime("_%Y%m%d_%H%M%S")
        song_base_name = os.path.basename(args.song).split('.')[0]
        outname = 'output/' + song_base_name + timestamp + '.mp4'
    
    print(f"Saving output to: {outname}\n")

    print('Reading audio\n')
    
    load_duration = args.duration if args.duration else None
    y, sr = librosa.load(song, sr=sr_to_use, duration=load_duration)

    audio_duration_seconds = len(y)/sr
    frame_lim = get_frame_lim(audio_duration_seconds, frame_length, batch_size)
    
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {torch.cuda.get_device_name(0)} (CUDA)\n")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print('Using device: MPS (Apple Silicon)\n')
    else:
        device = torch.device('cpu')
        print('Using device: CPU\n')

    # Load pre-trained model
    print('\nLoading StyleGAN3 \n')
    network_pkl = 'models/stylegan3-r-afhqv2-512x512.pkl'
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        data = legacy.load_network_pkl(f)
    
    G = data['G_ema']
    # CRITICAL FIX 1: Extract the W_AVG vector from the mapping network
    W_AVG = G.mapping.w_avg.cpu().numpy()

    model = G.to(device)
    NUM_WS = model.num_ws

    print('Generating W+ vectors \n')
    
    # --- PASS ALL ARGUMENTS TO UTILS ---
    w_vectors = generate_w_vectors(
        y, sr, 
        walk_rate_sensitivity, 
        pitch_sensitivity, 
        melodic_sensitivity, 
        sax_sensitivity, 
        chroma_content_sensitivity, 
        truncation, 
        frame_length, 
        noise_floor_magnitude, 
        jump_threshold, 
        NUM_WS, 
        W_DIM, 
        jitter, 
        use_last_vectors,
        W_AVG
    )
    
    w_vectors = torch.Tensor(w_vectors) 

    # Generate frames in batches of batch_size
    print('Generating frames \n')
    w_vectors = w_vectors.to(device)
    
    frames = visualize(w_vectors, model, batch_size, frame_lim)
    
    true_frame_count = len(frames)
    true_video_duration = true_frame_count * frame_length / sr_to_use
    
    # Save video  
    aud = AudioFileClip(song, fps=44100) 
    aud = aud.subclipped(0, true_video_duration)
    
    clip = ImageSequenceClip(frames, fps=sr_to_use/frame_length)
    clip = clip.with_duration(true_video_duration)

    clip = clip.with_audio(aud)
    clip.write_videofile(outname, audio_codec='aac')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")