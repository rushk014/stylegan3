import torch
import numpy as np
import dnnlib
import legacy
import os
from PIL import Image

# 1. Setup Device and Model
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
network_pkl = 'models/stylegan3-t-ffhqu-256x256.pkl'
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema']
model = G.to(device)
synthesis_net = model.synthesis

# 2. Load Anchor W+ Vectors (saved from the previous step)
anchor_w_plus = np.load('saved_vectors/anchor_ws_check.npy').astype(np.float32)  # Shape: (num_anchors, num_ws, w_dim)
w_batch = torch.from_numpy(anchor_w_plus).to(device)  # Convert to torch tensor

# 3. Generate Images
with torch.no_grad():
    output = synthesis_net(w_batch, noise_mode='const')
    
# Convert to image format (uint8, 0-255)
output_images = (output * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

# 4. Save Images
output_dir = 'output/anchor_check'
os.makedirs(output_dir, exist_ok=True)

for i, img_array in enumerate(output_images):
    img = Image.fromarray(img_array)
    img.save(f"{output_dir}/anchor_{i+1}.png")
    
print(f"\nSuccessfully generated and saved {len(output_images)} anchor images to {output_dir}")