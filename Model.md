# Classifier & Image Reconstructor Model

This repository contains a **PyTorch implementation** of a Convolutional Neural Network (CNN) designed for dual-purpose tasks:

- Image Classification  
- Latent-Space-Based Image Reconstruction  

---

## 🧠 Architecture Overview

The model follows an **Encoder–Decoder bottleneck architecture**.

### 🔹 Encoder

- Four `Conv2d` layers  
- `LeakyReLU` activations  
- Progressive spatial reduction  
- Increasing feature depth  
- Fully connected bottleneck layer  
- Outputs **2-class classification** (e.g., Real vs. Fake)

---

### 🔹 Decoder

- Symmetrical expansion network  
- `ConvTranspose2d` layers  
- Takes **2 latent features** as input  
- Uses `Unflatten` strategy  
- Learned upsampling  
- Produces high-resolution RGB image  

---

## ⚙️ Key Technical Specifications

| Component | Specification |
|------------|--------------|
| **Input Size** | `1 × 128 × 128` (Grayscale) |
| **Latent Bottleneck** | 2 Features |
| **Reconstruction Output** | `3 × 256 × 256` (RGB) |
| **Activation (Hidden Layers)** | `LeakyReLU (α = 0.1)` |
| **Output Activation** | `Softmax` |

---

## 🚀 Usage Example

You can use the model to either:

- Classify an image  
- Generate a reconstruction from a class vector  

### Python Example

```python
import torch
from model import Classifier  # Assuming your class is in model.py

# 1️⃣ Initialize Model
model = Classifier()
model.eval()

# 2️⃣ Classification Forward Pass
# Input shape: (Batch, Channels, Height, Width)
dummy_input = torch.randn(1, 1, 128, 128)
class_logits = model(dummy_input)

print(f"Classification Output Shape: {class_logits.shape}")  
# Expected: [1, 2]

# 3️⃣ Reconstruction / Decoding Pass
reconstructed_img = model.decode(class_logits)

print(f"Reconstructed Image Shape: {reconstructed_img.shape}")  
# Expected: [1, 3, 256, 256]
