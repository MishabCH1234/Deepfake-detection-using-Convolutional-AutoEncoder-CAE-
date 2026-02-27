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


### Model Example

---python
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
            # Correct the input features for the first linear layer based on the 128x128 input size
            nn.Linear(1024 * 8 * 8, 1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, 2)  # Two output classes: real or fake
        )

        # Decoder - Note: The decoder dimensions are also likely incorrect if designed for a 16x16 spatial size after convolution.
        # If the decoder is intended to reconstruct the original 256x256 image, its architecture and the first linear layer
        # might need to be adjusted as well based on the output of the encoder's classification layer (which has 2 features).
        # However, the current error is in the encoder's forward pass during `summary`.
        self.decoder_fc = nn.Sequential(
            nn.Linear(2, 1024), # This takes the 2 output features from the encoder
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, 1024 * 16 * 16), # This is likely incorrect for reconstructing from 2 features to 256x256.
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (1024, 16, 16)), # This expects the output of decoder_fc to be reshapeable to (1024, 16, 16)
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Softmax()  # To get pixel values in the range [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, encoding):
        x = self.decoder_fc(encoding)
        x = self.decoder_conv(x)
        return x
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

