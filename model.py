#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch
torch.cuda.is_available()


# In[1]:


import os
import json
import torch
import torchaudio
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch.optim as optim
from tqdm import tqdm


# In[2]:


import torch.nn.functional as F

class MusicBenchDataset(Dataset):
    def __init__(self, json_path, audio_dir, max_text_length=77, sample_rate=22050, n_mels=128, max_frames=860):  
        # max_frames corresponds to 10 seconds of mel spectrogram at your chosen hop length
        self.data = self.load_data(json_path)
        self.audio_dir = audio_dir
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.max_text_length = max_text_length
        self.sample_rate = sample_rate
        self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.audio_transform = torchaudio.transforms.Resample(orig_freq=44100, new_freq=sample_rate)
        self.max_frames = max_frames  # Maximum number of frames to pad/truncate to

    def load_data(self, json_path):
        data = []
        with open(json_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve and preprocess data at the specified index."""
        entry = self.data[idx]
        
        # Text processing
        main_caption = entry["main_caption"]
        text_encoding = self.tokenizer(
            main_caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        # Audio processing
        audio_path = os.path.join(self.audio_dir, entry["location"])
        waveform, _ = torchaudio.load(audio_path)
        waveform = self.audio_transform(waveform)
        mel_spec = self.mel_spectrogram(waveform).squeeze(0)  # Remove batch dimension if exists

        # Pad or truncate mel spectrogram
        mel_spec = self.pad_or_truncate(mel_spec)

        return {
            "text": text_encoding["input_ids"].squeeze(0),  # Text tokens
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "mel_spec": mel_spec,  # Fixed-size mel spectrogram
        }

    def pad_or_truncate(self, mel_spec):
        """Pad or truncate mel spectrogram to fixed size."""
        if mel_spec.size(1) < self.max_frames:  # Pad if shorter
            pad_size = self.max_frames - mel_spec.size(1)
            mel_spec = F.pad(mel_spec, (0, pad_size), mode="constant", value=0)
        else:  # Truncate if longer
            mel_spec = mel_spec[:, :self.max_frames]
        return mel_spec


# In[3]:


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, input_ids, attention_mask):
        return self.clip_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


# In[4]:


class TransformerAlignment(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=4, num_heads=8):
        super(TransformerAlignment, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads), num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))  # Aggregate sequence into a single vector


# In[5]:


class UNetGenerator(nn.Module):
    def __init__(self, input_dim, condition_dim, num_channels=64):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_channels * 2, num_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.condition = nn.Linear(condition_dim, num_channels * 2)

    def forward(self, x, condition):
        condition = self.condition(condition).unsqueeze(-1).unsqueeze(-1)
        encoded = self.encoder(x)
        conditioned = encoded + condition
        return self.decoder(conditioned)


# In[6]:


class TextToMusicModel(nn.Module):
    def __init__(self, text_dim=512, latent_dim=256):
        super(TextToMusicModel, self).__init__()
        self.text_encoder = TextEncoder()
        self.latent_alignment = TransformerAlignment(input_dim=text_dim, output_dim=latent_dim)
        self.generator = UNetGenerator(input_dim=1, condition_dim=latent_dim)

    def forward(self, input_ids, attention_mask, noise):
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        latent_condition = self.latent_alignment(text_embeddings)
        generated_spec = self.generator(noise, latent_condition)
        return generated_spec


# In[7]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextToMusicModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# In[8]:


def train_model(model, dataloader, epochs, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(dataloader):
            mel_spec = batch["mel_spec"].to(device)  # Shape: [batch_size, n_mels, max_frames]
            input_ids = batch["text"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Add a channel dimension for mel_spec (single channel)
            mel_spec = mel_spec.unsqueeze(1)  # Shape: [batch_size, 1, n_mels, max_frames]
            
            # Noise for generator
            noise = torch.randn_like(mel_spec, device=device)
            
            optimizer.zero_grad()
            output = model(input_ids, attention_mask, noise)
            loss = criterion(output, mel_spec)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    return model


# In[9]:


dataset = MusicBenchDataset(json_path="datashare/train.json", audio_dir="datashare")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# In[10]:


model = train_model(model, dataloader, epochs=10)


# In[ ]:


import torch

def save_model(model, path="trained_model.pth"):
    """
    Save the trained model to a file for later inference.
    
    Args:
        model: The trained PyTorch model.
        path: File path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Example Usage
save_model(model, "music_gen_model.pth")


# In[ ]:


def load_model(model_class, path="trained_model.pth", device='cuda'):
    """
    Load a saved model from file.
    
    Args:
        model_class: The class of the model (uninitialized).
        path: File path of the saved model weights.
        device: Device to load the model on.
    
    Returns:
        model: The loaded PyTorch model.
    """
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print("Model loaded successfully")
    return model

def generate_mel_spectrogram(model, prompt, tokenizer, device='cuda'):
    """
    Generate a mel spectrogram from a text prompt.
    
    Args:
        model: The loaded model.
        prompt: Text prompt for music generation.
        tokenizer: CLIP tokenizer.
        device: Device to run inference.
    
    Returns:
        mel_spec: The generated mel spectrogram.
    """
    tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    noise = torch.randn(1, 1, 128, 860).to(device)  # Example shape for mel spectrogram
    
    with torch.no_grad():
        mel_spec = model(input_ids, attention_mask, noise)
    
    return mel_spec.squeeze(0).cpu()


# In[ ]:


from transformers import CLIPTokenizer

def generate_mel_spectrogram(model, prompt, device='cuda'):
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    noise = torch.randn((1, 1, 128, 860), device=device)  # Shape: [batch_size, channels, n_mels, max_frames]
    
    with torch.no_grad():
        generated_spec = model(input_ids, attention_mask, noise)
    return generated_spec.squeeze(0).cpu().numpy()  # Remove batch and channel dimensions

prompt = "This song is recorded in low quality with the sound of an electric guitar tuning. The recording is gated, which means the guitar strings produce a lot of noise whenever they are plucked, but in between them, there is complete silence. The chord sequence of the song is D and G with a tempo of 88.0 bpm and a beat of 4. It is played in D minor key."
generated_mel = generate_mel_spectrogram(model, prompt)

print("Generated mel spectrogram shape:", generated_mel.shape)


# In[ ]:


generated_mel = generated_mel.reshape(128,860)
generated_mel = np.round(generated_mel,1)
generated_mel


# In[ ]:


import numpy as np
import torch
import torchaudio
from torchaudio.transforms import InverseMelScale, GriffinLim

def mel_to_audio(mel_spectrogram, sample_rate=22050, n_fft=2048, n_mels=128, hop_length=512, win_length=2048):
    """
    Convert a mel spectrogram to an audio waveform using the Griffin-Lim algorithm.
    
    Args:
        mel_spectrogram (numpy.ndarray or torch.Tensor): Mel spectrogram array of shape [n_mels, time_frames].
        sample_rate (int): Target sample rate for the audio.
        n_fft (int): FFT size.
        n_mels (int): Number of mel bins.
        hop_length (int): Hop length for STFT.
        win_length (int): Window length for STFT.
    
    Returns:
        waveform (numpy.ndarray): Reconstructed waveform.
    """
    # Ensure the mel spectrogram is a Torch Tensor
    if isinstance(mel_spectrogram, np.ndarray):
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)
    
    # Inverse Mel Scale to get the linear spectrogram
    inverse_mel = InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate).to(mel_spectrogram.device)
    linear_spectrogram = inverse_mel(mel_spectrogram)
    
    # Apply the Griffin-Lim algorithm to reconstruct the waveform
    griffin_lim = GriffinLim(n_fft=n_fft, hop_length=hop_length, win_length=win_length).to(mel_spectrogram.device)
    waveform = griffin_lim(linear_spectrogram)
    
    return waveform.cpu().numpy()

# Example usage
# if __name__ == "__main__":
#     # Simulate a generated mel spectrogram (replace with your actual generated mel spectrogram)
#     # generated_mel =   # Example mel spectrogram: [n_mels, time_frames]

#     # Convert mel spectrogram to audio
#     reconstructed_audio = mel_to_audio(
#         generated_mel*5,
#         sample_rate=22050,
#         n_fft=2048,
#         n_mels=128,
#         hop_length=512,
#         win_length=2048
#     )
    
#     # Save the reconstructed audio to a file
#     torchaudio.save("reconstructed_audio.wav", torch.tensor(reconstructed_audio).unsqueeze(0), sample_rate=22050)
#     print("Reconstructed audio saved to 'reconstructed_audio.wav'")


# In[13]:


import json
import torch
import torchaudio
from torchaudio.transforms import InverseMelScale, GriffinLim
from transformers import CLIPTokenizer
import numpy as np

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth", map_location=device)  # Load the trained model
model.eval()

# Define the mel-to-audio function

# Load captions from testA.json
def load_captions(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [data["main_caption"]]

# Generate mel spectrograms from captions
def generate_mel_from_captions(captions, model, tokenizer, device):
    mel_spectrograms = []
    for caption in captions:
        # Tokenize caption
        encoded_input = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        
        # Generate noise for the generator
        noise = torch.randn(1, 1, 128, 860, device=device)  # Match dimensions with your training
        
        # Generate mel spectrogram
        with torch.no_grad():
            generated_mel = model(
                encoded_input["input_ids"],
                encoded_input["attention_mask"],
                noise,
            )
        mel_spectrograms.append(generated_mel.squeeze(0).cpu())
    return mel_spectrograms

# Main process
if __name__ == "__main__":
    # File paths
    json_path = "data/datashare/testA.json"  # Replace with the path to testA.json
    output_dir = "generated_audio"  # Directory to save generated audio files
    os.makedirs(output_dir, exist_ok=True)

    # Load captions
    captions = load_captions(json_path)
    print("Captions loaded:", captions)

    # Initialize tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Generate mel spectrograms
    mel_spectrograms = generate_mel_from_captions(captions, model, tokenizer, device)
    
    # Convert mel spectrograms to audio and save
    for i, mel_spec in enumerate(mel_spectrograms):
        reconstructed_audio = mel_to_audio(
            mel_spec.numpy(),
            sample_rate=22050,
            n_fft=2048,
            n_mels=128,
            hop_length=512,
            win_length=2048,
        )
        
        # Save the audio
        audio_path = os.path.join(output_dir, f"generated_audio_{i}.wav")
        torchaudio.save(audio_path, torch.tensor(reconstructed_audio).unsqueeze(0), sample_rate=22050)
        print(f"Generated audio saved to {audio_path}")

