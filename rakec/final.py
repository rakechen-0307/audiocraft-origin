import os
import torchaudio
from audiocraft.models.musicgen import MusicGenCLAP
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio

model = MusicGenCLAP.get_pretrained('checkpoints/clapemb(spotify-small)')
model.set_generation_params(duration=30, cfg_coef=3.0)

