import os
import torchaudio
from audiocraft.models.musicgen import MusicGen, MusicGenCLAP
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio

model = MusicGenCLAP.get_pretrained('checkpoints/clapemb(spotify-small)')
model.set_generation_params(duration=10, cfg_coef=3.0)

sample_dir = "/work/u2614323/code/audiocraft-origin/samples/audios"
sample_files = sorted(os.listdir(sample_dir))
wav_files = []
file_names = []
for i in range(len(sample_files)):
    wav_files.append(sample_dir + "/" + sample_files[i])
    file_names.append(sample_files[i])

wavs = []
for file in wav_files:
    wav, sr = torchaudio.load(file)
    wav = convert_audio(wav, sr, model.sample_rate, model.audio_channels)
    wavs.append(wav)

wav = model.generate_with_clap_embed(wavs)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f"{file_names[idx].split('.')[0]}", one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)