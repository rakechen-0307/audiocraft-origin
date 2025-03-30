import os
import torchaudio
from audiocraft.models.musicgen import MusicGen, MusicGenCLAP
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio

seg_len = 10

model = MusicGenCLAP.get_pretrained('checkpoints/clapemb(spotify-small-new)')
model.set_generation_params(duration=10, cfg_coef=3.0)

sample_dir = "./samples/audios"
sample_files = sorted(os.listdir(sample_dir))
wav_files = []
file_names = []
for i in range(len(sample_files)):
    wav_files.append(sample_dir + "/" + sample_files[i])
    file_names.append(sample_files[i])

for i in range(len(wav_files)):
    file = wav_files[i]
    wav, sr = torchaudio.load(file)
    wav = convert_audio(wav, sr, model.sample_rate, model.audio_channels)
    audio = model.generate_with_clap_embed(wav.unsqueeze(0))

    audio_write(f"{file_names[i].split('.')[0]}", audio.cpu().squeeze(0), model.sample_rate, strategy="loudness", loudness_compressor=True)