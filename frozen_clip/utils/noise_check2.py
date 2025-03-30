import os
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment

def is_noise_only(audio_path, silence_threshold=-35.0):
    audio = AudioSegment.from_file(audio_path)
    loudness = np.array(audio.dBFS)

    if loudness.mean() < silence_threshold:
        return True
    else:
        return False

noise_file = []
audio_dir = "../spotify-small/clip_spotify-small/test"
files = sorted(os.listdir(audio_dir))
for i in tqdm(range(len(files))):
  if (files[i].split(".")[-1] == "mp3"):
    audio_path = os.path.join(audio_dir, files[i])
    is_noise = is_noise_only(audio_path)
    if (is_noise):
      noise_file.append(files[i])

print(noise_file)