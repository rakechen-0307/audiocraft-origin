import os
from tqdm import tqdm
from pydub import AudioSegment
import numpy as np

def is_noise_only(audio_path, silence_threshold=-35.0):
    audio = AudioSegment.from_file(audio_path)
    loudness = np.array(audio.dBFS)

    if loudness.mean() < silence_threshold:
        return True
    else:
        return False

noise_file = []
audio_dir = "../clipclap/valid/audios"
video_dir = "../clipclap/valid/videos"
dirs = sorted(os.listdir(audio_dir))
for i in tqdm(range(len(dirs))):
    files = sorted(os.listdir(os.path.join(audio_dir, dirs[i])))
    for j in range(len(files)):
        audio_path = os.path.join(audio_dir, dirs[i], files[j])
        video_path = os.path.join(video_dir, dirs[i], f"{files[j].split('.')[0]}.mp4")
        is_noise = is_noise_only(audio_path)
        if (is_noise):
            os.system("rm {}".format(audio_path))
            os.system("rm {}".format(video_path))
            print(f"remove {dirs[i]}/{files[j]}")