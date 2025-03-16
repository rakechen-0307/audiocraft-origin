import os
from tqdm import tqdm

audio_dir = "../audios"
output_dir = "../data/audios"

dirs = sorted(os.listdir(audio_dir))
for i in tqdm(range(len(dirs))):
    os.makedirs(os.path.join(output_dir, dirs[i]))
    files = sorted(os.listdir(os.path.join(audio_dir, dirs[i])))
    