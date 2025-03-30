import os
from tqdm import tqdm

source_dir = "../clip_spotify-small/test"
target_dir = "../clap_spotify-small/valid"
files = sorted(os.listdir(source_dir))
for i in tqdm(range(len(files))):
    if (files[i].split('.')[-1] != 'mp4'):
        os.system("cp {source} {target}".format(
            source=os.path.join(source_dir, files[i]),
            target=target_dir
        ))