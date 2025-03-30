import os
import json
import subprocess
from tqdm import tqdm
import shlex

def path_remake(path):
    # Use shlex.quote to properly escape paths
    return shlex.quote(path)

data_dir = "../audios"
files = sorted(os.listdir(data_dir))
failed = []

for i in tqdm(range(len(files))):
    try:
        source_path = os.path.join(data_dir, files[i])
        subprocess.run(["demucs", "--two-stems=vocals", "--mp3", "--mp3-preset", "2", source_path], check=True)

    except Exception as e:
        print(f"Failed to process file {files[i]}: {e}")
        failed.append(source_path)

# Write failed files to JSON
if failed:
    failed_dict = {"data": failed}
    with open("failed.json", "w") as outfile:
        json.dump(failed_dict, outfile, indent=4)