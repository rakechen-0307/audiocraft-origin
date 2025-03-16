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
    # Generate padded number
    # num = files[i].split('.')[0]

    try:
        # Construct full source and destination paths
        source_path = os.path.join(data_dir, files[i])
        # dest_path = os.path.join(data_dir, f"{num}.mp3")

        # Use subprocess instead of os.system for better error handling
        # Copy the file
        # subprocess.run(["mv", source_path, dest_path], check=True)

        # Run demucs
        subprocess.run(["allin1", source_path], check=True)

    except Exception as e:
        print(f"Failed to process file {files[i]}: {e}")
        failed.append(files[i])

# Write failed files to JSON
if failed:
    failed_dict = {"data": failed}
    with open("failed.json", "w") as outfile:
        json.dump(failed_dict, outfile, indent=4)