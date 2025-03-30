import os
import json
import subprocess
from tqdm import tqdm
import shlex

def path_remake(path):
    # Use shlex.quote to properly escape paths
    return shlex.quote(path)

failed_file = "./failed.json"
failed = []
with open(failed_file, 'r') as file:
    data = json.load(file)['data']
for i in tqdm(range(len(data))):
    try:
        source_path = data[i]
        subprocess.run(["demucs", "--two-stems=vocals", "--mp3", "--mp3-preset", "2", source_path], check=True)

    except Exception as e:
        print(f"Failed to process file {source_path}: {e}")
        failed.append(source_path)

# Write failed files to JSON
if failed:
    failed_dict = {"data": failed}
    with open("failed2.json", "w") as outfile:
        json.dump(failed_dict, outfile, indent=4)