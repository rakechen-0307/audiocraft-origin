import os
import json
import argparse

parser = argparse.ArgumentParser(description="modify json files")
parser.add_argument('--folder', '-f', dest="folder", required=True, type=str, help="json folder")

args = parser.parse_args()
print("args:")
print(args)

json_path = args.folder
files = sorted(os.listdir(json_path))
for i in range(len(files)):
    if (files[i].split('.')[-1] == "json"):
        filename = json_path + "/" + files[i]
        with open(filename, 'r') as f:
            data = json.load(f)
            data["sample_rate"] = 32000  # change sample rate

        os.remove(filename)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)