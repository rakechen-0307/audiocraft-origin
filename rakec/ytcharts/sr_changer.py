import os
import json
import argparse

parser = argparse.ArgumentParser(description="modify audio sample rate")
parser.add_argument('--input', '-i', dest="input", required=True, type=str, help="input audio folder")
parser.add_argument('--output', '-o', dest="output", required=True, type=str, help="output audio folder")

args = parser.parse_args()
print("args:")
print(args)

input_path = args.input
output_path = args.output

files = sorted(os.listdir(input_path))
for i in range(len(files)):
    if (files[i].split('.')[-1] == "mp3"):
        os.system("ffmpeg -i {target} -ar 32000 {output}".format(
            target = input_path + "/" + files[i],
            output = output_path + "/" + files[i]
        ))
    else:
        os.system("cp {target} {output}".format(
            target = input_path + "/" + files[i],
            output = output_path + "/" + files[i]
        ))
        with open(output_path + "/" + files[i], 'r') as f:
            data = json.load(f)
            data["sample_rate"] = 32000  # change sample rate

        os.remove(output_path + "/" + files[i])
        with open(output_path + "/" + files[i], 'w') as f:
            json.dump(data, f, indent=4)