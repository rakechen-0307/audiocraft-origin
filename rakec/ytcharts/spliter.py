import os
import argparse

parser = argparse.ArgumentParser(description="split train & test")
parser.add_argument('--ref', '-r', dest="ref", required=True, type=str, help="reference path")
parser.add_argument('--target', '-t', dest="target", required=True, type=str, help="target path")
parser.add_argument('--output', '-o', dest="output", required=True, type=str, help="output path")

args = parser.parse_args()
print("args:")
print(args)

ref_path = args.ref
target_path = args.target
output_path = args.output

files = sorted(os.listdir(ref_path))
for i in range(len(files)):
    if (files[i].split('.')[-1] == 'json'):
        ytid = files[i].split('.')[0]
        # move mp3
        os.system("mv {target} {output}".format(
            target = target_path + "/" + ytid + ".mp3",
            output = output_path
        ))
        # move json
        os.system("mv {target} {output}".format(
            target = target_path + "/" + ytid + ".json",
            output = output_path
        ))