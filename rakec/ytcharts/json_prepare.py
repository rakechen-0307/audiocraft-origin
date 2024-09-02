import os
import argparse

parser = argparse.ArgumentParser(description="handle json file")
parser.add_argument('--path_json', '-pj', dest="path_json", required=True, type=str, help="json data path")
parser.add_argument('--output', '-o', dest="output", required=True, type=str, help="output path")

args = parser.parse_args()
print("args:")
print(args)

json_path = args.path_json
output_path = args.output

files = sorted(os.listdir(json_path))
for i in range(len(files)):
    if (files[i].split('.')[-1] == 'json'):
        os.system('cp {target} {output}'.format(
            target = json_path + "/" + files[i],
            output = output_path + "/" + files[i]
        ))