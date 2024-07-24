import os
import argparse
import tqdm

parser = argparse.ArgumentParser(description="handle ytcharts dataset")
parser.add_argument('--path_mp3', '-pm', dest="path_mp3", required=True, type=str, help="mp3 data path")
parser.add_argument('--path_json', '-pj', dest="path_json", required=True, type=str, help="json data path")
parser.add_argument('--output', '-o', dest="output", required=True, type=str, help="output path")

args = parser.parse_args()
print("args:")
print(args)

mp3_path = args.path_mp3
json_path = args.path_json
output_path = args.output

mp3_dirs = sorted(os.listdir(mp3_path))
for i in tqdm(range(len(mp3_dirs))):
    ytid = mp3_dirs[i]
    # move audio file
    os.system('cp {target} {output}'.format(
        target = mp3_path + "/" + ytid + "/" + "no_vocals.mp3",
        output = output_path + "/" + ytid + ".mp3"
    ))
    # move json file
    os.system('cp {target} {output}'.format(
        target = json_path + "/" + ytid + ".json",
        output = output_path + "/" + ytid + ".json"
    ))