import os
import json
from tqdm import tqdm

struct_path = "./struct"
audio_path = "../data/valid/audios"
video_path = "../data/valid/videos"
output_path = "../clipclap/valid"

files = sorted(os.listdir(audio_path))
for i in tqdm(range(len(files))):
    filename = files[i].split(".")[0]
    idx = filename[5:]
    struct_file = os.path.join(struct_path, f"{idx}.json")
    audio_file = os.path.join(audio_path, f"{filename}.mp3")
    video_file = os.path.join(video_path, f"{filename}.mp4")

    # create folders
    audio_output_dir = os.path.join(output_path, "audios", filename)
    video_output_dir = os.path.join(output_path, "videos", filename)
    os.makedirs(audio_output_dir)
    os.makedirs(video_output_dir)

    # read json
    with open(struct_file, 'r') as sf:
        segments = json.load(sf)['segments']

    # separate data
    count = 0
    for j in range(len(segments)):
        if (segments[j]['label'] == 'bridge' or segments[j]['label'] == 'inst' or 
            segments[j]['label'] == 'solo' or segments[j]['label'] == 'verse' or 
            segments[j]['label'] == 'chorus'):
            split_list = []
            if (segments[j]['end'] - segments[j]['start'] >= 60):
                split_list = [
                    segments[j]['start'] + 2,
                    (segments[j]['start'] + segments[j]['end']) / 2 - 4,
                    segments[j]['end'] - 12
                ]
            elif (segments[j]['end'] - segments[j]['start'] >= 30):
                split_list = [
                    segments[j]['start'] + 2,
                    segments[j]['end'] - 12
                ]
            elif (segments[j]['end'] - segments[j]['start'] >= 12):
                split_list = [
                    segments[j]['start'] + 2
                ]

            for split_time in split_list:
                if (count < 10):
                    num = "00" + str(count)
                elif (count < 100):
                    num = "0" + str(count)
                else: 
                    num = str(count)
            
                audio_output = os.path.join(audio_output_dir, f"{num}.mp3")
                video_output = os.path.join(video_output_dir, f"{num}.mp4")
                os.system("ffmpeg -i {target} -ss {time} -t 8 -c:a libmp3lame -q:a 0 {output}".format(
                    target=audio_file,
                    time=split_time,
                    output=audio_output
                ))
                os.system("ffmpeg -i {target} -ss {time} -t 8 -c:v libx264 -preset fast -crf 23 -an {output}".format(
                    target=video_file,
                    time=split_time,
                    output=video_output
                ))

                count += 1
    
    os.system("rm {}".format(audio_file))
    os.system("rm {}".format(video_file))