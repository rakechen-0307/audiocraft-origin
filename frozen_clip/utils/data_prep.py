import os
import json
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pydub import AudioSegment

struct_path = "./struct"
audio_path = "../audios"
video_path = "../videos"
output_path = "../clipclap"

files = sorted(os.listdir(struct_path))
for i in tqdm(range(1550)):
# for i in tqdm(range(1974, 1975)):
    filename = files[i].split(".")[0]
    struct_file = os.path.join(struct_path, f"{filename}.json")
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
    audio = AudioSegment.from_file(audio_file)
    count = 0
    for j in range(len(segments)):
        if (segments[j]['end'] - segments[j]['start'] >= 10):
            split_list = []
            if (segments[j]['end'] - segments[j]['start'] >= 30):
                split_list = [
                    segments[j]['start'] + (1/3) * (segments[j]['end'] - segments[j]['start']),
                    segments[j]['start'] + (2/3) * (segments[j]['end'] - segments[j]['start'])
                ]
            elif (segments[j]['end'] - segments[j]['start'] >= 15):
                split_list = [
                    segments[j]['start'] + (1/2) * (segments[j]['end'] - segments[j]['start'])
                ]
            
            if (j + 1 < len(segments) and segments[j + 1]['end'] - segments[j + 1]['start'] >= 10):
                split_list.append(segments[j + 1]['start'])

            for split_time in split_list:
                if (count < 10):
                    num = "00" + str(count)
                elif (count < 100):
                    num = "0" + str(count)
                else: 
                    num = str(count)
            
                audio_output = os.path.join(audio_output_dir, f"{num}.mp3")
                video_output = os.path.join(video_output_dir, f"{num}.mp4")
                audio_seg = audio[(split_time-5.0)*1000:(split_time+5.0)*1000+1]
                audio_seg.export(audio_output, format="mp3")
                ffmpeg_extract_subclip(video_file, split_time-5.0, split_time+5.0, targetname=video_output)

                count += 1