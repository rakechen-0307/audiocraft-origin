import os
import json
from tqdm import tqdm
from pydub import AudioSegment

seg_length = 40
stride = 20
offset = 20

data_dir = "../data/test"
output_dir = "../clip_spotify-small/test"
files = sorted(os.listdir(data_dir))
audio_files = []
for i in range(len(files)):
    if (files[i].split(".")[-1] == "mp3"):
        audio_files.append(files[i])

for i in tqdm(range(len(audio_files))):
    ytid = audio_files[i].split('.')[0]
    audio_file = os.path.join(data_dir, f"{ytid}.mp3")
    video_file = os.path.join(data_dir, f"{ytid}.mp4")

    count = 1
    audio = AudioSegment.from_mp3(audio_file)
    audio_length = audio.duration_seconds
    start = offset
    while (start <= audio_length - offset - seg_length):
        audio_outfile = os.path.join(output_dir, f"{ytid}_{count}.mp3")
        video_outfile = os.path.join(output_dir, f"{ytid}_{count}.mp4")

        os.system("ffmpeg -i {target} -ss {start} -t {time} -c:a libmp3lame -q:a 0 {output}".format(
            target=audio_file,
            start=start,
            time=seg_length,
            output=audio_outfile
        ))
        os.system("ffmpeg -i {target} -ss {start} -t {time} -c:v libx264 -preset fast -crf 23 -an {output}".format(
            target=video_file,
            start=start,
            time=seg_length,
            output=video_outfile
        ))

        count += 1
        start += stride