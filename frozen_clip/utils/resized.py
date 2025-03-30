import os
from tqdm import tqdm
from moviepy import VideoFileClip

resized_h = 600
resized_w = 800

dir = "../spotify-small/clip_spotify-small/train"
output_dir = "../resized"
files = sorted(os.listdir(dir))
for i in tqdm(range(len(files))):
    if (files[i].split('.')[-1] == "mp4"):
        video_file = os.path.join(dir, files[i])
        video = VideoFileClip(video_file)
        resized_video = video.resized((resized_w, resized_h))
        output_file = os.path.join(output_dir, f"{files[i].split('.')[0]}.mp4")
        resized_video.write_videofile(output_file)