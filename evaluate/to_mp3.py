import os
from moviepy import VideoFileClip

video_dir = "./comparison/ours"
audio_dir = "./mp3/ours"
files = sorted(os.listdir(video_dir))
for i in range(len(files)):
    file_idx = files[i].split('.')[0]
    video_file = os.path.join(video_dir, files[i])
    video = VideoFileClip(video_file)
    audio_file = os.path.join(audio_dir, f"{file_idx}.mp3")
    video.audio.write_audiofile(audio_file)