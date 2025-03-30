import os

audio_dir = "../clipclap/train/audios"
video_dir = "../clipclap/train/videos"
dirs = sorted(os.listdir(audio_dir))
for i in range(len(dirs)):
    if (len(os.listdir(os.path.join(audio_dir, dirs[i]))) != len(os.listdir(os.path.join(video_dir, dirs[i])))):
        print(dirs[i])