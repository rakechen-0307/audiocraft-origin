import os

audio_dir = "../clipclap/valid/audios"
video_dir = "../clipclap/valid/videos"
dirs = sorted(os.listdir(audio_dir))
for i in range(len(dirs)):
    files = sorted(os.listdir(os.path.join(audio_dir, dirs[i])))
    if (len(files) == 0):
        print(f"{dirs[i]} is empty")
        os.system("rm {}".format(os.path.join(audio_dir, dirs[i])))
        os.system("rm {}".format(os.path.join(video_dir, dirs[i])))
    elif (int(files[-1].split('.')[0]) != len(files) - 1):
        print(f"modify {dirs[i]}")
        for j in range(len(files)):
            if (j < 10):
                num = "00" + str(j)
            elif (j < 100):
                num = "0" + str(j)
            else: 
                num = str(j)

            os.system("mv {target} {output}".format(
                target=os.path.join(audio_dir, dirs[i], files[j]),
                output=os.path.join(audio_dir, dirs[i], f"{num}.mp3")
            ))
            os.system("mv {target} {output}".format(
                target=os.path.join(video_dir, dirs[i], f"{files[j].split('.')[0]}.mp4"),
                output=os.path.join(video_dir, dirs[i], f"{num}.mp4")
            ))
