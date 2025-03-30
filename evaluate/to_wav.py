import os

audio_dirs = "./samples/audios"
files = sorted(os.listdir(audio_dirs))
for i in range(len(files)):
    if (files[i].split('.')[-1] == "mp3"):
        audio_file = os.path.join(audio_dirs, files[i])
        output_file = os.path.join(audio_dirs, f"{files[i].split('.')[0]}.wav")

        os.system("ffmpeg -i {input} {output}".format(
            input=audio_file,
            output=output_file
        ))
        os.system("rm {}".format(audio_file))