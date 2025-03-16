import os
import cv2
from moviepy.editor import *

files = sorted(os.listdir("../originals"))

for i in range(len(files)):
    origin = "../originals/" + files[i]
    cap = cv2.VideoCapture(origin)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'Checking Video {i+1} Frames {frames} fps: {fps}')

    if (frames != 0):
        video = VideoFileClip(origin)

        if (i < 10):
            num = "0000" + str(i)
        elif (i < 100):
            num = "000" + str(i)
        elif (i < 1000):
            num = "00" + str(i)
        elif (i < 10000):
            num = "0" + str(i)
        else:
            num = str(i)

        ## extract video
        video_output = "../videos/" + num + ".mp4"
        video = video.without_audio()
        video.write_videofile(video_output)