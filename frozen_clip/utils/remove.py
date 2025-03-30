import os

list = ['1-19-92knlV0KiBY_13.mp3', '1-19-92knlV0KiBY_14.mp3', '1-19-92knlV0KiBY_15.mp3', '1-19-92knlV0KiBY_16.mp3', '1-19-92knlV0KiBY_17.mp3', '1-19-92knlV0KiBY_20.mp3', '1-19-92knlV0KiBY_21.mp3', '3-13-Ae5WtA_Oqfs_3.mp3']
dir = "../spotify-small/clap_spotify-small/valid"
for i in range(len(list)):
    id = list[i].split('.')[0]
    os.system("rm {}".format(os.path.join(dir, f"{id}.json")))
    os.system("rm {}".format(os.path.join(dir, f"{id}.mp3")))
    os.system("rm {}".format(os.path.join(dir, f"{id}.mp4")))