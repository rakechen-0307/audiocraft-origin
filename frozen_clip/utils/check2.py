import os

list_txt = "../list.txt"
file_dir = "../clipclap/train/videos"
files = sorted(os.listdir(file_dir))
id_file = open(list_txt, "r+")  
id_list = id_file.readlines()
for id in id_list:
    id = id.replace(" ", "")
    id = id.replace("\n", "")
    if id not in files:
        print(id)