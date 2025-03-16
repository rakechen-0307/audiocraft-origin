import os

dir = "../struct"
files = sorted(os.listdir(dir))

for i in range(len(files)):
    if (int(files[i].split('.')[0]) != i):
        print(i)
        break
