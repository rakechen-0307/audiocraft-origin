import os

files = sorted(os.listdir("../originals"))
for i in range(len(files)):
    file = os.path.join("../originals", files[i])
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
    os.system("cp {target} {output}".format(
        target=file,
        output=os.path.join("../audios", files[i])
    ))
    os.system("mv {target} {output}".format(
        target=os.path.join("../audios", files[i]),
        output=os.path.join("../audios", num+".mp3")
    ))